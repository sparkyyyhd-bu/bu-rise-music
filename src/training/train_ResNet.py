import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchmetrics import MeanAbsoluteError, R2Score

from training.data_utils import (
    MelPopularityDataset,
    checkpoint_matches_model,
    load_popularity_by_id,
    spec_augment,
    sync_to_local_scratch,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Bottleneck(nn.Module):
    """1x1, 3x3, 1x1 residual bottleneck used by ResNet-50."""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        expanded_channels = out_channels * self.expansion
        if stride != 1 or in_channels != expanded_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    expanded_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(expanded_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))
        return F.relu(x + identity, inplace=True)


class PopularityResNet(nn.Module):
    """ResNet-50 regressor adapted for single-channel mel spectrograms."""

    def __init__(self, layers=(3, 4, 6, 3), dropout=0.3):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(512 * Bottleneck.expansion, 1)
        self._initialize_weights()

    def _make_layer(self, out_channels, num_blocks, stride=1):
        blocks = [Bottleneck(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * Bottleneck.expansion
        blocks.extend(
            Bottleneck(self.in_channels, out_channels)
            for _ in range(1, num_blocks)
        )
        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).flatten(1)
        return torch.sigmoid(self.head(self.dropout(x))).squeeze(1)


def collate_fn(batch):
    mels, targets = zip(*batch)
    max_frames = max(mel.shape[-1] for mel in mels)
    padded = torch.stack(
        [F.pad(mel, (0, max_frames - mel.shape[-1])) for mel in mels]
    )
    return padded.unsqueeze(1), torch.stack(targets)


def run_epoch(
    model,
    loader,
    criterion,
    mae_metric,
    r2_metric,
    optimizer=None,
    warmup_scheduler=None,
):
    is_train = optimizer is not None
    model.train(is_train)
    mae_metric.reset()
    r2_metric.reset()
    total_loss = 0.0
    prediction_sum = 0.0
    prediction_squared_sum = 0.0
    prediction_count = 0

    with torch.set_grad_enabled(is_train):
        for mels, targets in loader:
            mels = mels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if is_train:
                mels = spec_augment(mels, frequency_dim=2, time_dim=3)

            predictions = model(mels)
            loss = criterion(predictions, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if warmup_scheduler is not None:
                    warmup_scheduler.step()

            total_loss += loss.item() * mels.size(0)
            predictions = predictions.detach()
            prediction_sum += predictions.sum().item()
            prediction_squared_sum += predictions.square().sum().item()
            prediction_count += predictions.numel()
            mae_metric.update(predictions * 100, targets * 100)
            r2_metric.update(predictions * 100, targets * 100)

    prediction_mean = prediction_sum / prediction_count
    prediction_variance = max(
        0.0, prediction_squared_sum / prediction_count - prediction_mean**2
    )
    return (
        total_loss / len(loader.dataset),
        mae_metric.compute().item(),
        r2_metric.compute().item(),
        prediction_mean * 100,
        prediction_variance**0.5 * 100,
    )


def main():
    torch.manual_seed(0)

    network_mel_dir = os.path.join(
        "/net/scc1/scratch", os.environ["USER"], "mel_spectrograms"
    )
    local_scratch_dir = os.path.join("/scratch", os.environ["USER"])
    checkpoint_dir = os.path.join(
        "/net/scc1/scratch", os.environ["USER"], "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    mel_dir = sync_to_local_scratch(network_mel_dir, local_scratch_dir)

    dataset = MelPopularityDataset(mel_dir, load_popularity_by_id())
    print(f"loaded {len(dataset)} labeled mel spectrograms")
    val_size = max(1, int(0.15 * len(dataset)))
    test_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size - test_size
    train_set, val_set, _ = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(0),
    )

    batch_size = 16
    loader_options = {
        "batch_size": batch_size,
        "num_workers": 16,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)

    model = PopularityResNet().to(device)
    criterion = nn.MSELoss()
    learning_rate = 1e-3
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    warmup_epochs = 3
    warmup_steps = max(1, warmup_epochs * len(train_loader))
    warmup_scheduler = LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
    )
    mae_metric = MeanAbsoluteError().to(device)
    r2_metric = R2Score().to(device)

    epochs = 60
    hyperparams = {
        "architecture": "ResNet-50",
        "layers": [3, 4, 6, 3],
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "dropout": 0.3,
        "warmup_epochs": warmup_epochs,
    }
    last_checkpoint_path = os.path.join(
        checkpoint_dir, "last_ResNet_checkpoint.pt"
    )
    best_checkpoint_path = os.path.join(
        checkpoint_dir, "best_ResNet_model.pt"
    )
    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(
            last_checkpoint_path, map_location=device, weights_only=True
        )
        compatible, reason = checkpoint_matches_model(model, checkpoint)
        if compatible:
            model.load_state_dict(checkpoint["model_state_dict"])
            for state_name, stateful in (
                ("optimizer_state_dict", optimizer),
                ("scheduler_state_dict", scheduler),
                ("warmup_scheduler_state_dict", warmup_scheduler),
            ):
                try:
                    stateful.load_state_dict(checkpoint[state_name])
                except (KeyError, ValueError, RuntimeError) as error:
                    print(f"could not restore {state_name}: {error}")
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint["best_val_loss"]
            hyperparams = checkpoint.get("hyperparams", hyperparams)
            print(
                f"continuing from epoch {start_epoch} in {last_checkpoint_path}",
                flush=True,
            )
            print(f"hyperparameters: {hyperparams}", flush=True)
        else:
            print(
                f"checkpoint at {last_checkpoint_path} doesn't match the current "
                f"model architecture ({reason}); starting fresh",
                flush=True,
            )

    epoch = start_epoch
    for epoch in range(start_epoch + 1, epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            criterion,
            mae_metric,
            r2_metric,
            optimizer,
            warmup_scheduler if epoch <= warmup_epochs else None,
        )
        val_stats = run_epoch(
            model, val_loader, criterion, mae_metric, r2_metric
        )
        train_loss, train_mae, train_r2, train_mean, train_std = train_stats
        val_loss, val_mae, val_r2, val_mean, val_std = val_stats
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} mae {train_mae:.2f} "
            f"r2 {train_r2:.3f} pred {train_mean:.1f}±{train_std:.1f} | "
            f"val loss {val_loss:.4f} mae {val_mae:.2f} "
            f"r2 {val_r2:.3f} pred {val_mean:.1f}±{val_std:.1f}",
            flush=True,
        )

        if epoch > warmup_epochs:
            scheduler.step(val_loss)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "hyperparams": hyperparams,
        }
        torch.save(checkpoint, last_checkpoint_path)
        if is_best:
            torch.save(checkpoint, best_checkpoint_path)

    print(
        f"finished epoch {epoch}; best val loss {best_val_loss:.4f}; "
        f"latest checkpoint saved to {last_checkpoint_path}"
    )


if __name__ == "__main__":
    main()
