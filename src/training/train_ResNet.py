import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchmetrics import MeanAbsoluteError, R2Score
from torchvision.models import ResNet34_Weights, resnet34

from training.data_utils import (
    MelPopularityDataset,
    checkpoint_matches_model,
    load_popularity_by_id,
    spec_augment,
    sync_to_local_scratch,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PopularityResNet(nn.Module):
    """ImageNet-pretrained ResNet-34 fine-tuned on mel spectrograms."""

    def __init__(
        self,
        dropout=0.5,
        weights=ResNet34_Weights.IMAGENET1K_V1,
    ):
        super().__init__()
        self.backbone = resnet34(weights=weights)
        self.trainable_stage = "all"
        rgb_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1,
            rgb_conv.out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            self.backbone.conv1.weight.copy_(
                rgb_conv.weight.mean(dim=1, keepdim=True)
            )

        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feature_dim, 1)

    def set_trainable_stage(self, stage):
        """Train the head, then layer4, then the full pretrained backbone."""
        if stage not in {"head", "layer4", "all"}:
            raise ValueError(f"unknown trainable stage: {stage}")
        self.trainable_stage = stage
        for parameter in self.backbone.parameters():
            parameter.requires_grad = stage == "all"
        if stage == "layer4":
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad = True
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if mode and self.trainable_stage != "all":
            # Keep frozen BatchNorm running statistics fixed as well as weights.
            self.backbone.eval()
            if self.trainable_stage == "layer4":
                self.backbone.layer4.train()
        return self

    def forward(self, x):
        x = self.backbone(x)
        return torch.sigmoid(self.head(self.dropout(x))).squeeze(1)


def collate_fn(batch):
    mels, targets = zip(*batch)
    max_frames = max(mel.shape[-1] for mel in mels)
    padded = torch.stack(
        [F.pad(mel, (0, max_frames - mel.shape[-1])) for mel in mels]
    )
    return padded.unsqueeze(1), torch.stack(targets)


def augment_mels(mels):
    """Apply stronger masking plus small waveform-independent perturbations."""
    mels = spec_augment(
        mels,
        frequency_dim=2,
        time_dim=3,
        probability=0.9,
        max_frequency_width=16,
        max_time_width=120,
        frequency_masks=2,
        time_masks=2,
    )
    max_shift = max(1, int(0.05 * mels.shape[-1]))
    shift = int(
        torch.randint(-max_shift, max_shift + 1, (), device=mels.device).item()
    )
    mels = torch.roll(mels, shifts=shift, dims=-1)
    return mels + 0.01 * torch.randn_like(mels)


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
                mels = augment_mels(mels)

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
    head_only_epochs = 3
    layer4_only_epochs = 7
    model.set_trainable_stage("head")
    criterion = nn.MSELoss()
    backbone_learning_rate = 1e-4
    head_learning_rate = 1e-3
    weight_decay = 5e-4
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": backbone_learning_rate,
            },
            {"params": model.head.parameters(), "lr": head_learning_rate},
        ],
        weight_decay=weight_decay,
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
    early_stopping_patience = 10
    early_stopping_min_delta = 1e-4
    hyperparams = {
        "architecture": "ResNet-34",
        "pretrained_weights": "IMAGENET1K_V1",
        "fine_tuning": {
            "head_only_epochs": head_only_epochs,
            "layer4_only_through_epoch": layer4_only_epochs,
        },
        "batch_size": batch_size,
        "backbone_learning_rate": backbone_learning_rate,
        "head_learning_rate": head_learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "dropout": 0.5,
        "warmup_epochs": warmup_epochs,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "augmentation": {
            "probability": 0.9,
            "frequency_masks": 2,
            "max_frequency_width": 16,
            "time_masks": 2,
            "max_time_width": 120,
            "max_time_shift_fraction": 0.05,
            "gaussian_noise_std": 0.01,
        },
    }
    last_checkpoint_path = os.path.join(
        checkpoint_dir, "last_ResNet_checkpoint.pt"
    )
    best_checkpoint_path = os.path.join(
        checkpoint_dir, "best_ResNet_model.pt"
    )
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

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
            epochs_without_improvement = checkpoint.get(
                "epochs_without_improvement", 0
            )
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
        if epoch <= head_only_epochs:
            trainable_stage = "head"
        elif epoch <= layer4_only_epochs:
            trainable_stage = "layer4"
        else:
            trainable_stage = "all"
        if model.trainable_stage != trainable_stage:
            model.set_trainable_stage(trainable_stage)
            print(
                f"epoch {epoch:03d} | trainable stage: {trainable_stage}",
                flush=True,
            )

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
        is_best = val_loss < best_val_loss - early_stopping_min_delta
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "hyperparams": hyperparams,
        }
        torch.save(checkpoint, last_checkpoint_path)
        if is_best:
            torch.save(checkpoint, best_checkpoint_path)
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"early stopping after {early_stopping_patience} epochs "
                "without meaningful validation improvement",
                flush=True,
            )
            break

    print(
        f"finished epoch {epoch}; best val loss {best_val_loss:.4f}; "
        f"latest checkpoint saved to {last_checkpoint_path}"
    )


if __name__ == "__main__":
    main()
