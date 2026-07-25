import torch
import torch.nn as nn
from training.data_utils import (
    MelPopularityDataset,
    augment_mels,
    checkpoint_matches_model,
    load_popularity_by_id,
    sync_to_local_scratch,
)
import os
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torchmetrics import MeanAbsoluteError, R2Score
from audio_config import MAX_FRAMES

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class PopularityCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Pool time as well as frequency. Four stages reduce ~2,584 frames to
        # ~161 useful LSTM steps instead of making recurrence the bottleneck.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            dropout=0.5,
            batch_first=True,
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # (B, C, F, T)
        x = x.mean(dim=2).transpose(1, 2)  # (B, T, C)
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x).squeeze(1)


def run_epoch(model, loader, criterion, mae_metric, r2_metric, optimizer=None, warmup_scheduler=None, accumulation_steps=1):
    is_train = optimizer is not None
    model.train(is_train)
    mae_metric.reset()
    r2_metric.reset()
    total_loss = 0.0
    prediction_sum = 0.0
    prediction_squared_sum = 0.0
    prediction_count = 0

    if is_train:
        optimizer.zero_grad()
    with torch.set_grad_enabled(is_train):
        for batch_index, (mels, targets) in enumerate(loader):
            mels = mels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_train:
                mels = augment_mels(mels)
            predictions = model(mels)
            loss = criterion(predictions, targets)
            if is_train:
                (loss / accumulation_steps).backward()
                should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
                if should_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if warmup_scheduler is not None:
                        warmup_scheduler.step()

            total_loss += loss.item() * mels.size(0)
            detached_predictions = predictions.detach()
            prediction_sum += detached_predictions.sum().item()
            prediction_squared_sum += detached_predictions.square().sum().item()
            prediction_count += detached_predictions.numel()
            mae_metric.update(detached_predictions * 100, targets * 100)
            r2_metric.update(detached_predictions * 100, targets * 100)

    avg_loss = total_loss / len(loader.dataset)
    prediction_mean = prediction_sum / prediction_count
    prediction_variance = max(
        0.0, prediction_squared_sum / prediction_count - prediction_mean**2
    )
    return (
        avg_loss,
        mae_metric.compute().item(),
        r2_metric.compute().item(),
        prediction_mean * 100,
        prediction_variance**0.5 * 100,
    )


def collate_fn(batch):
    mels, targets = zip(*batch)
    padded = torch.stack(
        [
            F.pad(mel, (0, max(0, MAX_FRAMES - mel.shape[-1])))[..., :MAX_FRAMES]
            for mel in mels
        ]
    )
    padded = padded.unsqueeze(1)  # (B, 1, n_mels, time)
    targets = torch.stack(targets)
    return padded, targets


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

    popularity_by_id = load_popularity_by_id()
    dataset = MelPopularityDataset(mel_dir, popularity_by_id)
    print(f"loaded {len(dataset)} labeled mel spectrograms")

    val_size = max(1, int(0.15 * len(dataset)))
    test_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size - test_size
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(0),
    )

    batch_size = 32
    accumulation_steps = 2
    train_loader = DataLoader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = PopularityCNNLSTM().to(device)
    criterion = nn.MSELoss()
    learning_rate = 1e-4
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    warmup_epochs = 3
    warmup_steps = max(1, warmup_epochs * ((len(train_loader) + accumulation_steps - 1) // accumulation_steps))
    warmup_scheduler = LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
    )
    mae_metric = MeanAbsoluteError().to(device)
    r2_metric = R2Score().to(device)

    epochs = 150
    hyperparams = {
        "batch_size": batch_size,
        "effective_batch_size": batch_size * accumulation_steps,
        "accumulation_steps": accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "conv_channels": [32, 64, 128, 256],
        "pool_size": [2, 2],
        "hidden_size": 256,
        "num_layers": 2,
        "fc_dims": [128],
        "dropout": 0.5,
        "augmentation": {
            "probability": 0.9,
            "frequency_masks": 2,
            "max_frequency_width": 16,
            "time_masks": 2,
            "max_time_width": 120,
            "max_time_shift_fraction": 0.05,
            "gaussian_noise_std": 0.01,
        },
        "warmup_epochs": warmup_epochs,
    }
    last_checkpoint_path = os.path.join(checkpoint_dir, "last_CNN_LSTM_checkpoint.pt")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_CNN_LSTM_model.pt")
    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(
            last_checkpoint_path, map_location=device, weights_only=True
        )
        compatible, reason = checkpoint_matches_model(model, checkpoint)
        if compatible:
            model.load_state_dict(checkpoint["model_state_dict"])
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (KeyError, ValueError, RuntimeError) as e:
                print(f"could not restore optimizer state; using a new optimizer ({e})")
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except (KeyError, ValueError, RuntimeError) as e:
                print(f"could not restore scheduler state; using a new scheduler ({e})")
            try:
                warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
            except (KeyError, ValueError, RuntimeError) as e:
                print(f"could not restore warmup scheduler state; using a new warmup scheduler ({e})")
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint["best_val_loss"]
            # Older checkpoints predate hyperparameter logging; assume the
            # hyperparameters currently set in this script were used.
            hyperparams = checkpoint.get("hyperparams", hyperparams)
            print(
                f"continuing from epoch {start_epoch} in {last_checkpoint_path}",
                flush=True,
            )
            print(f"hyperparameters: {hyperparams}", flush=True)
        else:
            print(
                f"checkpoint at {last_checkpoint_path} doesn't match the current "
                f"model architecture ({reason}); starting fresh with the new architecture",
                flush=True,
            )

    epoch = start_epoch
    for epoch in range(start_epoch + 1, epochs + 1):
        train_loss, train_mae, train_r2, train_pred_mean, train_pred_std = run_epoch(
            model, train_loader, criterion, mae_metric, r2_metric, optimizer,
            warmup_scheduler if epoch <= warmup_epochs else None, accumulation_steps,
        )
        val_loss, val_mae, val_r2, val_pred_mean, val_pred_std = run_epoch(
            model, val_loader, criterion, mae_metric, r2_metric
        )
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} mae {train_mae:.2f} r2 {train_r2:.3f} "
            f"pred {train_pred_mean:.1f}±{train_pred_std:.1f} | "
            f"val loss {val_loss:.4f} mae {val_mae:.2f} r2 {val_r2:.3f} "
            f"pred {val_pred_mean:.1f}±{val_pred_std:.1f}",
            flush=True,
        )

        if epoch > warmup_epochs:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            is_best = True
        else:
            is_best = False

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
