import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torchmetrics import MeanAbsoluteError, R2Score
from training.data_utils import (
    MelPopularityDataset,
    checkpoint_matches_model,
    load_popularity_by_id,
    sync_to_local_scratch,
)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

network_mel_dir = os.path.join("/net/scc1/scratch", os.environ["USER"], "mel_spectrograms")
local_scratch_dir = os.path.join("/scratch", os.environ["USER"])
checkpoint_dir = os.path.join("/net/scc1/scratch", os.environ["USER"], "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)




mel_dir = sync_to_local_scratch(network_mel_dir, local_scratch_dir)


class PopularityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Summarize to a fixed size regardless of spectrogram length.
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(512 * 4 * 4, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))  # popularity normalized to [0, 1]
        return x.squeeze(1)


def run_epoch(model, loader, criterion, mae_metric, r2_metric, optimizer=None, warmup_scheduler=None):
    is_train = optimizer is not None
    model.train(is_train)
    mae_metric.reset()
    r2_metric.reset()
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for mels, targets in loader:
            mels = mels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

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
            detached_predictions = predictions.detach()
            mae_metric.update(detached_predictions * 100, targets * 100)
            r2_metric.update(detached_predictions * 100, targets * 100)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, mae_metric.compute().item(), r2_metric.compute().item()

def collate_fn(batch):
    mels, targets = zip(*batch)
    max_frames = max(mel.shape[-1] for mel in mels)
    padded = torch.stack(
        [F.pad(mel, (0, max_frames - mel.shape[-1])) for mel in mels]
    )
    padded = padded.unsqueeze(1)  # (B, 1, n_mels, time)
    targets = torch.stack(targets)
    return padded, targets

def main():
    torch.manual_seed(0)

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

    # Early convolutional feature maps retain the full spectrogram resolution
    # and are large during backpropagation.
    batch_size = 16
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = PopularityCNN().to(device)
    criterion = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    warmup_epochs = 3
    warmup_steps = max(1, warmup_epochs * len(train_loader))
    warmup_scheduler = LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
    )
    mae_metric = MeanAbsoluteError().to(device)
    r2_metric = R2Score().to(device)

    epochs = 150
    hyperparams = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "conv_channels": [64, 128, 256, 512],
        "fc_dims": [2048, 256],
        "warmup_epochs": warmup_epochs,
    }
    last_checkpoint_path = os.path.join(checkpoint_dir, "last_CNN_checkpoint.pt")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_CNN_model.pt")
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

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        train_loss, train_mae, train_r2 = run_epoch(
            model, train_loader, criterion, mae_metric, r2_metric, optimizer,
            warmup_scheduler if epoch <= warmup_epochs else None,
        )
        val_loss, val_mae, val_r2 = run_epoch(
            model, val_loader, criterion, mae_metric, r2_metric
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} mae {train_mae:.2f} r2 {train_r2:.3f} | "
            f"val loss {val_loss:.4f} mae {val_mae:.2f} r2 {val_r2:.3f}"
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
