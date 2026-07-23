import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchmetrics import MeanAbsoluteError, R2Score

from audio_config import MAX_FRAMES, N_MELS
from training.data_utils import (
    MelPopularityDataset,
    checkpoint_matches_model,
    load_popularity_by_id,
    sync_to_local_scratch,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchEmbedding(nn.Module):
    """Turn a mel spectrogram into a sequence of non-overlapping 2-D patches."""

    def __init__(self, patch_size=(16, 32), embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            1, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        frequency_pad = (-x.shape[-2]) % self.patch_size[0]
        time_pad = (-x.shape[-1]) % self.patch_size[1]
        if frequency_pad or time_pad:
            x = F.pad(x, (0, time_pad, 0, frequency_pad))
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class PopularityViT(nn.Module):
    """Vision Transformer regressor for normalized song popularity."""

    def __init__(
        self,
        image_size=(N_MELS, MAX_FRAMES),
        patch_size=(16, 32),
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        mlp_dim=1024,
        dropout=0.1,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.patch_embedding = PatchEmbedding(patch_size, embed_dim)
        grid_height = (image_size[0] + patch_size[0] - 1) // patch_size[0]
        grid_width = (image_size[1] + patch_size[1] - 1) // patch_size[1]
        num_patches = grid_height * grid_width

        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim)
        )
        self.head = nn.Linear(embed_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embedding.projection.weight, std=0.02)
        if self.patch_embedding.projection.bias is not None:
            nn.init.zeros_(self.patch_embedding.projection.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.patch_embedding(x)
        if x.shape[1] + 1 != self.position_embedding.shape[1]:
            raise ValueError(
                "input produced a different number of patches than image_size; "
                "pad or crop it with collate_fn"
            )
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.embedding_dropout(x + self.position_embedding)
        x = self.encoder(x)
        return self.head(x[:, 0]).squeeze(1)


def to_max_frames(mel):
    if mel.shape[-1] < MAX_FRAMES:
        return F.pad(mel, (0, MAX_FRAMES - mel.shape[-1]))
    return mel[..., :MAX_FRAMES]


def collate_fn(batch):
    mels, targets = zip(*batch)
    mels = torch.stack([to_max_frames(mel) for mel in mels]).unsqueeze(1)
    return mels, torch.stack(targets)


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

    average_loss = total_loss / len(loader.dataset)
    prediction_mean = prediction_sum / prediction_count
    prediction_variance = max(
        0.0, prediction_squared_sum / prediction_count - prediction_mean**2
    )
    return (
        average_loss,
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

    batch_size = 32
    loader_options = {
        "batch_size": batch_size,
        "num_workers": 16,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)

    model = PopularityViT().to(device)
    criterion = nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
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
    early_stopping_patience = 8
    hyperparams = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 1e-4,
        "epochs": epochs,
        "image_size": [N_MELS, MAX_FRAMES],
        "patch_size": [16, 32],
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 4,
        "mlp_dim": 1024,
        "dropout": 0.1,
        "warmup_epochs": warmup_epochs,
        "early_stopping_patience": early_stopping_patience,
    }
    last_checkpoint_path = os.path.join(
        checkpoint_dir, "last_ViT_checkpoint.pt"
    )
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_ViT_model.pt")
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
        train_loss, train_mae, train_r2, train_pred_mean, train_pred_std = run_epoch(
            model,
            train_loader,
            criterion,
            mae_metric,
            r2_metric,
            optimizer,
            warmup_scheduler if epoch <= warmup_epochs else None,
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
        is_best = val_loss < best_val_loss
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
            "hyperparams": hyperparams,
        }
        torch.save(checkpoint, last_checkpoint_path)
        if is_best:
            torch.save(checkpoint, best_checkpoint_path)
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"early stopping after {early_stopping_patience} epochs "
                "without validation improvement",
                flush=True,
            )
            break

    print(
        f"finished epoch {epoch}; best val loss {best_val_loss:.4f}; "
        f"latest checkpoint saved to {last_checkpoint_path}"
    )


if __name__ == "__main__":
    main()
