import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchmetrics import MeanAbsoluteError, R2Score
from torchvision.models import ViT_B_16_Weights, vit_b_16

from audio_config import MAX_FRAMES, N_MELS
from training.data_utils import (
    MelPopularityDataset,
    checkpoint_matches_model,
    load_popularity_by_id,
    spec_augment,
    sync_to_local_scratch,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PopularityViT(nn.Module):
    """ImageNet-pretrained Audio Spectrogram Transformer regressor.

    This follows Gong et al. (2021): adapt a ViT-B/16 encoder to a
    single-channel spectrogram, use overlapping patches, resize its 2-D
    positional embeddings, replace the task head, and fine-tune end to end.
    """

    def __init__(
        self,
        image_size=(N_MELS, MAX_FRAMES),
        patch_size=(16, 16),
        patch_stride=(10, 10),
        dropout=0.1,
        weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,
    ):
        super().__init__()
        if patch_size != (16, 16):
            raise ValueError("pretrained ViT-B/16 requires 16x16 patches")

        pretrained_vit = vit_b_16(weights=weights, dropout=dropout)
        pretrained_projection = pretrained_vit.conv_proj
        self.patch_embedding = nn.Conv2d(
            1,
            pretrained_projection.out_channels,
            kernel_size=patch_size,
            stride=patch_stride,
        )
        # AST averages the RGB kernels to transfer the image patch projection.
        # Summing would be equivalent to copying a mono spectrogram into RGB;
        # averaging follows the paper exactly and keeps the inherited scale tame.
        with torch.no_grad():
            self.patch_embedding.weight.copy_(
                pretrained_projection.weight.mean(dim=1, keepdim=True)
            )
            self.patch_embedding.bias.copy_(pretrained_projection.bias)

        self.class_token = pretrained_vit.class_token
        self.encoder = pretrained_vit.encoder
        self.grid_size = (
            1 + (image_size[0] - patch_size[0]) // patch_stride[0],
            1 + (image_size[1] - patch_size[1]) // patch_stride[1],
        )
        adapted_position_embedding = self._adapt_position_embedding(
            self.encoder.pos_embedding,
            source_grid_size=pretrained_vit.image_size // pretrained_vit.patch_size,
            target_grid_size=self.grid_size,
        )
        self.encoder.pos_embedding = nn.Parameter(adapted_position_embedding)
        self.head = nn.Linear(pretrained_vit.hidden_dim, 1)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _adapt_position_embedding(
        position_embedding, source_grid_size, target_grid_size
    ):
        """Bilinearly resize ViT's 2-D positions while preserving [CLS]."""
        class_position = position_embedding[:, :1]
        patch_positions = position_embedding[:, 1:]
        hidden_dim = patch_positions.shape[-1]
        patch_positions = patch_positions.reshape(
            1, source_grid_size, source_grid_size, hidden_dim
        ).permute(0, 3, 1, 2)
        patch_positions = F.interpolate(
            patch_positions,
            size=target_grid_size,
            mode="bilinear",
            align_corners=False,
        )
        patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(
            1, target_grid_size[0] * target_grid_size[1], hidden_dim
        )
        return torch.cat((class_position, patch_positions), dim=1)

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        expected_patches = self.grid_size[0] * self.grid_size[1]
        if x.shape[1] != expected_patches:
            raise ValueError(
                "input produced a different number of patches than image_size; "
                "pad or crop it with collate_fn"
            )
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.encoder(x)
        return torch.sigmoid(self.head(x[:, 0])).squeeze(1)


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
    scaler=None,
    accumulation_steps=1,
):
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
                mels = spec_augment(mels, frequency_dim=2, time_dim=3)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                predictions = model(mels)
                loss = criterion(predictions, targets)

            if is_train:
                should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
                if scaler is not None:
                    scaler.scale(loss / accumulation_steps).backward()
                    if should_step:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    (loss / accumulation_steps).backward()
                    if should_step:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                if should_step:
                    optimizer.zero_grad()
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

    batch_size = 12
    accumulation_steps = 4
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
    learning_rate = 5e-5
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    warmup_epochs = 3
    warmup_steps = max(1, warmup_epochs * ((len(train_loader) + accumulation_steps - 1) // accumulation_steps))
    warmup_scheduler = LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
    )
    mae_metric = MeanAbsoluteError().to(device)
    r2_metric = R2Score().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    epochs = 40
    hyperparams = {
        "batch_size": batch_size,
        "effective_batch_size": batch_size * accumulation_steps,
        "accumulation_steps": accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": 1e-4,
        "epochs": epochs,
        "image_size": [N_MELS, MAX_FRAMES],
        "patch_size": [16, 16],
        "patch_stride": [10, 10],
        "patch_overlap": [6, 6],
        "encoder": "vit_b_16",
        "pretrained_weights": "IMAGENET1K_SWAG_E2E_V1",
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "mlp_dim": 3072,
        "dropout": 0.1,
        "warmup_epochs": warmup_epochs,
        "fine_tune_encoder": True,
    }
    last_checkpoint_path = os.path.join(
        checkpoint_dir, "last_ViT_checkpoint.pt"
    )
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_ViT_model.pt")
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
        train_loss, train_mae, train_r2, train_pred_mean, train_pred_std = run_epoch(
            model,
            train_loader,
            criterion,
            mae_metric,
            r2_metric,
            optimizer,
            warmup_scheduler if epoch <= warmup_epochs else None,
            scaler,
            accumulation_steps,
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
