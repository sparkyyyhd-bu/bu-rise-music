import torch
import torch.nn as nn
from training.data_utils import (
    MelPopularityDataset,
    checkpoint_matches_model,
    load_popularity_by_id,
    sync_to_local_scratch,
)
import os
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, R2Score
from audio_config import N_MELS, MAX_FRAMES

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

network_mel_dir = os.path.join("/net/scc1/scratch", os.environ["USER"], "mel_spectrograms")
local_scratch_dir = os.path.join("/scratch", os.environ["USER"])
checkpoint_dir = os.path.join("/net/scc1/scratch", os.environ["USER"], "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
mel_dir = sync_to_local_scratch(network_mel_dir, local_scratch_dir)


D_MODEL = 768


class PopularityTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(N_MELS, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = D_MODEL,
            nhead = 8,
            dim_feedforward=3072,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, MAX_FRAMES, D_MODEL) * 0.02
        )
        self.fc = nn.Linear(D_MODEL, 1)

    def forward(self,x):
        x = self.input_proj(x)
        positions = self.position_embedding[:,:x.shape[1],:]
        x = x + positions
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = torch.sigmoid(self.fc(x))
        return x.squeeze(1)


def run_epoch(model, loader, criterion, mae_metric, r2_metric, optimizer = None):
    is_train = optimizer is not None
    model.train(is_train)
    mae_metric.reset()
    r2_metric.reset()
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for mels, targets in loader:
            mels = mels.to(device, non_blocking = True)
            targets = targets.to(device, non_blocking = True)

            predictions = model(mels)
            loss = criterion(predictions, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * mels.size(0)
            detached_predictions = predictions.detach()
            mae_metric.update(detached_predictions * 100, targets * 100)
            r2_metric.update(detached_predictions * 100, targets * 100)
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, mae_metric.compute().item(), r2_metric.compute().item()


def to_max_frames(mel):
    num_frames = mel.shape[-1]
    if num_frames < MAX_FRAMES:
        return F.pad(mel, (0, MAX_FRAMES - num_frames))
    return mel[..., :MAX_FRAMES]


def collate_fn(batch):
    mels, targets = zip(*batch)
    padded = torch.stack([to_max_frames(mel) for mel in mels])
    padded = padded.transpose(1, 2)  # (B, time, n_mels)
    targets = torch.stack(targets)
    return padded, targets


def main():
    torch.manual_seed(0)

    popularity_by_id = load_popularity_by_id()
    epochs = 100
    dataset = MelPopularityDataset(mel_dir, popularity_by_id)

    val_size = max(1, int(0.15 * len(dataset)))
    test_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size - test_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    batch_size = 128
    train_loader = DataLoader(
        train_set,
        batch_size,
        shuffle = True,
        num_workers = 16,
        pin_memory = (device.type == "cuda"),
        collate_fn = collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size,
        shuffle = False,
        num_workers = 16,
        pin_memory = (device.type == "cuda"),
        collate_fn = collate_fn
    )
    test_loader = DataLoader(
        test_set,
        batch_size,
        shuffle = False,
        num_workers = 16,
        pin_memory = (device.type == "cuda"),
        collate_fn = collate_fn
    )
    model = PopularityTransformer().to(device)
    criterion = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mae_metric = MeanAbsoluteError().to(device)
    r2_metric = R2Score().to(device)

    hyperparams = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "d_model": D_MODEL,
        "nhead": 8,
        "dim_feedforward": 3072,
        "num_layers": 6,
    }
    last_checkpoint_path = os.path.join(checkpoint_dir, "last_transformer_checkpoint.pt")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_transformer_model.pt")
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
    for epoch in range(start_epoch+1,epochs+1):
        train_loss, train_mae, train_r2 = run_epoch(
            model, train_loader, criterion, mae_metric, r2_metric, optimizer
        )
        val_loss, val_mae, val_r2 = run_epoch(
            model, val_loader, criterion, mae_metric, r2_metric
        )
    
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} mae {train_mae:.2f} r2 {train_r2:.3f} | "
            f"val loss {val_loss:.4f} mae {val_mae:.2f} r2 {val_r2:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            is_best = True
        else:
            is_best = False

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_mae, test_r2 = run_epoch(
        model, test_loader, criterion, mae_metric, r2_metric
    )
    print(f"test (best model) | loss {test_loss:.4f} mae {test_mae:.2f} r2 {test_r2:.3f}")



if __name__ == "__main__":
    main()
    
