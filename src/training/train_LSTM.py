from training.data_utils import MelPopularityDataset, load_popularity_by_id, sync_to_local_scratch
import torch
import os
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, R2Score


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

network_mel_dir = os.path.join("/net/scc1/scratch", os.environ["USER"], "mel_spectrograms")
local_scratch_dir = os.path.join("/scratch", os.environ["USER"])
checkpoint_dir = os.path.join("/net/scc1/scratch", os.environ["USER"], "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
mel_dir = sync_to_local_scratch(network_mel_dir, local_scratch_dir)

class PopularityLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = 128,
            hidden_size = 512,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Linear(512, 1)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
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
            mels = mels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            predictions = model(mels)
            loss = criterion(predictions, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * mels.size(0)
            mae_metric.update(predictions * 100, targets * 100)
            r2_metric.update(predictions * 100, targets * 100)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, mae_metric.compute().item(), r2_metric.compute().item()

def collate_fn(batch):
    mels, targets = zip(*batch)
    max_frames = max(mel.shape[-1] for mel in mels)
    padded = torch.stack(
        [F.pad(mel, (0, max_frames - mel.shape[-1])) for mel in mels]
    )
    padded = padded.transpose(1, 2)  # (B, time, n_mels)
    targets = torch.stack(targets)
    return padded, targets

def main():
    torch.manual_seed(0)
    popularity_by_id = load_popularity_by_id()
    dataset = MelPopularityDataset(mel_dir,popularity_by_id)
    print(f"loaded {len(dataset)} labeled mel spectrograms")

    val_size = max(1, int(0.15 * len(dataset)))
    test_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size - test_size
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(0),
    )

    batch_size = 16
    train_loader = DataLoader(
        train_set,
        batch_size,
        shuffle = True,
        num_workers = 8,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size,
        shuffle = False,
        num_workers = 8,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn
    )
    model = PopularityLSTM().to(device)
    checkpoint_path = os.path.join(checkpoint_dir, "best_LSTM_model.pt")
    resumed = os.path.exists(checkpoint_path)
    if resumed:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        print(f"continuing from {checkpoint_path}", flush=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mae_metric = MeanAbsoluteError().to(device)
    r2_metric = R2Score().to(device)

    epochs = 50
    if resumed:
        best_val_loss, _, _ = run_epoch(
            model, val_loader, criterion, mae_metric, r2_metric
        )
        print(f"resumed validation loss: {best_val_loss:.4f}", flush=True)
    else:
        best_val_loss = float("inf")


    for epoch in range(1, epochs + 1):
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
            torch.save(model.state_dict(), checkpoint_path)

    print(f"best val loss {best_val_loss:.4f}, checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
