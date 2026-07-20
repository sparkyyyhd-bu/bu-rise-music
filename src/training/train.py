import os
import shutil

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import MeanAbsoluteError, R2Score

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
network_mel_dir = os.path.join("/net/scc1/scratch", os.environ["USER"], "mel_spectrograms")
local_scratch_dir = os.path.join("/scratch", os.environ["USER"])
tracks_csv = os.path.join(repo_root, "data", "SpotGenTrack", "Data Sources", "spotify_tracks.csv")
checkpoint_dir = os.path.join("/net/scc1/scratch", os.environ["USER"], "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

MEL_SUFFIX = "_mel.pt"


def sync_to_local_scratch(source_dir, local_root):
    """Copy source_dir onto the compute node's local scratch disk once, then
    reuse that copy on subsequent runs so training reads local disk instead
    of the network-mounted scratch space."""
    local_dir = os.path.join(local_root, os.path.basename(source_dir.rstrip("/")))
    done_marker = local_dir + ".sync_complete"
    if not os.path.exists(done_marker):
        os.makedirs(local_dir, exist_ok=True)
        filenames = [entry.name for entry in os.scandir(source_dir) if entry.is_file()]
        for name in tqdm(filenames, desc="syncing mel spectrograms to local scratch"):
            shutil.copy2(os.path.join(source_dir, name), os.path.join(local_dir, name))
        open(done_marker, "w").close()
    return local_dir


mel_dir = sync_to_local_scratch(network_mel_dir, local_scratch_dir)


class MelPopularityDataset(Dataset):
    def __init__(self, mel_dir, popularity_by_id):
        self.entries = []
        for entry in os.scandir(mel_dir):
            if not entry.is_file() or not entry.name.endswith(MEL_SUFFIX):
                continue
            track_id = entry.name[: -len(MEL_SUFFIX)]
            popularity = popularity_by_id.get(track_id)
            if popularity is None:
                continue
            self.entries.append((entry.path, popularity))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, popularity = self.entries[index]
        mel = torch.load(path)
        if mel.dim() == 3:
            mel = mel.mean(dim=0)  # collapse stereo channels to mono
        target = torch.tensor(popularity / 100.0, dtype=torch.float32)
        return mel, target


def collate_fn(batch):
    mels, targets = zip(*batch)
    max_frames = max(mel.shape[-1] for mel in mels)
    padded = torch.stack(
        [F.pad(mel, (0, max_frames - mel.shape[-1])) for mel in mels]
    )
    padded = padded.unsqueeze(1)  # (B, 1, n_mels, time)
    targets = torch.stack(targets)
    return padded, targets


class PopularityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Summarize to a fixed size regardless of spectrogram length.
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))  # popularity normalized to [0, 1]
        return x.squeeze(1)


def load_popularity_by_id():
    df = pd.read_csv(tracks_csv, usecols=["id", "popularity"])
    df = df.dropna(subset=["id", "popularity"])
    return dict(zip(df["id"], df["popularity"]))


def run_epoch(model, loader, criterion, mae_metric, r2_metric, optimizer=None):
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

    batch_size = 16
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = PopularityCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mae_metric = MeanAbsoluteError().to(device)
    r2_metric = R2Score().to(device)

    epochs = 50
    best_val_loss = float("inf")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

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
