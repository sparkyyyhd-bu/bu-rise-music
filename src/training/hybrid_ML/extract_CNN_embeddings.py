"""Precalculate the best CNN's 64-dimensional fc2 embedding for every track."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
USER = os.environ["USER"]
MEL_DIR = Path("/net/scc1/scratch") / USER / "mel_spectrograms"
CHECKPOINT_DIR = Path("/net/scc1/scratch") / USER / "checkpoints"
CNN_CHECKPOINT = CHECKPOINT_DIR / "best_CNN_model.pt"
OUTPUT_PATH = CHECKPOINT_DIR / "CNN_embeddings.npz"
TRACKS_CSV = (
    REPO_ROOT
    / "data"
    / "SpotGenTrack"
    / "Data Sources"
    / "spotify_tracks.csv"
)
BATCH_SIZE = 32
NUM_WORKERS = 8
MEL_SUFFIX = "_mel.pt"


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
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.4)

    def embedding(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x).flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.dropout(F.relu(self.fc2(x)))

    def forward(self, x):
        return torch.sigmoid(self.fc3(self.embedding(x))).squeeze(1)


class MelTrackDataset(Dataset):
    def __init__(self, mel_dir, labeled_ids):
        self.entries = [
            (entry.path, entry.name[: -len(MEL_SUFFIX)])
            for entry in os.scandir(mel_dir)
            if entry.is_file()
            and entry.name.endswith(MEL_SUFFIX)
            and entry.name[: -len(MEL_SUFFIX)] in labeled_ids
        ]
        if not self.entries:
            raise ValueError(f"no labeled {MEL_SUFFIX} files found in {mel_dir}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, track_id = self.entries[index]
        mel = torch.load(path, map_location="cpu", weights_only=True)
        if mel.dim() == 3:
            mel = mel.mean(dim=0)
        mel = mel.float()
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)
        return mel, track_id


def collate_mels(batch):
    mels, track_ids = zip(*batch)
    max_frames = max(mel.shape[-1] for mel in mels)
    padded = torch.stack(
        [F.pad(mel, (0, max_frames - mel.shape[-1])) for mel in mels]
    ).unsqueeze(1)
    return padded, list(track_ids)


def load_cnn(device):
    checkpoint = torch.load(CNN_CHECKPOINT, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model = PopularityCNN()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labeled_tracks = pd.read_csv(
        TRACKS_CSV, usecols=["id", "popularity"]
    ).dropna(subset=["id", "popularity"])
    dataset = MelTrackDataset(MEL_DIR, set(labeled_tracks["id"]))
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_mels,
    )
    model = load_cnn(device)
    all_track_ids = []
    all_embeddings = []
    print(
        f"extracting 64 CNN features for {len(dataset)} tracks on {device}",
        flush=True,
    )
    with torch.inference_mode():
        for batch_number, (mels, track_ids) in enumerate(loader, start=1):
            embeddings = model.embedding(
                mels.to(device, non_blocking=True)
            ).cpu().numpy()
            all_track_ids.extend(track_ids)
            all_embeddings.append(embeddings.astype(np.float32, copy=False))
            if batch_number % 100 == 0:
                print(
                    f"processed {min(batch_number * BATCH_SIZE, len(dataset))}"
                    f"/{len(dataset)} tracks",
                    flush=True,
                )

    embedding_array = np.concatenate(all_embeddings)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        track_ids=np.asarray(all_track_ids),
        embeddings=embedding_array,
    )
    size_mib = OUTPUT_PATH.stat().st_size / (1024**2)
    print(
        f"saved {embedding_array.shape} embeddings to {OUTPUT_PATH} "
        f"({size_mib:.1f} MiB)",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
