"""Train a Random Forest on frozen CNN embeddings and all audio-only features.

The 64 values produced by the best CNN's final hidden layer (fc2) are joined
with both Spotify audio descriptors and the extracted low-level audio features.
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset, random_split


REPO_ROOT = Path(__file__).resolve().parents[3]
USER = os.environ["USER"]
MEL_DIR = Path("/net/scc1/scratch") / USER / "mel_spectrograms"
CNN_CHECKPOINT = REPO_ROOT / "models" / "best_CNN_model.pt"
TRACKS_CSV = (
    REPO_ROOT
    / "data"
    / "SpotGenTrack"
    / "Data Sources"
    / "spotify_tracks.csv"
)
LOW_LEVEL_CSV = (
    REPO_ROOT
    / "data"
    / "SpotGenTrack"
    / "Features Extracted"
    / "low_level_audio_features.csv"
)
OUTPUT_PATH = (
    Path("/net/scc1/scratch")
    / USER
    / "checkpoints"
    / "best_CNN_RF_model.joblib"
)
BATCH_SIZE = 32
NUM_WORKERS = 8
N_TREES = 500
RANDOM_SEED = 0
MEL_SUFFIX = "_mel.pt"
SPOTIFY_AUDIO_FEATURES = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "time_signature",
    "valence",
]


class PopularityCNN(nn.Module):
    """Architecture used by train_CNN.py, with access to its last hidden vector."""

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
        mel = (mel.float() - mel.float().mean()) / (mel.float().std() + 1e-5)
        return mel, track_id


def collate_mels(batch):
    mels, track_ids = zip(*batch)
    max_frames = max(mel.shape[-1] for mel in mels)
    padded = torch.stack(
        [F.pad(mel, (0, max_frames - mel.shape[-1])) for mel in mels]
    ).unsqueeze(1)
    return padded, list(track_ids)


def load_cnn(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model = PopularityCNN()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def extract_embeddings(model, subset, device, batch_size, num_workers):
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_mels,
    )
    rows = []
    with torch.inference_mode():
        for mels, track_ids in loader:
            values = model.embedding(
                mels.to(device, non_blocking=True)
            ).cpu().numpy()
            rows.extend(
                {"id": track_id, **{f"cnn_{i:02d}": value for i, value in enumerate(vector)}}
                for track_id, vector in zip(track_ids, values)
            )
    return pd.DataFrame(rows)


def load_tabular_features():
    tracks = pd.read_csv(
        TRACKS_CSV,
        usecols=["id", "popularity", *SPOTIFY_AUDIO_FEATURES],
    )
    tracks = tracks.dropna(subset=["id", "popularity"]).drop_duplicates("id")
    feature_names = list(SPOTIFY_AUDIO_FEATURES)

    low_level = pd.read_csv(LOW_LEVEL_CSV)
    low_level = low_level.drop(
        columns=[
            column
            for column in low_level.columns
            if column.startswith("Unnamed:")
        ],
        errors="ignore",
    ).rename(columns={"track_id": "id"})
    low_level = low_level.drop_duplicates("id")
    low_level_names = [
        column
        for column in low_level.columns
        if column != "id" and pd.api.types.is_numeric_dtype(low_level[column])
    ]
    tracks = tracks.merge(
        low_level[["id", *low_level_names]],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    feature_names.extend(low_level_names)

    return tracks, feature_names


def evaluate(model, x, y, label):
    predictions = np.clip(model.predict(x), 0.0, 100.0)
    metrics = {
        "mae": mean_absolute_error(y, predictions),
        "rmse": mean_squared_error(y, predictions) ** 0.5,
        "r2": r2_score(y, predictions),
    }
    print(
        f"{label}: MAE {metrics['mae']:.3f} | "
        f"RMSE {metrics['rmse']:.3f} | R2 {metrics['r2']:.3f}",
        flush=True,
    )
    return metrics


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"extracting CNN embeddings on {device}", flush=True)

    tabular, audio_feature_names = load_tabular_features()
    labeled_tracks = pd.read_csv(
        TRACKS_CSV, usecols=["id", "popularity"]
    ).dropna(subset=["id", "popularity"])
    # Partition the same full labeled population as train_CNN.py, then join
    # low-level features. Filtering before the split would shift assignments.
    dataset = MelTrackDataset(MEL_DIR, set(labeled_tracks["id"]))
    val_size = max(1, int(0.15 * len(dataset)))
    test_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size - test_size
    if train_size < 1:
        raise ValueError("at least three mel spectrograms are required")

    # Reconstruct train_CNN.py's seed-0 partition so RF validation examples
    # were not used to fit the CNN. The original CNN script always used seed 0.
    train_set, val_set, _ = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(0),
    )
    cnn = load_cnn(CNN_CHECKPOINT, device)
    train_embeddings = extract_embeddings(
        cnn, train_set, device, BATCH_SIZE, NUM_WORKERS
    )
    val_embeddings = extract_embeddings(
        cnn, val_set, device, BATCH_SIZE, NUM_WORKERS
    )

    cnn_feature_names = [f"cnn_{i:02d}" for i in range(64)]
    feature_names = [*cnn_feature_names, *audio_feature_names]

    train = train_embeddings.merge(tabular, on="id", how="inner", validate="one_to_one")
    validation = val_embeddings.merge(
        tabular, on="id", how="inner", validate="one_to_one"
    )
    if train.empty or validation.empty:
        raise ValueError("no labeled track IDs overlap the mel and tabular datasets")
    print(
        f"matched {len(train)} training and {len(validation)} validation tracks; "
        f"using {len(feature_names)} features",
        flush=True,
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=N_TREES,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    max_features="sqrt",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )
    pipeline.fit(train[feature_names], train["popularity"])
    train_metrics = evaluate(
        pipeline, train[feature_names], train["popularity"], "train"
    )
    val_metrics = evaluate(
        pipeline, validation[feature_names], validation["popularity"], "validation"
    )

    artifact = {
        "model": pipeline,
        "feature_names": feature_names,
        "cnn_feature_names": cnn_feature_names,
        "audio_feature_names": audio_feature_names,
        "cnn_checkpoint": str(CNN_CHECKPOINT),
        "split_seed": 0,
        "rf_seed": RANDOM_SEED,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUTPUT_PATH)
    metrics_path = OUTPUT_PATH.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {"train": train_metrics, "validation": val_metrics},
            indent=2,
        )
        + "\n"
    )
    print(f"saved model to {OUTPUT_PATH}", flush=True)
    print(f"saved metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
