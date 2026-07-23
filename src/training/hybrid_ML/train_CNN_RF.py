"""Train a Random Forest from cached CNN embeddings and all audio features."""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from torch.utils.data import random_split


REPO_ROOT = Path(__file__).resolve().parents[3]
USER = os.environ["USER"]
CHECKPOINT_DIR = Path("/net/scc1/scratch") / USER / "checkpoints"
EMBEDDINGS_PATH = CHECKPOINT_DIR / "CNN_embeddings.npz"
OUTPUT_PATH = CHECKPOINT_DIR / "best_CNN_RF_model.joblib"
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
N_TREES = 500
RANDOM_SEED = 0
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


def load_embeddings():
    with np.load(EMBEDDINGS_PATH, allow_pickle=False) as cache:
        track_ids = cache["track_ids"].astype(str)
        embeddings = cache["embeddings"].astype(np.float32, copy=False)
    if embeddings.ndim != 2 or embeddings.shape[1] != 64:
        raise ValueError(
            f"expected embeddings with shape (tracks, 64), got {embeddings.shape}"
        )
    if len(track_ids) != len(embeddings):
        raise ValueError("embedding and track ID counts do not match")
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("embedding cache contains duplicate track IDs")
    return track_ids, embeddings


def load_audio_features():
    tracks = pd.read_csv(
        TRACKS_CSV,
        usecols=["id", "popularity", *SPOTIFY_AUDIO_FEATURES],
    )
    tracks = tracks.dropna(subset=["id", "popularity"]).drop_duplicates("id")

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
    return tracks, [*SPOTIFY_AUDIO_FEATURES, *low_level_names]


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
    track_ids, embeddings = load_embeddings()

    val_size = max(1, int(0.15 * len(track_ids)))
    test_size = max(1, int(0.15 * len(track_ids)))
    train_size = len(track_ids) - val_size - test_size
    if train_size < 1:
        raise ValueError("at least three cached embeddings are required")

    # The cache preserves MelPopularityDataset's scan order. Reusing seed 0
    # therefore reconstructs the split from train_CNN.py.
    train_subset, val_subset, _ = random_split(
        range(len(track_ids)),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(0),
    )
    train_indices = np.asarray(train_subset.indices)
    val_indices = np.asarray(val_subset.indices)

    cnn_feature_names = [f"cnn_{index:02d}" for index in range(64)]
    embedding_frame = pd.DataFrame(embeddings, columns=cnn_feature_names)
    embedding_frame.insert(0, "id", track_ids)
    tabular, audio_feature_names = load_audio_features()
    combined = embedding_frame.merge(
        tabular, on="id", how="inner", validate="one_to_one"
    ).set_index("id")

    train_ids = set(track_ids[train_indices])
    val_ids = set(track_ids[val_indices])
    train = combined.loc[combined.index.isin(train_ids)]
    validation = combined.loc[combined.index.isin(val_ids)]
    if train.empty or validation.empty:
        raise ValueError("no cached IDs overlap the tabular audio features")

    feature_names = [*cnn_feature_names, *audio_feature_names]
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
    validation_metrics = evaluate(
        pipeline,
        validation[feature_names],
        validation["popularity"],
        "validation",
    )

    artifact = {
        "model": pipeline,
        "feature_names": feature_names,
        "cnn_feature_names": cnn_feature_names,
        "audio_feature_names": audio_feature_names,
        "embeddings_path": str(EMBEDDINGS_PATH),
        "split_seed": 0,
        "rf_seed": RANDOM_SEED,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUTPUT_PATH)
    metrics_path = OUTPUT_PATH.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {"train": train_metrics, "validation": validation_metrics},
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
