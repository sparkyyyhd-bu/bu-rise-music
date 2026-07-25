"""Train hybrid regressors using CNN, ResNet, and audio features."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR
from torch.utils.data import random_split
from xgboost import XGBRegressor


REPO_ROOT = Path(__file__).resolve().parents[3]
USER = os.environ["USER"]
CHECKPOINT_DIR = Path("/net/scc1/scratch") / USER / "checkpoints"
CNN_EMBEDDINGS_PATH = CHECKPOINT_DIR / "CNN_embeddings.npz"
RESNET_EMBEDDINGS_PATH = CHECKPOINT_DIR / "ResNet_embeddings.npz"
OUTPUT_PATH = CHECKPOINT_DIR / "best_CNN_ResNet_hybrid_model.joblib"
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
DEFAULT_MODELS = [
    "ridge",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "knn",
    "linear_svm",
    "rbf_svm",
    "mlp",
]
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


def load_embeddings(path, expected_dimensions, label):
    with np.load(path, allow_pickle=False) as cache:
        track_ids = cache["track_ids"].astype(str)
        embeddings = cache["embeddings"].astype(np.float32, copy=False)
    if embeddings.ndim != 2 or embeddings.shape[1] != expected_dimensions:
        raise ValueError(
            f"expected {label} embeddings with shape "
            f"(tracks, {expected_dimensions}), got {embeddings.shape}"
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
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(mean_squared_error(y, predictions) ** 0.5),
        "r2": float(r2_score(y, predictions)),
    }
    print(
        f"{label}: MAE {metrics['mae']:.3f} | "
        f"RMSE {metrics['rmse']:.3f} | R2 {metrics['r2']:.3f}",
        flush=True,
    )
    return metrics


def scaled_pipeline(regressor):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )


def tree_pipeline(regressor):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", regressor),
        ]
    )


def build_models():
    """Return common regressors and optional training-set caps.

    Exact RBF SVM and KNN do not scale gracefully to 100k high-dimensional
    samples, so their fits are deterministically capped while evaluation still
    uses the complete validation set.
    """
    return {
        "ridge": (scaled_pipeline(Ridge(alpha=10.0)), None),
        "random_forest": (
            tree_pipeline(
                RandomForestRegressor(
                    n_estimators=N_TREES,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    max_features="sqrt",
                    min_samples_leaf=2,
                )
            ),
            None,
        ),
        "extra_trees": (
            tree_pipeline(
                ExtraTreesRegressor(
                    n_estimators=N_TREES,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    max_features="sqrt",
                    min_samples_leaf=2,
                )
            ),
            None,
        ),
        "hist_gradient_boosting": (
            tree_pipeline(
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=RANDOM_SEED,
                )
            ),
            None,
        ),
        "xgboost": (
            tree_pipeline(
                XGBRegressor(
                    n_estimators=1_000,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                )
            ),
            None,
        ),
        "knn": (
            scaled_pipeline(
                KNeighborsRegressor(
                    n_neighbors=20, weights="distance", n_jobs=-1
                )
            ),
            30_000,
        ),
        "linear_svm": (
            scaled_pipeline(
                LinearSVR(
                    C=1.0,
                    epsilon=0.1,
                    dual="auto",
                    max_iter=10_000,
                    random_state=RANDOM_SEED,
                )
            ),
            None,
        ),
        "rbf_svm": (
            scaled_pipeline(SVR(C=10.0, epsilon=0.1, kernel="rbf")),
            20_000,
        ),
        "mlp": (
            scaled_pipeline(
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    early_stopping=True,
                    max_iter=300,
                    batch_size=256,
                    random_state=RANDOM_SEED,
                )
            ),
            None,
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="model names to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"best-model artifact path (default: {OUTPUT_PATH})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    available_models = build_models()
    unknown_models = sorted(set(args.models) - set(available_models))
    if unknown_models:
        raise ValueError(
            f"unknown models: {', '.join(unknown_models)}; choices are "
            f"{', '.join(available_models)}"
        )
    track_ids, cnn_embeddings = load_embeddings(
        CNN_EMBEDDINGS_PATH, 64, "CNN"
    )
    resnet_track_ids, resnet_embeddings = load_embeddings(
        RESNET_EMBEDDINGS_PATH, 512, "ResNet"
    )

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
    resnet_feature_names = [
        f"resnet_{index:03d}" for index in range(512)
    ]
    cnn_frame = pd.DataFrame(cnn_embeddings, columns=cnn_feature_names)
    cnn_frame.insert(0, "id", track_ids)
    resnet_frame = pd.DataFrame(
        resnet_embeddings, columns=resnet_feature_names
    )
    resnet_frame.insert(0, "id", resnet_track_ids)
    embedding_frame = cnn_frame.merge(
        resnet_frame, on="id", how="inner", validate="one_to_one"
    )
    if embedding_frame.empty:
        raise ValueError("CNN and ResNet embedding caches have no IDs in common")
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

    feature_names = [
        *cnn_feature_names,
        *resnet_feature_names,
        *audio_feature_names,
    ]
    print(
        f"matched {len(train)} training and {len(validation)} validation tracks; "
        f"using {len(feature_names)} features",
        flush=True,
    )
    x_train = train[feature_names]
    y_train = train["popularity"]
    x_validation = validation[feature_names]
    y_validation = validation["popularity"]
    rng = np.random.default_rng(RANDOM_SEED)
    results = {}
    best_name = None
    best_model = None

    for name in args.models:
        estimator, sample_cap = available_models[name]
        model = clone(estimator)
        if sample_cap is not None and len(x_train) > sample_cap:
            fit_positions = np.sort(
                rng.choice(len(x_train), size=sample_cap, replace=False)
            )
            fit_x = x_train.iloc[fit_positions]
            fit_y = y_train.iloc[fit_positions]
        else:
            fit_x, fit_y = x_train, y_train

        print(
            f"\ntraining {name} on {len(fit_x)} samples...",
            flush=True,
        )
        start = time.perf_counter()
        model.fit(fit_x, fit_y)
        fit_seconds = time.perf_counter() - start
        train_metrics = evaluate(model, fit_x, fit_y, f"{name} train")
        validation_metrics = evaluate(
            model, x_validation, y_validation, f"{name} validation"
        )
        results[name] = {
            "fit_samples": len(fit_x),
            "fit_seconds": fit_seconds,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        if (
            best_name is None
            or validation_metrics["rmse"]
            < results[best_name]["validation"]["rmse"]
        ):
            best_name, best_model = name, model

    print("\nValidation leaderboard (lower RMSE is better):", flush=True)
    for rank, (name, result) in enumerate(
        sorted(results.items(), key=lambda item: item[1]["validation"]["rmse"]),
        start=1,
    ):
        metrics = result["validation"]
        print(
            f"{rank:2d}. {name:24s} RMSE {metrics['rmse']:.3f} | "
            f"MAE {metrics['mae']:.3f} | R2 {metrics['r2']:.3f}",
            flush=True,
        )

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": feature_names,
        "cnn_feature_names": cnn_feature_names,
        "resnet_feature_names": resnet_feature_names,
        "audio_feature_names": audio_feature_names,
        "cnn_embeddings_path": str(CNN_EMBEDDINGS_PATH),
        "resnet_embeddings_path": str(RESNET_EMBEDDINGS_PATH),
        "split_seed": 0,
        "random_seed": RANDOM_SEED,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps({"best_model": best_name, "models": results}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nbest model: {best_name}", flush=True)
    print(f"saved model to {args.output}", flush=True)
    print(f"saved metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
