"""Compare hybrid regressors using every best-model embedding plus audio features."""

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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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
OUTPUT_PATH = CHECKPOINT_DIR / "best_all_embeddings_hybrid_model.joblib"
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
RANDOM_SEED = 0
N_TREES = 500
DEFAULT_MODELS = [
    "ridge",
    "random_forest",
    "extra_trees",
    "xgboost",
    "knn",
    "linear_svm",
    "rbf_svm",
    "mlp",
]
EMBEDDING_DIMENSIONS = {
    "CNN": 64,
    "CNN_LSTM": 128,
    "LSTM": 512,
    "ResNet": 512,
    "ViT": 768,
}
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


def load_embeddings(model_name, expected_dimensions):
    path = CHECKPOINT_DIR / f"{model_name}_embeddings.npz"
    with np.load(path, allow_pickle=False) as cache:
        track_ids = cache["track_ids"].astype(str)
        embeddings = cache["embeddings"].astype(np.float32, copy=False)
    expected = (len(track_ids), expected_dimensions)
    if embeddings.shape != expected:
        raise ValueError(
            f"expected {model_name} embeddings with shape {expected}, "
            f"got {embeddings.shape}"
        )
    if len(set(track_ids)) != len(track_ids):
        raise ValueError(f"{model_name} embedding cache contains duplicate IDs")
    feature_names = [
        f"{model_name.lower()}_{index:04d}"
        for index in range(expected_dimensions)
    ]
    frame = pd.DataFrame(embeddings, columns=feature_names)
    frame.insert(0, "id", track_ids)
    return path, track_ids, frame, feature_names


def load_all_embeddings():
    embedding_frame = None
    paths = {}
    groups = {}
    split_track_ids = None
    for model_name, dimensions in EMBEDDING_DIMENSIONS.items():
        path, track_ids, frame, feature_names = load_embeddings(
            model_name, dimensions
        )
        paths[model_name] = str(path)
        groups[model_name] = feature_names
        if split_track_ids is None:
            split_track_ids = track_ids
        embedding_frame = (
            frame
            if embedding_frame is None
            else embedding_frame.merge(
                frame, on="id", how="inner", validate="one_to_one"
            )
        )
    if embedding_frame is None or embedding_frame.empty:
        raise ValueError("embedding caches have no track IDs in common")
    return split_track_ids, embedding_frame, paths, groups


def load_audio_features():
    tracks = pd.read_csv(
        TRACKS_CSV,
        usecols=["id", "popularity", *SPOTIFY_AUDIO_FEATURES],
    )
    tracks["id"] = tracks["id"].astype(str)
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
    low_level["id"] = low_level["id"].astype(str)
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


def build_models(xgboost_device):
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
                    device=xgboost_device,
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


def grouped_feature_importance(model, feature_names, embedding_groups):
    raw = model.named_steps["regressor"].feature_importances_.astype(float)
    if len(raw) != len(feature_names):
        raise ValueError("XGBoost feature importance count does not match inputs")
    importance_by_feature = dict(zip(feature_names, raw))
    embedding_columns = {
        column for columns in embedding_groups.values() for column in columns
    }
    grouped = {
        model_name: float(sum(importance_by_feature[name] for name in names))
        for model_name, names in embedding_groups.items()
    }
    grouped.update(
        {
            name: float(importance_by_feature[name])
            for name in feature_names
            if name not in embedding_columns
        }
    )
    total = sum(grouped.values())
    if total:
        grouped = {name: value / total for name, value in grouped.items()}
    return dict(sorted(grouped.items(), key=lambda item: item[1], reverse=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"models to compare (default: {', '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"model artifact path (default: {OUTPUT_PATH})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA GPU is available; submit this script with its GPU qsub file"
        )
    gpu_name = torch.cuda.get_device_name(0)
    print(f"using GPU for XGBoost: {gpu_name}", flush=True)
    available_models = build_models("cuda")
    unknown_models = sorted(set(args.models) - set(available_models))
    if unknown_models:
        raise ValueError(
            f"unknown models: {', '.join(unknown_models)}; choices are "
            f"{', '.join(available_models)}"
        )

    split_track_ids, embeddings, embedding_paths, embedding_groups = (
        load_all_embeddings()
    )
    tabular, audio_feature_names = load_audio_features()
    combined = embeddings.merge(
        tabular, on="id", how="inner", validate="one_to_one"
    ).set_index("id")

    val_size = max(1, int(0.15 * len(split_track_ids)))
    test_size = max(1, int(0.15 * len(split_track_ids)))
    train_size = len(split_track_ids) - val_size - test_size
    if train_size < 1:
        raise ValueError("at least three cached embeddings are required")
    train_subset, val_subset, _ = random_split(
        range(len(split_track_ids)),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(0),
    )
    train_ids = set(split_track_ids[np.asarray(train_subset.indices)])
    validation_ids = set(split_track_ids[np.asarray(val_subset.indices)])
    train = combined.loc[combined.index.isin(train_ids)]
    validation = combined.loc[combined.index.isin(validation_ids)]
    if train.empty or validation.empty:
        raise ValueError("no cached IDs overlap the tabular audio features")

    feature_names = [
        *(
            name
            for model_name in EMBEDDING_DIMENSIONS
            for name in embedding_groups[model_name]
        ),
        *audio_feature_names,
    ]
    x_train = train[feature_names]
    y_train = train["popularity"]
    x_validation = validation[feature_names]
    y_validation = validation["popularity"]
    print(
        f"matched {len(train)} training and {len(validation)} validation tracks; "
        f"using {len(feature_names)} features from {len(embedding_groups)} "
        "embeddings and audio metadata",
        flush=True,
    )
    rng = np.random.default_rng(RANDOM_SEED)
    results = {}
    fitted_models = {}
    for name in args.models:
        estimator, sample_cap = available_models[name]
        model = clone(estimator)
        if sample_cap is not None and len(x_train) > sample_cap:
            positions = np.sort(
                rng.choice(len(x_train), size=sample_cap, replace=False)
            )
            fit_x = x_train.iloc[positions]
            fit_y = y_train.iloc[positions]
        else:
            fit_x, fit_y = x_train, y_train
        print(f"\ntraining {name} on {len(fit_x)} samples...", flush=True)
        start = time.perf_counter()
        model.fit(fit_x, fit_y)
        results[name] = {
            "fit_samples": len(fit_x),
            "fit_seconds": time.perf_counter() - start,
            "train": evaluate(model, fit_x, fit_y, f"{name} train"),
            "validation": evaluate(
                model, x_validation, y_validation, f"{name} validation"
            ),
        }
        fitted_models[name] = model

    leaderboard = sorted(
        results, key=lambda name: results[name]["validation"]["rmse"]
    )
    best_name = leaderboard[0]
    best_model = fitted_models[best_name]
    print("\nValidation leaderboard (lower RMSE is better):", flush=True)
    for rank, name in enumerate(leaderboard, start=1):
        metrics = results[name]["validation"]
        print(
            f"{rank:2d}. {name:24s} RMSE {metrics['rmse']:.3f} | "
            f"MAE {metrics['mae']:.3f} | R2 {metrics['r2']:.3f}",
            flush=True,
        )

    if "xgboost" not in fitted_models:
        raise ValueError(
            "xgboost must be included in --models to calculate feature importance"
        )
    importances = grouped_feature_importance(
        fitted_models["xgboost"], feature_names, embedding_groups
    )

    print("\nGrouped XGBoost feature importance:", flush=True)
    for rank, (name, importance) in enumerate(importances.items(), start=1):
        print(f"{rank:3d}. {name:32s} {importance:.6f}", flush=True)

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": feature_names,
        "embedding_feature_names": embedding_groups,
        "audio_feature_names": audio_feature_names,
        "embedding_paths": embedding_paths,
        "grouped_feature_importance": importances,
        "split_seed": 0,
        "random_seed": RANDOM_SEED,
        "results": results,
        "feature_importance_model": "xgboost",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "best_model": best_name,
                "models": results,
                "feature_importance_model": "xgboost",
                "grouped_feature_importance": importances,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    importance_path = args.output.with_suffix(".feature_importance.csv")
    pd.DataFrame(
        importances.items(), columns=["feature", "importance"]
    ).to_csv(importance_path, index=False)
    print(f"\nbest model: {best_name}", flush=True)
    print(f"saved model to {args.output}", flush=True)
    print(f"saved metrics to {metrics_path}", flush=True)
    print(f"saved grouped feature importance to {importance_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
