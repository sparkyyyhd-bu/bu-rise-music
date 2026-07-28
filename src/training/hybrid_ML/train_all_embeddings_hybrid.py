"""Compare fixed and legacy hybrid regressors on their matching held-out splits."""

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
from training.data_utils import grouped_track_id_split


REPO_ROOT = Path(__file__).resolve().parents[3]
USER = os.environ["USER"]
CHECKPOINT_DIR = Path("/net/scc1/scratch") / USER / "checkpoints"
OUTPUT_PATH = CHECKPOINT_DIR / "all_embeddings_hybrid_comparison.joblib"
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
LYRICS_CSV = (
    REPO_ROOT
    / "data"
    / "SpotGenTrack"
    / "Features Extracted"
    / "lyrics_features.csv"
)
ARTISTS_CSV = (
    REPO_ROOT
    / "data"
    / "SpotGenTrack"
    / "Data Sources"
    / "spotify_artists.csv"
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
ARTIST_STAT_FEATURES = ["artist_popularity", "followers"]
EXPERIMENTS = {
    "fixed_without_artist_stats": {
        "embedding_suffix": "_fixed",
        "include_artist_stats": False,
        "description": "Fixed embeddings + audio + lyrics (no artist stats)",
    },
    "fixed_with_everything": {
        "embedding_suffix": "_fixed",
        "include_artist_stats": True,
        "description": "Fixed embeddings + audio + lyrics + artist stats",
    },
    "legacy_leaky_with_everything": {
        "embedding_suffix": "",
        "include_artist_stats": True,
        "description": (
            "Legacy artist/album-leaky embeddings + audio + lyrics + artist stats"
        ),
    },
}


def load_embeddings(model_name, expected_dimensions, embedding_suffix):
    path = CHECKPOINT_DIR / f"{model_name}{embedding_suffix}_embeddings.npz"
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


def load_all_embeddings(embedding_suffix):
    embedding_frame = None
    paths = {}
    groups = {}
    split_track_ids = None
    for model_name, dimensions in EMBEDDING_DIMENSIONS.items():
        path, track_ids, frame, feature_names = load_embeddings(
            model_name, dimensions, embedding_suffix
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


def numeric_feature_frame(path, id_column, prefix):
    frame = pd.read_csv(path)
    frame = frame.drop(
        columns=[
            column
            for column in frame.columns
            if column.startswith("Unnamed:")
        ],
        errors="ignore",
    ).rename(columns={id_column: "id"})
    frame["id"] = frame["id"].astype(str)
    feature_names = [
        column
        for column in frame.columns
        if column != "id" and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rename = {name: f"{prefix}{name}" for name in feature_names}
    frame = frame[["id", *feature_names]].rename(columns=rename)
    return frame, list(rename.values())


def load_tabular_features():
    tracks = pd.read_csv(
        TRACKS_CSV,
        usecols=["id", "popularity", *SPOTIFY_AUDIO_FEATURES],
    )
    tracks["id"] = tracks["id"].astype(str)
    tracks = tracks.dropna(subset=["id", "popularity"]).drop_duplicates("id")

    low_level, low_level_names = numeric_feature_frame(
        LOW_LEVEL_CSV, "track_id", "low_level_"
    )
    low_level = low_level.drop_duplicates("id")
    tracks = tracks.merge(
        low_level[["id", *low_level_names]],
        on="id",
        how="inner",
        validate="one_to_one",
    )

    lyrics, lyric_names = numeric_feature_frame(
        LYRICS_CSV, "track_id", "lyrics_"
    )
    lyrics = lyrics.drop_duplicates("id")
    tracks = tracks.merge(lyrics, on="id", how="inner", validate="one_to_one")

    artists = pd.read_csv(
        ARTISTS_CSV, usecols=["track_id", *ARTIST_STAT_FEATURES]
    ).rename(columns={"track_id": "id"})
    artists["id"] = artists["id"].astype(str)
    artist_names = [f"artist_{name}" for name in ARTIST_STAT_FEATURES]
    artists = (
        artists.groupby("id", as_index=False)[ARTIST_STAT_FEATURES]
        .mean()
        .rename(columns=dict(zip(ARTIST_STAT_FEATURES, artist_names)))
    )
    tracks = tracks.merge(artists, on="id", how="inner", validate="one_to_one")
    feature_groups = {
        "audio": [*SPOTIFY_AUDIO_FEATURES, *low_level_names],
        "lyrics": lyric_names,
        "artist_stats": artist_names,
    }
    return tracks, feature_groups


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


def legacy_track_id_split(track_ids):
    """Reproduce the original deterministic random 70/15/15 split."""
    validation_size = max(1, int(0.15 * len(track_ids)))
    test_size = max(1, int(0.15 * len(track_ids)))
    train_size = len(track_ids) - validation_size - test_size
    if train_size < 1:
        raise ValueError("at least three cached embeddings are required")
    subsets = random_split(
        range(len(track_ids)),
        [train_size, validation_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    return tuple(
        [track_ids[index] for index in subset.indices] for subset in subsets
    )


def run_experiment(
    key,
    config,
    split_name,
    combined,
    embedding_groups,
    embedding_paths,
    tabular_groups,
    split_ids,
    model_templates,
    selected_models,
):
    train_ids, validation_ids, test_ids = map(set, split_ids)
    partitions = {
        "train": combined.loc[combined.index.isin(train_ids)],
        "validation": combined.loc[combined.index.isin(validation_ids)],
        "test": combined.loc[combined.index.isin(test_ids)],
    }
    if any(frame.empty for frame in partitions.values()):
        raise ValueError(f"{key} has an empty data partition")

    tabular_feature_names = [
        *tabular_groups["audio"],
        *tabular_groups["lyrics"],
    ]
    if config["include_artist_stats"]:
        tabular_feature_names.extend(tabular_groups["artist_stats"])
    feature_names = [
        *(
            feature
            for model_name in EMBEDDING_DIMENSIONS
            for feature in embedding_groups[model_name]
        ),
        *tabular_feature_names,
    ]
    print(f"\n{'=' * 78}\n{config['description']}", flush=True)
    print(
        f"split={split_name}; features={len(feature_names)}; "
        + ", ".join(
            f"{name}={len(frame)}" for name, frame in partitions.items()
        ),
        flush=True,
    )

    x = {name: frame[feature_names] for name, frame in partitions.items()}
    y = {name: frame["popularity"] for name, frame in partitions.items()}
    rng = np.random.default_rng(RANDOM_SEED)
    results = {}
    fitted_models = {}
    for name in selected_models:
        estimator, sample_cap = model_templates[name]
        model = clone(estimator)
        if sample_cap is not None and len(x["train"]) > sample_cap:
            positions = np.sort(
                rng.choice(len(x["train"]), size=sample_cap, replace=False)
            )
            fit_x = x["train"].iloc[positions]
            fit_y = y["train"].iloc[positions]
        else:
            fit_x, fit_y = x["train"], y["train"]
        print(f"\ntraining {name} on {len(fit_x)} samples...", flush=True)
        start = time.perf_counter()
        model.fit(fit_x, fit_y)
        results[name] = {
            "fit_samples": len(fit_x),
            "fit_seconds": time.perf_counter() - start,
            "train": evaluate(model, fit_x, fit_y, f"{name} train"),
            "validation": evaluate(
                model, x["validation"], y["validation"], f"{name} validation"
            ),
            "test": evaluate(model, x["test"], y["test"], f"{name} test"),
        }
        fitted_models[name] = model

    leaderboard = sorted(
        results, key=lambda name: results[name]["validation"]["rmse"]
    )
    best_name = leaderboard[0]
    print(f"\n{config['description']} leaderboard:", flush=True)
    for rank, name in enumerate(leaderboard, start=1):
        validation = results[name]["validation"]
        test = results[name]["test"]
        print(
            f"{rank:2d}. {name:18s} "
            f"validation RMSE {validation['rmse']:.3f} | "
            f"test RMSE {test['rmse']:.3f}, MAE {test['mae']:.3f}, "
            f"R2 {test['r2']:.3f}",
            flush=True,
        )

    importances = {}
    if "xgboost" in fitted_models:
        importances = grouped_feature_importance(
            fitted_models["xgboost"], feature_names, embedding_groups
        )
    return {
        "description": config["description"],
        "split": split_name,
        "split_counts": {
            name: len(frame) for name, frame in partitions.items()
        },
        "model": fitted_models[best_name],
        "model_name": best_name,
        "feature_names": feature_names,
        "embedding_feature_names": embedding_groups,
        "tabular_feature_names": tabular_feature_names,
        "embedding_paths": embedding_paths,
        "grouped_feature_importance": importances,
        "results": results,
    }


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

    fixed_ids, fixed_embeddings, fixed_paths, fixed_groups = (
        load_all_embeddings("_fixed")
    )
    legacy_ids, legacy_embeddings, legacy_paths, legacy_groups = (
        load_all_embeddings("")
    )
    tabular, tabular_groups = load_tabular_features()
    fixed_split_ids = grouped_track_id_split(fixed_ids)
    # Preserve the legacy cache order: it matches the os.scandir order used by
    # the original neural training split.
    legacy_split_ids = legacy_track_id_split(legacy_ids)
    splits_by_suffix = {
        "_fixed": (
            fixed_split_ids,
            "artist_album_isolated_fixed_70_15_15",
        ),
        "": (
            legacy_split_ids,
            "legacy_random_70_15_15",
        ),
    }
    embeddings_by_suffix = {
        "_fixed": (fixed_embeddings, fixed_paths, fixed_groups),
        "": (legacy_embeddings, legacy_paths, legacy_groups),
    }
    experiments = {}
    for key, config in EXPERIMENTS.items():
        embeddings, paths, groups = embeddings_by_suffix[
            config["embedding_suffix"]
        ]
        split_ids, split_name = splits_by_suffix[config["embedding_suffix"]]
        combined = embeddings.merge(
            tabular, on="id", how="inner", validate="one_to_one"
        ).set_index("id")
        experiments[key] = run_experiment(
            key,
            config,
            split_name,
            combined,
            groups,
            paths,
            tabular_groups,
            split_ids,
            available_models,
            args.models,
        )
    artifact = {
        "experiments": experiments,
        "random_seed": RANDOM_SEED,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "experiments": {
                    key: {
                        field: value
                        for field, value in experiment.items()
                        if field != "model"
                    }
                    for key, experiment in experiments.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    importance_path = args.output.with_suffix(".feature_importance.csv")
    importance_rows = [
        (key, feature, importance)
        for key, experiment in experiments.items()
        for feature, importance in experiment["grouped_feature_importance"].items()
    ]
    pd.DataFrame(
        importance_rows, columns=["experiment", "feature", "importance"]
    ).to_csv(importance_path, index=False)
    print(f"saved model to {args.output}", flush=True)
    print(f"saved metrics to {metrics_path}", flush=True)
    print(f"saved grouped feature importance to {importance_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
