"""Randomized grouped-CV search for the fixed multimodal no-artist XGBoost.

The feature set exactly matches ``fixed_without_artist_stats`` in
intermediate_concatenation/train_intermediate_concatenation.py: all fixed neural
embeddings, Spotify and
low-level audio features, and lyrics features. Artist popularity and follower
counts are deliberately excluded.

Hyperparameters are selected with artist/album-component GroupKFold on the
fixed training partition. The fixed validation partition is reported once
after selection; the final model is then refit on train + validation and
evaluated once on the untouched fixed test partition.
"""

import argparse
import ast
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from training.data_utils import grouped_track_id_split
from training.hybrid_ML.intermediate_concatenation.train_intermediate_concatenation import (
    CHECKPOINT_DIR,
    EMBEDDING_DIMENSIONS,
    RANDOM_SEED,
    TRACKS_CSV,
    evaluate,
    load_all_embeddings,
    load_tabular_features,
    tree_pipeline,
)
from xgboost import XGBRegressor


DEFAULT_OUTPUT = (
    CHECKPOINT_DIR
    / "hybrid"
    / "intermediate_concatenation"
    / "no_artist"
    / "intermediate_no_artist_xgboost_random_search.joblib"
)


def json_safe(value):
    """Convert NumPy values sampled by SciPy into JSON-native values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"final artifact path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def _artist_ids(value):
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        parsed = [value]
    if isinstance(parsed, str):
        parsed = [parsed]
    return sorted({str(artist) for artist in parsed if str(artist)})


def artist_album_groups(track_ids):
    """Return one connected-component label per track ID."""
    track_ids = list(map(str, track_ids))
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("track IDs must be unique")
    wanted = set(track_ids)
    metadata = pd.read_csv(
        TRACKS_CSV,
        usecols=["id", "album_id", "artists_id"],
        dtype=str,
    ).dropna(subset=["id"])
    metadata = metadata.loc[metadata["id"].isin(wanted)]
    missing = wanted - set(metadata["id"])
    if missing:
        raise ValueError(
            f"{len(missing)} tracks have no artist/album metadata"
        )

    parent = {track_id: track_id for track_id in track_ids}

    def find(track_id):
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owners = {}
    for row in metadata.itertuples(index=False):
        entities = []
        if pd.notna(row.album_id) and row.album_id:
            entities.append(("album", row.album_id))
        entities.extend(
            ("artist", artist) for artist in _artist_ids(row.artists_id)
        )
        for entity in entities:
            union(row.id, owners.setdefault(entity, row.id))

    roots = {track_id: find(track_id) for track_id in track_ids}
    root_labels = {
        root: index for index, root in enumerate(sorted(set(roots.values())))
    }
    return np.asarray([root_labels[roots[track_id]] for track_id in track_ids])


def build_search(device, n_iter, cv_folds):
    estimator = tree_pipeline(
        XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            device=device,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    )
    distributions = {
        "regressor__n_estimators": randint(300, 1_601),
        "regressor__learning_rate": loguniform(0.01, 0.2),
        "regressor__max_depth": randint(3, 11),
        "regressor__min_child_weight": loguniform(0.5, 20.0),
        "regressor__subsample": uniform(0.6, 0.4),
        "regressor__colsample_bytree": uniform(0.5, 0.5),
        "regressor__gamma": loguniform(1e-3, 5.0),
        "regressor__reg_alpha": loguniform(1e-4, 10.0),
        "regressor__reg_lambda": loguniform(0.1, 30.0),
    }
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        cv=GroupKFold(n_splits=cv_folds),
        refit=True,
        random_state=RANDOM_SEED,
        verbose=2,
        return_train_score=True,
    )


def main():
    args = parse_args()
    if args.n_iter < 1:
        raise ValueError("--n-iter must be positive")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")

    split_track_ids, embeddings, embedding_paths, embedding_groups = (
        load_all_embeddings("_fixed")
    )
    tabular, tabular_groups = load_tabular_features()
    combined = embeddings.merge(
        tabular, on="id", how="inner", validate="one_to_one"
    ).set_index("id")
    train_ids, validation_ids, test_ids = grouped_track_id_split(
        split_track_ids
    )
    partitions = {
        name: combined.loc[combined.index.isin(ids)]
        for name, ids in (
            ("train", train_ids),
            ("validation", validation_ids),
            ("test", test_ids),
        )
    }
    if any(frame.empty for frame in partitions.values()):
        raise ValueError("at least one fixed data partition is empty")

    feature_names = [
        *(
            feature
            for model_name in EMBEDDING_DIMENSIONS
            for feature in embedding_groups[model_name]
        ),
        *tabular_groups["audio"],
        *tabular_groups["lyrics"],
    ]
    x = {name: frame[feature_names] for name, frame in partitions.items()}
    y = {name: frame["popularity"] for name, frame in partitions.items()}
    groups = artist_album_groups(x["train"].index)
    if len(np.unique(groups)) < args.cv_folds:
        raise ValueError(
            f"only {len(np.unique(groups))} artist/album components are "
            f"available for {args.cv_folds}-fold CV"
        )

    search = build_search(args.device, args.n_iter, args.cv_folds)
    print(
        f"searching {args.n_iter} candidates with {args.cv_folds}-fold "
        f"grouped CV on {len(x['train'])} tracks and "
        f"{len(feature_names)} no-artist features",
        flush=True,
    )
    started = time.perf_counter()
    search.fit(x["train"], y["train"], groups=groups)
    search_seconds = time.perf_counter() - started
    validation_metrics = evaluate(
        search.best_estimator_,
        x["validation"],
        y["validation"],
        "selected model validation",
    )

    final_model = clone(search.best_estimator_)
    final_x = pd.concat([x["train"], x["validation"]])
    final_y = pd.concat([y["train"], y["validation"]])
    final_model.fit(final_x, final_y)
    test_metrics = evaluate(
        final_model, x["test"], y["test"], "refit model test"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "best_params": search.best_params_,
        "best_cv_rmse": float(-search.best_score_),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "feature_names": feature_names,
        "embedding_feature_names": embedding_groups,
        "embedding_paths": embedding_paths,
        "split": "artist_album_isolated_fixed_70_15_15",
        "split_counts": {
            name: len(frame) for name, frame in partitions.items()
        },
        "cv": "artist_album_component_group_k_fold",
        "cv_folds": args.cv_folds,
        "n_iter": args.n_iter,
        "device": args.device,
        "random_seed": RANDOM_SEED,
        "search_seconds": search_seconds,
    }
    joblib.dump(artifact, args.output)

    cv_results_path = args.output.with_suffix(".cv_results.csv")
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        cv_results_path, index=False
    )
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            json_safe(
                {
                    key: value
                    for key, value in artifact.items()
                    if key != "model"
                }
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved final model to {args.output}", flush=True)
    print(f"saved CV results to {cv_results_path}", flush=True)
    print(f"saved metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
