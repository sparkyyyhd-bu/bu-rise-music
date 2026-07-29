"""Train one fixed ViT-embeddings-plus-lyrics XGBoost configuration.

This run adapts a supplied scikit-learn RandomForest configuration to XGBoost:

* n_estimators=300 and max_depth=30 are used directly.
* min_samples_leaf=1 maps approximately to min_child_weight=1.
* max_features="sqrt" maps to colsample_bytree=sqrt(p) / p.
* min_samples_split and bootstrap have no XGBoost equivalents.

Only the fixed ViT embeddings and lyrics features are included. Spotify audio,
low-level audio, artist statistics, and embeddings from the other neural
architectures are excluded. The held-out partitions use the shared fixed
artist/album-isolated split.
"""

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np

from training.data_utils import grouped_track_id_split
from training.hybrid_ML.train_all_embeddings_hybrid import (
    CHECKPOINT_DIR,
    EMBEDDING_DIMENSIONS,
    RANDOM_SEED,
    evaluate,
    load_embeddings,
    load_tabular_features,
    tree_pipeline,
)
from xgboost import XGBRegressor


DEFAULT_OUTPUT = CHECKPOINT_DIR / "fixed_no_artist_xgboost_specific.joblib"
REQUESTED_PARAMETERS = {
    "n_estimators": 500,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "max_depth": 35,
    "bootstrap": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"model artifact path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(RANDOM_SEED)

    embedding_path, split_track_ids, embeddings, embedding_features = (
        load_embeddings(
            "ViT", EMBEDDING_DIMENSIONS["ViT"], "_fixed"
        )
    )
    embedding_paths = {"ViT": str(embedding_path)}
    embedding_groups = {"ViT": embedding_features}
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
        *embedding_features,
        *tabular_groups["lyrics"],
    ]
    x = {name: frame[feature_names] for name, frame in partitions.items()}
    y = {name: frame["popularity"] for name, frame in partitions.items()}

    # XGBoost expresses feature sampling as a fraction rather than sklearn's
    # "sqrt" label. With p features, sqrt(p) / p selects the same expected
    # feature count per tree.
    colsample_bytree = math.sqrt(len(feature_names)) / len(feature_names)
    applied_parameters = {
        "n_estimators": REQUESTED_PARAMETERS["n_estimators"],
        "max_depth": REQUESTED_PARAMETERS["max_depth"],
        "min_child_weight": REQUESTED_PARAMETERS["min_samples_leaf"],
        "colsample_bytree": colsample_bytree,
    }
    unsupported_parameters = {
        "min_samples_split": REQUESTED_PARAMETERS["min_samples_split"],
        "bootstrap": REQUESTED_PARAMETERS["bootstrap"],
    }
    model = tree_pipeline(
        XGBRegressor(
            **applied_parameters,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            device=args.device,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    )

    print(
        f"training fixed ViT + lyrics XGBoost on {len(x['train'])} tracks "
        f"with {len(embedding_features)} ViT and "
        f"{len(tabular_groups['lyrics'])} lyrics features",
        flush=True,
    )
    print(f"applied parameters: {applied_parameters}", flush=True)
    print(
        "unsupported RandomForest parameters (not applied): "
        f"{unsupported_parameters}",
        flush=True,
    )
    started = time.perf_counter()
    model.fit(x["train"], y["train"])
    fit_seconds = time.perf_counter() - started
    metrics = {
        split: evaluate(model, x[split], y[split], f"xgboost {split}")
        for split in ("train", "validation", "test")
    }

    artifact = {
        "model": model,
        "requested_random_forest_parameters": REQUESTED_PARAMETERS,
        "applied_xgboost_parameters": applied_parameters,
        "unsupported_parameters": unsupported_parameters,
        "metrics": metrics,
        "fit_seconds": fit_seconds,
        "feature_names": feature_names,
        "embedding_feature_names": embedding_groups,
        "embedding_paths": embedding_paths,
        "modalities": ["ViT_fixed_embeddings", "lyrics"],
        "split": "artist_album_isolated_fixed_70_15_15",
        "split_counts": {
            name: len(frame) for name, frame in partitions.items()
        },
        "device": args.device,
        "random_seed": RANDOM_SEED,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {key: value for key, value in artifact.items() if key != "model"},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved model to {args.output}", flush=True)
    print(f"saved metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
