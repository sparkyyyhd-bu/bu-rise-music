"""Train a fixed ViT-linear-layer-plus-lyrics Random Forest stack.

A Ridge linear layer first reduces the fixed ViT embedding to one supervised
feature. The Random Forest receives that feature plus the lyrics features.
Out-of-fold linear predictions are used to train the forest, preventing target
leakage between the two stages.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict

from training.data_utils import grouped_track_id_split
from training.hybrid_ML.train_all_embeddings_hybrid import (
    CHECKPOINT_DIR,
    EMBEDDING_DIMENSIONS,
    RANDOM_SEED,
    evaluate,
    load_embeddings,
    load_tabular_features,
    scaled_pipeline,
    tree_pipeline,
)
from training.hybrid_ML.search_fixed_no_artist_xgboost import (
    artist_album_groups,
)


DEFAULT_OUTPUT = CHECKPOINT_DIR / "fixed_vit_lyrics_random_forest.joblib"
MODEL_PARAMETERS = {
    "n_estimators": 500,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "max_depth": 35,
    "bootstrap": True,
}
LINEAR_ALPHA = 10.0
STACKING_FOLDS = 5
VIT_LINEAR_FEATURE = "vit_linear_prediction"


def stacked_features(frame, linear_predictions, lyric_features):
    features = frame[lyric_features].copy()
    features.insert(0, VIT_LINEAR_FEATURE, np.asarray(linear_predictions))
    return features


class ViTLinearLyricsRandomForest:
    """Serializable two-stage predictor operating on the original columns."""

    def __init__(
        self, linear_model, forest_model, embedding_features, lyric_features
    ):
        self.linear_model = linear_model
        self.forest_model = forest_model
        self.embedding_features = list(embedding_features)
        self.lyric_features = list(lyric_features)

    def predict(self, frame):
        linear_predictions = self.linear_model.predict(
            frame[self.embedding_features]
        )
        return self.forest_model.predict(
            stacked_features(
                frame, linear_predictions, self.lyric_features
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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

    linear_model = scaled_pipeline(Ridge(alpha=LINEAR_ALPHA))
    forest_model = tree_pipeline(
        RandomForestRegressor(
            **MODEL_PARAMETERS,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    )

    print(
        f"training fixed ViT linear layer + lyrics Random Forest on "
        f"{len(x['train'])} tracks "
        f"with {len(embedding_features)} ViT and "
        f"{len(tabular_groups['lyrics'])} lyrics features",
        flush=True,
    )
    print(
        f"creating {STACKING_FOLDS}-fold artist/album-grouped "
        "out-of-fold ViT linear predictions",
        flush=True,
    )
    print(f"parameters: {MODEL_PARAMETERS}", flush=True)
    started = time.perf_counter()
    train_groups = artist_album_groups(x["train"].index)
    oof_linear_predictions = cross_val_predict(
        linear_model,
        x["train"][embedding_features],
        y["train"],
        groups=train_groups,
        cv=GroupKFold(n_splits=STACKING_FOLDS),
        method="predict",
        n_jobs=1,
    )
    linear_model.fit(x["train"][embedding_features], y["train"])
    forest_train_x = stacked_features(
        x["train"], oof_linear_predictions, tabular_groups["lyrics"]
    )
    forest_model.fit(forest_train_x, y["train"])
    model = ViTLinearLyricsRandomForest(
        linear_model,
        forest_model,
        embedding_features,
        tabular_groups["lyrics"],
    )
    fit_seconds = time.perf_counter() - started
    metrics = {
        split: evaluate(model, x[split], y[split], f"random forest {split}")
        for split in ("train", "validation", "test")
    }

    artifact = {
        "model": {
            "linear_model": linear_model,
            "random_forest": forest_model,
        },
        "linear_model": "Ridge",
        "linear_alpha": LINEAR_ALPHA,
        "stacking_folds": STACKING_FOLDS,
        "stacking_cv": "artist_album_component_group_k_fold",
        "model_parameters": MODEL_PARAMETERS,
        "metrics": metrics,
        "fit_seconds": fit_seconds,
        "input_feature_names": feature_names,
        "random_forest_feature_names": [
            VIT_LINEAR_FEATURE,
            *tabular_groups["lyrics"],
        ],
        "embedding_feature_names": embedding_groups,
        "embedding_paths": embedding_paths,
        "modalities": ["ViT_fixed_embeddings", "lyrics"],
        "split": "artist_album_isolated_fixed_70_15_15",
        "split_counts": {
            name: len(frame) for name, frame in partitions.items()
        },
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
