"""Compare lyrics RF, the trained ViT head, and their combination.

The original fixed ViT checkpoint's trained Linear + sigmoid head reduces each
cached ViT embedding to its popularity prediction. The combined Random Forest
receives that prediction plus the lyrics features. All three models use the
same fixed population and artist/album-isolated partitions.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor

from training.data_utils import grouped_track_id_split
from training.hybrid_ML.train_all_embeddings_hybrid import (
    CHECKPOINT_DIR,
    EMBEDDING_DIMENSIONS,
    LYRICS_CSV,
    RANDOM_SEED,
    TRACKS_CSV,
    evaluate,
    load_embeddings,
    numeric_feature_frame,
    tree_pipeline,
)


DEFAULT_OUTPUT = CHECKPOINT_DIR / "fixed_vit_lyrics_model_comparison.joblib"
MODEL_PARAMETERS = {
    "n_estimators": 500,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "max_depth": 35,
    "bootstrap": True,
}
VIT_CHECKPOINT = CHECKPOINT_DIR / "best_ViT_fixed_model.pt"
VIT_HEAD_FEATURE = "vit_checkpoint_prediction"


def stacked_features(frame, vit_predictions, lyric_features):
    features = frame[lyric_features].copy()
    features.insert(0, VIT_HEAD_FEATURE, np.asarray(vit_predictions))
    return features


class ViTCheckpointHead:
    """Apply the checkpoint's trained Linear + sigmoid head to embeddings."""

    def __init__(self, weight, bias, embedding_features):
        self.weight = np.asarray(weight).reshape(-1)
        self.bias = float(bias)
        self.embedding_features = list(embedding_features)

    def predict(self, frame):
        logits = (
            frame[self.embedding_features].to_numpy() @ self.weight + self.bias
        )
        return 100.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


class ViTHeadLyricsRandomForest:
    """Apply the checkpoint head, then combine its output with lyrics."""

    def __init__(self, vit_head, forest_model, lyric_features):
        self.vit_head = vit_head
        self.forest_model = forest_model
        self.lyric_features = list(lyric_features)

    def predict(self, frame):
        vit_predictions = self.vit_head.predict(frame)
        return self.forest_model.predict(
            stacked_features(frame, vit_predictions, self.lyric_features)
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

    tracks = pd.read_csv(
        TRACKS_CSV, usecols=["id", "popularity"], dtype={"id": str}
    ).dropna(subset=["id", "popularity"])
    tracks = tracks.drop_duplicates("id")
    lyrics, lyric_features = numeric_feature_frame(
        LYRICS_CSV, "track_id", "lyrics_"
    )
    lyrics = lyrics.drop_duplicates("id")
    # Lyrics are optional and median-imputed by each Random Forest pipeline.
    # Loading them directly avoids filtering on unrelated audio availability.
    tabular = tracks.merge(lyrics, on="id", how="left", validate="one_to_one")
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
        *lyric_features,
    ]
    x = {name: frame[feature_names] for name, frame in partitions.items()}
    y = {name: frame["popularity"] for name, frame in partitions.items()}

    checkpoint = torch.load(
        VIT_CHECKPOINT, map_location="cpu", weights_only=True
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    try:
        head_weight = state["head.1.weight"].detach().cpu().numpy()
        head_bias = state["head.1.bias"].detach().cpu().numpy()
    except KeyError as error:
        raise ValueError(
            f"{VIT_CHECKPOINT} does not contain the expected ViT head"
        ) from error
    if head_weight.shape != (1, len(embedding_features)):
        raise ValueError(
            f"ViT head has shape {head_weight.shape}; expected "
            f"(1, {len(embedding_features)})"
        )
    vit_head = ViTCheckpointHead(
        head_weight, head_bias.item(), embedding_features
    )
    lyrics_forest = tree_pipeline(
        RandomForestRegressor(
            **MODEL_PARAMETERS,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    )
    combined_forest = tree_pipeline(
        RandomForestRegressor(
            **MODEL_PARAMETERS,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    )

    print(f"training three models on {len(x['train'])} tracks", flush=True)
    print(
        "using the fixed ViT checkpoint's trained Linear + sigmoid head",
        flush=True,
    )
    print(f"parameters: {MODEL_PARAMETERS}", flush=True)
    fit_seconds = {}
    fit_seconds["vit_checkpoint_head"] = 0.0

    started = time.perf_counter()
    lyrics_forest.fit(
        x["train"][lyric_features], y["train"]
    )
    fit_seconds["lyrics_random_forest"] = time.perf_counter() - started

    combined_train_x = stacked_features(
        x["train"], vit_head.predict(x["train"]), lyric_features
    )
    started = time.perf_counter()
    combined_forest.fit(combined_train_x, y["train"])
    fit_seconds["combined_stack"] = time.perf_counter() - started
    combined_model = ViTHeadLyricsRandomForest(
        vit_head, combined_forest, lyric_features
    )
    metrics = {
        "vit_checkpoint_head": {
            split: evaluate(
                vit_head,
                x[split],
                y[split],
                f"ViT checkpoint {split}",
            )
            for split in ("train", "validation", "test")
        },
        "lyrics_random_forest": {
            split: evaluate(
                lyrics_forest,
                x[split][lyric_features],
                y[split],
                f"lyrics Random Forest {split}",
            )
            for split in ("train", "validation", "test")
        },
        "combined_stack": {
            split: evaluate(
                combined_model,
                x[split],
                y[split],
                f"combined stack {split}",
            )
            for split in ("train", "validation", "test")
        },
    }

    artifact = {
        "models": {
            "vit_checkpoint_head": {
                "weight": head_weight,
                "bias": head_bias,
            },
            "lyrics_random_forest": lyrics_forest,
            "combined_random_forest": combined_forest,
        },
        "vit_head": "checkpoint Linear(768, 1) + sigmoid",
        "vit_checkpoint": str(VIT_CHECKPOINT),
        "model_parameters": MODEL_PARAMETERS,
        "metrics": metrics,
        "fit_seconds": fit_seconds,
        "input_feature_names": feature_names,
        "random_forest_feature_names": [
            VIT_HEAD_FEATURE,
            *lyric_features,
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
            {
                key: value
                for key, value in artifact.items()
                if key != "models"
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved model to {args.output}", flush=True)
    print(f"saved metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
