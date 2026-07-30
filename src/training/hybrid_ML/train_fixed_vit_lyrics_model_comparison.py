"""Compare lyrics RF, the trained ViT head, and a convex blend.

The fixed ViT embedding population defines the checkpoint's canonical
artist/album-isolated split. The lyrics RF trains on every valid lyrics example
in the canonical training partition; ViT and lyrics are intersected only
within the fixed partitions for comparison and blending. The original fixed
ViT checkpoint's trained Linear + sigmoid head reduces each cached ViT
embedding to its popularity prediction. A single convex-blend weight is
selected on validation RMSE and then evaluated once on the untouched test
partition.
"""

import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from langdetect import DetectorFactory, detect
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from training.data_utils import grouped_track_id_split
from training.hybrid_ML.train_all_embeddings_hybrid import (
    CHECKPOINT_DIR,
    EMBEDDING_DIMENSIONS,
    LYRICS_CSV,
    RANDOM_SEED,
    TRACKS_CSV,
    evaluate,
    load_embeddings,
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
RF_RANDOM_SEED = 42
VIT_CHECKPOINT = CHECKPOINT_DIR / "best_ViT_fixed_model.pt"
LYRIC_FEATURES = [
    "mean_syllables_word",
    "mean_words_sentence",
    "n_sentences",
    "n_words",
    "sentence_similarity",
    "vocabulary_wealth",
    "word_density",
    "is_english",
    "repetition_score",
    "avg_word_length",
    "syllables_per_line",
    "syllables_per_word",
]


def clean_lyric_text(value):
    if pd.isna(value) or not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[\r\n]+", " ", value)).strip()


def lyric_language(value):
    try:
        if len(str(value).strip()) < 10:
            return "unknown"
        return detect(value)
    except Exception:
        return "unknown"


def repetition_score(value):
    words = str(value).lower().split()
    if not words:
        return 0.0
    return 1.0 - len(set(words)) / len(words)


def average_word_length(value):
    words = str(value).split()
    if not words:
        return 0.0
    return sum(map(len, words)) / len(words)


def syllable_features(value):
    lines = [line.strip() for line in str(value).split("\n") if line.strip()]
    if not lines:
        return 0.0, 0.0

    def count_syllables(word):
        return max(1, len(re.findall(r"[aeiouy]+", word.lower())))

    line_syllables = []
    total_words = 0
    total_syllables = 0
    for line in lines:
        words = re.findall(r"\b\w+\b", line.lower())
        syllables = sum(count_syllables(word) for word in words)
        line_syllables.append(syllables)
        total_words += len(words)
        total_syllables += syllables
    return (
        float(np.mean(line_syllables)),
        total_syllables / total_words if total_words else 0.0,
    )


def load_notebook_lyrics_features():
    """Reproduce the 12-feature population from popularity_predictor2."""
    tracks = pd.read_csv(
        TRACKS_CSV,
        usecols=["id", "popularity", "lyrics"],
        dtype={"id": str},
    ).dropna(subset=["id", "popularity"])
    tracks = tracks.drop_duplicates("id")
    tracks["lyrics"] = tracks["lyrics"].apply(clean_lyric_text)

    lyrics = pd.read_csv(LYRICS_CSV)
    lyrics = lyrics.drop(
        columns=[
            column
            for column in lyrics.columns
            if column.startswith("Unnamed:")
        ],
        errors="ignore",
    ).rename(columns={"track_id": "id"})
    lyrics["id"] = lyrics["id"].astype(str)
    lyrics = lyrics.loc[lyrics["mean_syllables_word"].ne(-1)].drop_duplicates(
        "id"
    )
    frame = lyrics.merge(tracks, on="id", how="inner", validate="one_to_one")
    frame["word_density"] = frame["n_words"] / (
        frame["n_sentences"] + 1e-5
    )
    DetectorFactory.seed = 0
    frame["is_english"] = frame["lyrics"].apply(
        lambda text: int(lyric_language(text) == "en")
    )
    frame["repetition_score"] = frame["lyrics"].apply(repetition_score)
    frame["avg_word_length"] = frame["lyrics"].apply(average_word_length)
    syllables = frame["lyrics"].apply(syllable_features)
    frame["syllables_per_line"] = syllables.str[0]
    frame["syllables_per_word"] = syllables.str[1]
    return frame[["id", "popularity", *LYRIC_FEATURES]]


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


class ConvexPredictionBlend:
    """Blend ViT and lyrics-RF predictions with a fixed validation weight."""

    def __init__(self, vit_head, forest_model, lyric_features, vit_weight):
        self.vit_head = vit_head
        self.forest_model = forest_model
        self.lyric_features = list(lyric_features)
        self.vit_weight = float(vit_weight)

    def predict(self, frame):
        vit_predictions = self.vit_head.predict(frame)
        lyrics_predictions = self.forest_model.predict(
            frame[self.lyric_features]
        )
        return (
            self.vit_weight * vit_predictions
            + (1.0 - self.vit_weight) * lyrics_predictions
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

    embedding_path, all_ids, embeddings, embedding_features = (
        load_embeddings(
            "ViT", EMBEDDING_DIMENSIONS["ViT"], "_fixed"
        )
    )
    embedding_paths = {"ViT": str(embedding_path)}
    embedding_groups = {"ViT": embedding_features}

    tabular = load_notebook_lyrics_features()
    lyric_features = LYRIC_FEATURES
    train_ids, validation_ids, test_ids = grouped_track_id_split(all_ids)
    canonical_ids = {
        "train": set(train_ids),
        "validation": set(validation_ids),
        "test": set(test_ids),
    }
    lyrics_train = tabular.loc[
        tabular["id"].isin(canonical_ids["train"])
    ].set_index("id")
    if lyrics_train.empty:
        raise ValueError("no lyrics examples are available for training")

    combined = embeddings.merge(
        tabular, on="id", how="inner", validate="one_to_one"
    ).set_index("id")
    partitions = {
        name: combined.loc[combined.index.isin(ids)]
        for name, ids in canonical_ids.items()
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
            random_state=RF_RANDOM_SEED,
        )
    )
    print(
        "canonical ViT split: "
        + ", ".join(
            f"{name}={len(ids)}" for name, ids in canonical_ids.items()
        ),
        flush=True,
    )
    print(
        f"training lyrics Random Forest on {len(lyrics_train)} tracks; "
        f"common comparison train={len(x['train'])}",
        flush=True,
    )
    print(
        "using the fixed ViT checkpoint's trained Linear + sigmoid head",
        flush=True,
    )
    print(f"parameters: {MODEL_PARAMETERS}", flush=True)
    fit_seconds = {}
    fit_seconds["vit_checkpoint_head"] = 0.0

    started = time.perf_counter()
    lyrics_forest.fit(
        lyrics_train[lyric_features], lyrics_train["popularity"]
    )
    fit_seconds["lyrics_random_forest"] = time.perf_counter() - started

    started = time.perf_counter()
    validation_vit = vit_head.predict(x["validation"])
    validation_lyrics = lyrics_forest.predict(
        x["validation"][lyric_features]
    )
    blend_rows = []
    for vit_weight in np.linspace(0.0, 1.0, 101):
        predictions = (
            vit_weight * validation_vit
            + (1.0 - vit_weight) * validation_lyrics
        )
        blend_rows.append(
            {
                "vit_weight": float(vit_weight),
                "lyrics_weight": float(1.0 - vit_weight),
                "validation_mae": float(
                    mean_absolute_error(y["validation"], predictions)
                ),
                "validation_rmse": float(
                    mean_squared_error(y["validation"], predictions) ** 0.5
                ),
                "validation_r2": float(
                    r2_score(y["validation"], predictions)
                ),
            }
        )
    best_blend = min(blend_rows, key=lambda row: row["validation_rmse"])
    combined_model = ConvexPredictionBlend(
        vit_head,
        lyrics_forest,
        lyric_features,
        best_blend["vit_weight"],
    )
    fit_seconds["convex_blend_search"] = time.perf_counter() - started
    print(
        f"best blend: ViT={best_blend['vit_weight']:.2f}, "
        f"lyrics RF={best_blend['lyrics_weight']:.2f}, "
        f"validation RMSE={best_blend['validation_rmse']:.3f}",
        flush=True,
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
        "convex_blend": {
            split: evaluate(
                combined_model,
                x[split],
                y[split],
                f"convex blend {split}",
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
        },
        "vit_head": "checkpoint Linear(768, 1) + sigmoid",
        "vit_checkpoint": str(VIT_CHECKPOINT),
        "model_parameters": MODEL_PARAMETERS,
        "random_forest_seed": RF_RANDOM_SEED,
        "blend_selection": {
            "criterion": "validation_rmse",
            "candidates": 101,
            "best": best_blend,
        },
        "metrics": metrics,
        "fit_seconds": fit_seconds,
        "input_feature_names": feature_names,
        "random_forest_feature_names": lyric_features,
        "embedding_feature_names": embedding_groups,
        "embedding_paths": embedding_paths,
        "modalities": ["ViT_fixed_embeddings", "lyrics"],
        "split": "artist_album_isolated_fixed_70_15_15",
        "split_population": "fixed_vit_embedding_track_ids",
        "split_counts": {
            name: len(frame) for name, frame in partitions.items()
        },
        "canonical_split_counts": {
            name: len(ids) for name, ids in canonical_ids.items()
        },
        "training_population_counts": {
            "lyrics_random_forest": len(lyrics_train),
        },
        "random_seed": RANDOM_SEED,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    blend_path = args.output.with_suffix(".blend_weights.csv")
    pd.DataFrame(blend_rows).to_csv(blend_path, index=False)
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
    print(f"saved blend weights to {blend_path}", flush=True)


if __name__ == "__main__":
    main()
