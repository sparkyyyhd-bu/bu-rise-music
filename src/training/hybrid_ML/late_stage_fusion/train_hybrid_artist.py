"""Blend fixed ViT, notebook-style lyrics RF, and artist XGBoost predictions.

The fixed ViT embedding population defines one canonical artist/album-isolated
split, matching the population available to the fixed ViT checkpoint. Lyrics
and artist models train on every example their modality has in the canonical
training partition. Modalities are intersected only for blend selection and
evaluation. The ViT prediction comes from the fixed checkpoint's trained
Linear + sigmoid head. Lyrics and artist models reproduce
popularity_predictor2.ipynb and artists_xgboost.ipynb respectively.
Nonnegative blend weights summing to one are selected on validation RMSE and
evaluated once on the untouched test set.
"""

import argparse
import ast
from collections import Counter
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from training.data_utils import grouped_track_id_split
from training.hybrid_ML.intermediate_concatenation.train_intermediate_concatenation import (
    ARTISTS_CSV,
    CHECKPOINT_DIR,
    EMBEDDING_DIMENSIONS,
    RANDOM_SEED,
    TRACKS_CSV,
    evaluate,
    load_embeddings,
)
from training.hybrid_ML.late_stage_fusion.train_hybrid_no_artist import (
    LYRIC_FEATURES,
    MODEL_PARAMETERS as LYRICS_RF_PARAMETERS,
    RF_RANDOM_SEED,
    VIT_CHECKPOINT,
    ViTCheckpointHead,
    load_notebook_lyrics_features,
    tree_pipeline,
)
from sklearn.ensemble import RandomForestRegressor


DEFAULT_OUTPUT = (
    CHECKPOINT_DIR
    / "hybrid"
    / "late_stage_fusion"
    / "artist"
    / "hybrid_artist.joblib"
)
ARTIST_XGBOOST_PARAMETERS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}
ARTIST_DROP_COLUMNS = [
    "track_id",
    "name",
    "track_name_prev",
    "artists_id",
    "album_id",
    "analysis_url",
    "href",
    "preview_url",
    "track_href",
    "uri",
    "lyrics",
    "available_markets",
    "country",
    "type",
    "track_number",
    "disc_number",
    "playlist",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"model artifact path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def drop_csv_index(frame):
    return frame.drop(
        columns=[
            column
            for column in frame.columns
            if column.startswith("Unnamed:")
        ],
        errors="ignore",
    )


def parse_genres(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def load_artist_notebook_features():
    """Reproduce the 116 inputs built by artists_xgboost.ipynb."""
    artists = drop_csv_index(pd.read_csv(ARTISTS_CSV))
    songs = drop_csv_index(pd.read_csv(TRACKS_CSV))
    artists["track_id"] = artists["track_id"].astype(str)
    songs["id"] = songs["id"].astype(str)
    artists["genres"] = artists["genres"].apply(parse_genres)

    genre_counts = Counter()
    for genres in artists["genres"]:
        genre_counts.update(genres)
    top_genres = [genre for genre, _count in genre_counts.most_common(100)]
    genre_columns = {
        f"genre_{genre}": artists["genres"].apply(
            lambda values, selected=genre: int(selected in values)
        )
        for genre in top_genres
    }
    artists = pd.concat(
        [artists, pd.DataFrame(genre_columns, index=artists.index)], axis=1
    )
    artist_features = (
        artists.groupby("track_id")
        .agg(
            artist_popularity=("artist_popularity", "mean"),
            followers=("followers", "mean"),
            num_artists=("id", "count"),
            **{
                f"genre_{genre}": (f"genre_{genre}", "max")
                for genre in top_genres
            },
        )
        .reset_index()
    )
    frame = songs.merge(
        artist_features,
        left_on="id",
        right_on="track_id",
        how="inner",
        validate="one_to_many",
    )
    frame = frame.drop(columns=ARTIST_DROP_COLUMNS, errors="ignore")
    nonnumeric = [
        column
        for column in frame.columns
        if column != "id"
        and not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if nonnumeric:
        raise ValueError(
            "artist notebook inputs contain nonnumeric columns: "
            + ", ".join(nonnumeric)
        )
    frame = frame.drop_duplicates("id")
    feature_names = [
        column for column in frame.columns if column not in {"id", "popularity"}
    ]
    return frame[["id", "popularity", *feature_names]], feature_names, top_genres


class ThreePredictionBlend:
    def __init__(
        self,
        vit_head,
        lyrics_model,
        artist_model,
        lyric_features,
        artist_features,
        weights,
    ):
        self.vit_head = vit_head
        self.lyrics_model = lyrics_model
        self.artist_model = artist_model
        self.lyric_features = list(lyric_features)
        self.artist_features = list(artist_features)
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, frame):
        predictions = np.column_stack(
            [
                self.vit_head.predict(frame),
                self.lyrics_model.predict(frame[self.lyric_features]),
                self.artist_model.predict(frame[self.artist_features]),
            ]
        )
        return predictions @ self.weights


def score_blend(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def main():
    args = parse_args()
    np.random.seed(RANDOM_SEED)

    embedding_path, all_ids, embeddings, embedding_features = load_embeddings(
        "ViT", EMBEDDING_DIMENSIONS["ViT"], "_fixed"
    )
    lyrics = load_notebook_lyrics_features()
    artist, artist_features, top_genres = load_artist_notebook_features()

    # Split before intersecting modalities. This preserves the fixed ViT
    # checkpoint's evaluation boundary while allowing each tabular model to
    # use all of its available canonical-training examples.
    train_ids, validation_ids, test_ids = grouped_track_id_split(all_ids)
    canonical_ids = {
        "train": set(train_ids),
        "validation": set(validation_ids),
        "test": set(test_ids),
    }
    lyrics_train = lyrics.loc[
        lyrics["id"].isin(canonical_ids["train"])
    ].set_index("id")
    artist_train = artist.loc[
        artist["id"].isin(canonical_ids["train"])
    ].set_index("id")
    if lyrics_train.empty or artist_train.empty:
        raise ValueError("at least one modality has no canonical training data")

    combined = (
        embeddings.merge(
            lyrics, on="id", how="inner", validate="one_to_one"
        )
        .merge(
            artist.drop(columns="popularity"),
            on="id",
            how="inner",
            validate="one_to_one",
        )
        .set_index("id")
    )
    partitions = {
        name: combined.loc[combined.index.isin(ids)]
        for name, ids in canonical_ids.items()
    }
    if any(frame.empty for frame in partitions.values()):
        raise ValueError("at least one full-hybrid partition is empty")
    y = {name: frame["popularity"] for name, frame in partitions.items()}

    checkpoint = torch.load(
        VIT_CHECKPOINT, map_location="cpu", weights_only=True
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    head_weight = state["head.1.weight"].detach().cpu().numpy()
    head_bias = state["head.1.bias"].detach().cpu().numpy()
    vit_head = ViTCheckpointHead(
        head_weight, head_bias.item(), embedding_features
    )
    lyrics_model = tree_pipeline(
        RandomForestRegressor(
            **LYRICS_RF_PARAMETERS,
            random_state=RF_RANDOM_SEED,
            n_jobs=-1,
        )
    )
    artist_model = XGBRegressor(**ARTIST_XGBOOST_PARAMETERS)

    print(
        "canonical ViT split: "
        + ", ".join(
            f"{name}={len(ids)}" for name, ids in canonical_ids.items()
        ),
        flush=True,
    )
    print(
        "modality training population: "
        f"lyrics={len(lyrics_train)}, artist={len(artist_train)}",
        flush=True,
    )
    print(
        "common blend population: "
        + ", ".join(
            f"{name}={len(frame)}" for name, frame in partitions.items()
        ),
        flush=True,
    )
    print(
        f"features: ViT={len(embedding_features)}, "
        f"lyrics={len(LYRIC_FEATURES)}, artist={len(artist_features)}",
        flush=True,
    )
    fit_seconds = {}
    started = time.perf_counter()
    lyrics_model.fit(
        lyrics_train[LYRIC_FEATURES], lyrics_train["popularity"]
    )
    fit_seconds["lyrics_random_forest"] = time.perf_counter() - started
    print(
        f"trained lyrics Random Forest in "
        f"{fit_seconds['lyrics_random_forest']:.1f}s",
        flush=True,
    )
    started = time.perf_counter()
    artist_model.fit(
        artist_train[artist_features], artist_train["popularity"]
    )
    fit_seconds["artist_xgboost"] = time.perf_counter() - started
    print(
        f"trained artist XGBoost in {fit_seconds['artist_xgboost']:.1f}s",
        flush=True,
    )

    validation_predictions = np.column_stack(
        [
            vit_head.predict(partitions["validation"]),
            lyrics_model.predict(
                partitions["validation"][LYRIC_FEATURES]
            ),
            artist_model.predict(
                partitions["validation"][artist_features]
            ),
        ]
    )
    blend_rows = []
    for vit_percent in range(101):
        for lyrics_percent in range(101 - vit_percent):
            artist_percent = 100 - vit_percent - lyrics_percent
            weights = np.asarray(
                [vit_percent, lyrics_percent, artist_percent], dtype=float
            ) / 100.0
            predicted = validation_predictions @ weights
            blend_rows.append(
                {
                    "vit_weight": weights[0],
                    "lyrics_weight": weights[1],
                    "artist_weight": weights[2],
                    **{
                        f"validation_{name}": value
                        for name, value in score_blend(
                            y["validation"], predicted
                        ).items()
                    },
                }
            )
    best_blend = min(blend_rows, key=lambda row: row["validation_rmse"])
    weights = [
        best_blend["vit_weight"],
        best_blend["lyrics_weight"],
        best_blend["artist_weight"],
    ]
    blend_model = ThreePredictionBlend(
        vit_head,
        lyrics_model,
        artist_model,
        LYRIC_FEATURES,
        artist_features,
        weights,
    )
    print(
        "best blend: "
        f"ViT={weights[0]:.2f}, lyrics={weights[1]:.2f}, "
        f"artist={weights[2]:.2f}, "
        f"validation RMSE={best_blend['validation_rmse']:.3f}",
        flush=True,
    )

    predictors = {
        "vit_checkpoint_head": (vit_head, None),
        "lyrics_random_forest": (lyrics_model, LYRIC_FEATURES),
        "artist_xgboost": (artist_model, artist_features),
        "hybrid_artist": (blend_model, None),
    }
    metrics = {}
    for model_name, (model, features) in predictors.items():
        metrics[model_name] = {}
        for split, frame in partitions.items():
            inputs = frame if features is None else frame[features]
            metrics[model_name][split] = evaluate(
                model, inputs, y[split], f"{model_name} {split}"
            )

    artifact = {
        "models": {
            "vit_checkpoint_head": {
                "weight": head_weight,
                "bias": head_bias,
            },
            "lyrics_random_forest": lyrics_model,
            "artist_xgboost": artist_model,
        },
        "best_blend": best_blend,
        "metrics": metrics,
        "fit_seconds": fit_seconds,
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
            "artist_xgboost": len(artist_train),
        },
        "embedding_path": str(embedding_path),
        "vit_checkpoint": str(VIT_CHECKPOINT),
        "embedding_features": embedding_features,
        "lyrics_features": LYRIC_FEATURES,
        "artist_features": artist_features,
        "top_artist_genres": top_genres,
        "lyrics_random_forest_parameters": LYRICS_RF_PARAMETERS,
        "artist_xgboost_parameters": ARTIST_XGBOOST_PARAMETERS,
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
    blend_path = args.output.with_suffix(".blend_weights.csv")
    pd.DataFrame(blend_rows).sort_values("validation_rmse").to_csv(
        blend_path, index=False
    )
    print(f"saved model to {args.output}", flush=True)
    print(f"saved metrics to {metrics_path}", flush=True)
    print(f"saved blend weights to {blend_path}", flush=True)


if __name__ == "__main__":
    main()
