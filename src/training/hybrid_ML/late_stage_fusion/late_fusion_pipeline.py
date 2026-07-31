"""Shared leakage-free late-stage fusion pipeline for all modalities."""

import argparse
import ast
from collections import Counter
import json
import re
import time
from pathlib import Path

import joblib
from langdetect import DetectorFactory, detect
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from training.data_utils import grouped_track_id_split
from training.hybrid_ML.intermediate_concatenation.train_intermediate_concatenation import (
    ARTISTS_CSV,
    CHECKPOINT_DIR,
    EMBEDDING_DIMENSIONS,
    LOW_LEVEL_CSV,
    LYRICS_CSV,
    RANDOM_SEED,
    SPOTIFY_AUDIO_FEATURES,
    TRACKS_CSV,
    load_embeddings,
    numeric_feature_frame,
    tree_pipeline,
)
from training.hybrid_ML.late_stage_fusion.generate_architecture_graph import (
    render_architecture_graph,
)


NEURAL_HEADS = {
    "cnn": {
        "embedding_name": "CNN",
        "checkpoint": "best_CNN_fixed_model.pt",
        "weight_key": "fc3.weight",
        "bias_key": "fc3.bias",
        "activation": "sigmoid",
    },
    "cnn_lstm": {
        "embedding_name": "CNN_LSTM",
        "checkpoint": "best_CNN_LSTM_fixed_model.pt",
        "weight_key": "fc2.weight",
        "bias_key": "fc2.bias",
        "activation": "identity",
    },
    "lstm": {
        "embedding_name": "LSTM",
        "checkpoint": "best_LSTM_small_fixed_model.pt",
        "weight_key": "fc.weight",
        "bias_key": "fc.bias",
        "activation": "sigmoid",
    },
    "resnet": {
        "embedding_name": "ResNet",
        "checkpoint": "best_ResNet_fixed_model.pt",
        "weight_key": "head.weight",
        "bias_key": "head.bias",
        "activation": "sigmoid",
    },
    "vit": {
        "embedding_name": "ViT",
        "checkpoint": "best_ViT_fixed_model.pt",
        "weight_key": "head.1.weight",
        "bias_key": "head.1.bias",
        "activation": "sigmoid",
    },
}

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
LYRICS_RF_PARAMETERS = {
    "n_estimators": 500,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "max_depth": 35,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}
AUDIO_XGBOOST_PARAMETERS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "random_state": 42,
    "n_jobs": -1,
}
AUDIO_RF_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}
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


class CachedNeuralHead:
    """Apply a trained checkpoint head to its matching cached embeddings."""

    def __init__(self, weight, bias, feature_names, activation):
        self.weight = np.asarray(weight, dtype=float).reshape(-1)
        self.bias = float(bias)
        self.feature_names = list(feature_names)
        self.activation = activation

    def predict(self, frame):
        values = (
            frame[self.feature_names].to_numpy() @ self.weight + self.bias
        )
        if self.activation == "sigmoid":
            values = 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))
        elif self.activation != "identity":
            raise ValueError(f"unknown head activation: {self.activation}")
        return np.clip(values * 100.0, 0.0, 100.0)


def drop_csv_index(frame):
    return frame.drop(
        columns=[
            column
            for column in frame.columns
            if column.startswith("Unnamed:")
        ],
        errors="ignore",
    )


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
    return 0.0 if not words else 1.0 - len(set(words)) / len(words)


def average_word_length(value):
    words = str(value).split()
    return 0.0 if not words else sum(map(len, words)) / len(words)


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


def load_lyrics_features():
    tracks = pd.read_csv(
        TRACKS_CSV,
        usecols=["id", "popularity", "lyrics"],
        dtype={"id": str},
    ).dropna(subset=["id", "popularity"])
    tracks = tracks.drop_duplicates("id")
    tracks["lyrics"] = tracks["lyrics"].apply(clean_lyric_text)

    lyrics = drop_csv_index(pd.read_csv(LYRICS_CSV)).rename(
        columns={"track_id": "id"}
    )
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


def load_audio_features():
    """Load every labeled track, imputing absent low-level extraction rows."""
    tracks = pd.read_csv(
        TRACKS_CSV,
        usecols=["id", "popularity", *SPOTIFY_AUDIO_FEATURES],
        dtype={"id": str},
    ).dropna(subset=["id", "popularity"]).drop_duplicates("id")
    low_level, low_level_features = numeric_feature_frame(
        LOW_LEVEL_CSV, "track_id", "low_level_"
    )
    low_level = low_level.drop_duplicates("id")
    frame = tracks.merge(
        low_level,
        on="id",
        how="left",
        validate="one_to_one",
    )
    features = [*SPOTIFY_AUDIO_FEATURES, *low_level_features]
    return frame[["id", "popularity", *features]], features


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


def load_artist_features():
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
    ).drop(columns=ARTIST_DROP_COLUMNS, errors="ignore")
    frame = frame.drop_duplicates("id")
    feature_names = [
        column for column in frame.columns if column not in {"id", "popularity"}
    ]
    nonnumeric = [
        column
        for column in feature_names
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if nonnumeric:
        raise ValueError(
            "artist inputs contain nonnumeric columns: "
            + ", ".join(nonnumeric)
        )
    return frame[["id", "popularity", *feature_names]], feature_names, top_genres


def load_neural_heads_and_embeddings():
    heads = {}
    frames = {}
    metadata = {}
    reference_ids = None
    for predictor_name, config in NEURAL_HEADS.items():
        embedding_name = config["embedding_name"]
        path, track_ids, frame, features = load_embeddings(
            embedding_name,
            EMBEDDING_DIMENSIONS[embedding_name],
            "_fixed",
        )
        checkpoint_path = CHECKPOINT_DIR / config["checkpoint"]
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        state = checkpoint.get("model_state_dict", checkpoint)
        weight = state[config["weight_key"]].detach().cpu().numpy()
        bias = state[config["bias_key"]].detach().cpu().numpy()
        if weight.shape != (1, len(features)):
            raise ValueError(
                f"{predictor_name} head has shape {weight.shape}; expected "
                f"(1, {len(features)})"
            )
        heads[predictor_name] = CachedNeuralHead(
            weight, bias.item(), features, config["activation"]
        )
        frames[predictor_name] = frame
        metadata[predictor_name] = {
            "embedding_path": str(path),
            "checkpoint": str(checkpoint_path),
            "feature_names": features,
            "track_count": len(track_ids),
        }
        if predictor_name == "vit":
            reference_ids = track_ids
    if reference_ids is None:
        raise RuntimeError("ViT embeddings are required as the split authority")
    return heads, frames, metadata, reference_ids


def score(actual, predicted):
    predicted = np.clip(np.asarray(predicted), 0.0, 100.0)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def fit_convex_weights(predictions, actual):
    predictions = np.asarray(predictions, dtype=float)
    actual = np.asarray(actual, dtype=float)
    model_count = predictions.shape[1]

    def objective(weights):
        residual = predictions @ weights - actual
        return float(np.mean(residual**2))

    def gradient(weights):
        residual = predictions @ weights - actual
        return 2.0 * predictions.T @ residual / len(actual)

    starts = [np.full(model_count, 1.0 / model_count)]
    starts.extend(np.eye(model_count))
    results = [
        minimize(
            objective,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * model_count,
            constraints={
                "type": "eq",
                "fun": lambda weights: weights.sum() - 1.0,
                "jac": lambda weights: np.ones_like(weights),
            },
            options={"ftol": 1e-12, "maxiter": 2_000},
        )
        for start in starts
    ]
    successful = [result for result in results if result.success]
    if not successful:
        messages = "; ".join(str(result.message) for result in results)
        raise RuntimeError(f"convex blend optimization failed: {messages}")
    best = min(successful, key=lambda result: result.fun)
    weights = np.clip(best.x, 0.0, 1.0)
    weights /= weights.sum()
    return weights, {
        "success": True,
        "message": str(best.message),
        "validation_mse": float(best.fun),
        "iterations": int(best.nit),
    }


def parse_args(include_artist):
    variant = "artist" if include_artist else "no_artist"
    default_output = (
        CHECKPOINT_DIR
        / "hybrid"
        / "late_stage_fusion"
        / variant
        / f"hybrid_{variant}.joblib"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"model artifact path (default: {default_output})",
    )
    return parser.parse_args()


def main(include_artist):
    args = parse_args(include_artist)
    np.random.seed(RANDOM_SEED)

    neural_heads, neural_frames, neural_metadata, all_ids = (
        load_neural_heads_and_embeddings()
    )
    train_ids, validation_ids, test_ids = grouped_track_id_split(all_ids)
    canonical_ids = {
        "train": set(train_ids),
        "validation": set(validation_ids),
        "test": set(test_ids),
    }

    lyrics = load_lyrics_features()
    audio, audio_features = load_audio_features()
    artist = None
    artist_features = []
    top_genres = []
    if include_artist:
        artist, artist_features, top_genres = load_artist_features()

    modality_frames = {
        **neural_frames,
        "audio": audio,
        "lyrics": lyrics,
    }
    if include_artist:
        modality_frames["artist"] = artist

    targets = pd.read_csv(
        TRACKS_CSV, usecols=["id", "popularity"], dtype={"id": str}
    ).dropna(subset=["id", "popularity"]).drop_duplicates("id")
    common = targets
    for name, frame in modality_frames.items():
        # Artist notebook inputs repeat Spotify audio columns already supplied
        # by the engineered-audio modality. Reuse those identical columns
        # instead of allowing pandas to suffix them and break feature lookup.
        columns = [
            column
            for column in frame.columns
            if column != "popularity"
            and (column == "id" or column not in common.columns)
        ]
        common = common.merge(
            frame[columns],
            on="id",
            how="inner",
            validate="one_to_one",
        )
    common = common.set_index("id")
    partitions = {
        name: common.loc[common.index.isin(ids)]
        for name, ids in canonical_ids.items()
    }
    if any(frame.empty for frame in partitions.values()):
        raise ValueError("at least one common late-fusion partition is empty")

    lyrics_train = lyrics.loc[
        lyrics["id"].isin(canonical_ids["train"])
    ].set_index("id")
    audio_train = audio.loc[
        audio["id"].isin(canonical_ids["train"])
    ].set_index("id")
    artist_train = None
    if include_artist:
        artist_train = artist.loc[
            artist["id"].isin(canonical_ids["train"])
        ].set_index("id")

    base_models = {
        **neural_heads,
        "audio_xgboost": tree_pipeline(
            XGBRegressor(**AUDIO_XGBOOST_PARAMETERS)
        ),
        "audio_random_forest": tree_pipeline(
            RandomForestRegressor(**AUDIO_RF_PARAMETERS)
        ),
        "lyrics_random_forest": tree_pipeline(
            RandomForestRegressor(**LYRICS_RF_PARAMETERS)
        ),
    }
    feature_sets = {
        **{name: None for name in neural_heads},
        "audio_xgboost": audio_features,
        "audio_random_forest": audio_features,
        "lyrics_random_forest": LYRIC_FEATURES,
    }
    training_frames = {
        "audio_xgboost": audio_train,
        "audio_random_forest": audio_train,
        "lyrics_random_forest": lyrics_train,
    }
    if include_artist:
        base_models["artist_xgboost"] = XGBRegressor(
            **ARTIST_XGBOOST_PARAMETERS
        )
        feature_sets["artist_xgboost"] = artist_features
        training_frames["artist_xgboost"] = artist_train

    variant = "artist" if include_artist else "no_artist"
    print(
        f"late-stage fusion variant={variant}; "
        + ", ".join(
            f"canonical_{name}={len(ids)}"
            for name, ids in canonical_ids.items()
        ),
        flush=True,
    )
    print(
        "modality availability: "
        + ", ".join(
            f"{name}={len(frame)}" for name, frame in modality_frames.items()
        ),
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
        "modality training population: "
        + ", ".join(
            f"{name}={len(frame)}"
            for name, frame in training_frames.items()
        ),
        flush=True,
    )

    fit_seconds = {name: 0.0 for name in neural_heads}
    for name, train_frame in training_frames.items():
        started = time.perf_counter()
        features = feature_sets[name]
        base_models[name].fit(
            train_frame[features], train_frame["popularity"]
        )
        fit_seconds[name] = time.perf_counter() - started
        print(f"trained {name} in {fit_seconds[name]:.1f}s", flush=True)

    predictor_names = list(base_models)
    prediction_matrices = {}
    metrics = {name: {} for name in predictor_names}
    for split, frame in partitions.items():
        columns = []
        actual = frame["popularity"]
        for name in predictor_names:
            features = feature_sets[name]
            inputs = frame if features is None else frame[features]
            predicted = np.clip(
                base_models[name].predict(inputs), 0.0, 100.0
            )
            columns.append(predicted)
            metrics[name][split] = score(actual, predicted)
            values = metrics[name][split]
            print(
                f"{name} {split}: MAE {values['mae']:.3f} | "
                f"RMSE {values['rmse']:.3f} | R2 {values['r2']:.3f}",
                flush=True,
            )
        prediction_matrices[split] = np.column_stack(columns)

    weights, optimization = fit_convex_weights(
        prediction_matrices["validation"],
        partitions["validation"]["popularity"],
    )
    weight_map = dict(zip(predictor_names, map(float, weights)))
    print(
        "best convex blend: "
        + ", ".join(f"{name}={weight_map[name]:.4f}" for name in predictor_names),
        flush=True,
    )

    blend_name = f"hybrid_{variant}"
    metrics[blend_name] = {}
    for split, frame in partitions.items():
        predicted = prediction_matrices[split] @ weights
        metrics[blend_name][split] = score(frame["popularity"], predicted)
        values = metrics[blend_name][split]
        print(
            f"{blend_name} {split}: MAE {values['mae']:.3f} | "
            f"RMSE {values['rmse']:.3f} | R2 {values['r2']:.3f}",
            flush=True,
        )

    artifact = {
        "models": base_models,
        "predictor_names": predictor_names,
        "feature_sets": feature_sets,
        "blend_weights": weight_map,
        "blend_optimization": optimization,
        "metrics": metrics,
        "fit_seconds": fit_seconds,
        "variant": variant,
        "fusion": "late_stage_prediction_fusion",
        "split": "artist_album_isolated_fixed_70_15_15",
        "split_population": "fixed_vit_embedding_track_ids",
        "canonical_split_counts": {
            name: len(ids) for name, ids in canonical_ids.items()
        },
        "common_split_counts": {
            name: len(frame) for name, frame in partitions.items()
        },
        "modality_availability_counts": {
            name: len(frame) for name, frame in modality_frames.items()
        },
        "training_population_counts": {
            **{
                name: len(canonical_ids["train"] & set(frame["id"]))
                for name, frame in neural_frames.items()
            },
            **{
                name: len(frame)
                for name, frame in training_frames.items()
            },
        },
        "neural_metadata": neural_metadata,
        "audio_features": audio_features,
        "lyrics_features": LYRIC_FEATURES,
        "artist_features": artist_features,
        "top_artist_genres": top_genres,
        "parameters": {
            "lyrics_random_forest": LYRICS_RF_PARAMETERS,
            "audio_xgboost": AUDIO_XGBOOST_PARAMETERS,
            "audio_random_forest": AUDIO_RF_PARAMETERS,
            "artist_xgboost": (
                ARTIST_XGBOOST_PARAMETERS if include_artist else None
            ),
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
    weights_path = args.output.with_suffix(".blend_weights.csv")
    pd.DataFrame(
        [
            {"predictor": name, "weight": weight_map[name]}
            for name in predictor_names
        ]
    ).sort_values("weight", ascending=False).to_csv(weights_path, index=False)
    architecture_path = args.output.with_suffix(".architecture.png")
    render_architecture_graph(
        weight_map,
        architecture_path,
        metrics[blend_name]["test"],
    )
    print(f"saved model to {args.output}", flush=True)
    print(f"saved metrics to {metrics_path}", flush=True)
    print(f"saved blend weights to {weights_path}", flush=True)
    print(f"saved architecture graph to {architecture_path}", flush=True)
