"""Run the missing pairwise/full multimodal ablations on one fixed population.

The existing single-modality experiments cover audio-only, metadata-only, and
lyrics-only.  This script fills in:

* audio + metadata
* audio + lyrics
* metadata + lyrics
* audio + metadata + lyrics

Every row uses the same tracks and the same artist/album-isolated split, so
changes in the metrics reflect feature ablation rather than changing samples.
The audio modality is the fixed ViT embedding plus Spotify and low-level audio
features; embeddings from the other neural architectures are not used.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from training.data_utils import grouped_track_id_split
from training.hybrid_ML.train_all_embeddings_hybrid import (
    CHECKPOINT_DIR,
    EMBEDDING_DIMENSIONS,
    RANDOM_SEED,
    build_models,
    load_embeddings,
    load_tabular_features,
)


DEFAULT_OUTPUT_DIR = CHECKPOINT_DIR / "modality_ablations"
ABLATIONS = {
    "audio_metadata": ("audio", "metadata"),
    "audio_lyrics": ("audio", "lyrics"),
    "metadata_lyrics": ("metadata", "lyrics"),
    "audio_metadata_lyrics": ("audio", "metadata", "lyrics"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def metrics(actual, predicted):
    predicted = np.clip(np.asarray(predicted), 0.0, 100.0)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def main():
    args = parse_args()
    np.random.seed(RANDOM_SEED)
    estimator, _ = build_models("cuda")["xgboost"]

    embedding_path, split_track_ids, embeddings, embedding_features = (
        load_embeddings(
            "ViT", EMBEDDING_DIMENSIONS["ViT"], "_fixed"
        )
    )
    embedding_paths = {"ViT": str(embedding_path)}
    tabular, tabular_groups = load_tabular_features()

    # This inner merge is performed once, before any ablation. Lyrics and
    # metadata remain left-joined/imputed inside load_tabular_features(), so
    # all ablations below have identical rows.
    combined = embeddings.merge(
        tabular, on="id", how="inner", validate="one_to_one"
    ).set_index("id")
    train_ids, validation_ids, test_ids = map(
        set, grouped_track_id_split(split_track_ids)
    )
    partitions = {
        "train": combined.loc[combined.index.isin(train_ids)],
        "validation": combined.loc[combined.index.isin(validation_ids)],
        "test": combined.loc[combined.index.isin(test_ids)],
    }
    if any(frame.empty for frame in partitions.values()):
        raise ValueError("at least one data partition is empty")

    modality_features = {
        "audio": [
            *embedding_features,
            *tabular_groups["audio"],
        ],
        "metadata": tabular_groups["artist_stats"],
        "lyrics": tabular_groups["lyrics"],
    }
    y = {
        split: frame["popularity"].to_numpy()
        for split, frame in partitions.items()
    }

    rows = []
    train_mean = float(np.mean(y["train"]))
    for split in partitions:
        result = metrics(
            y[split], np.full(len(y[split]), train_mean, dtype=float)
        )
        rows.append(
            {
                "ablation": "train_mean_baseline",
                "modalities": "none",
                "model": "train_mean",
                "split": split,
                "samples": len(y[split]),
                "features": 0,
                "fit_samples": len(y["train"]),
                "fit_seconds": 0.0,
                **result,
            }
        )

    for ablation_name, modalities in ABLATIONS.items():
        feature_names = [
            feature
            for modality in modalities
            for feature in modality_features[modality]
        ]
        x = {
            split: frame[feature_names]
            for split, frame in partitions.items()
        }
        print(
            f"\n{ablation_name}: {len(feature_names)} features; "
            + ", ".join(
                f"{split}={len(frame)}"
                for split, frame in partitions.items()
            ),
            flush=True,
        )
        model = clone(estimator)
        fit_x = x["train"]
        fit_y = y["train"]
        print(
            f"training GPU XGBoost on {len(fit_x)} samples...",
            flush=True,
        )
        started = time.perf_counter()
        model.fit(fit_x, fit_y)
        fit_seconds = time.perf_counter() - started
        for split in partitions:
            result = metrics(y[split], model.predict(x[split]))
            print(
                f"xgboost {split}: MAE {result['mae']:.3f} | "
                f"RMSE {result['rmse']:.3f} | "
                f"R2 {result['r2']:.3f}",
                flush=True,
            )
            rows.append(
                {
                    "ablation": ablation_name,
                    "modalities": "+".join(modalities),
                    "model": "xgboost",
                    "split": split,
                    "samples": len(y[split]),
                    "features": len(feature_names),
                    "fit_samples": len(fit_x),
                    "fit_seconds": fit_seconds,
                    **result,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    metrics_path = args.output_dir / "metrics.csv"
    results.to_csv(metrics_path, index=False)

    table_columns = ["ablation", "model", "mae", "rmse", "r2"]
    for split in ("validation", "test"):
        table = (
            results.loc[results["split"].eq(split), table_columns]
            .sort_values(["rmse", "mae"])
            .reset_index(drop=True)
        )
        table.to_csv(args.output_dir / f"{split}_table.csv", index=False)

    manifest = {
        "split": "artist_album_isolated_fixed_70_15_15",
        "random_seed": RANDOM_SEED,
        "split_counts": {
            split: len(frame) for split, frame in partitions.items()
        },
        "train_target_mean": train_mean,
        "embedding_paths": embedding_paths,
        "audio_embedding": "ViT_fixed",
        "modality_feature_counts": {
            name: len(features)
            for name, features in modality_features.items()
        },
        "ablations": {
            name: list(modalities)
            for name, modalities in ABLATIONS.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote ablation outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
