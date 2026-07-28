"""Evaluate trained neural networks on their original held-out test split."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.data_utils import (
    MelPopularityDataset,
    grouped_train_val_test_split,
    load_popularity_by_id,
    sync_to_local_scratch,
)
from training.train_CNN import PopularityCNN, collate_fn as cnn_collate
from training.train_CNN_LSTM import (
    PopularityCNNLSTM,
    collate_fn as cnn_lstm_collate,
)
from training.train_LSTM import PopularityLSTM, collate_fn as lstm_collate
from training.train_ResNet import PopularityResNet, collate_fn as resnet_collate
from training.train_ViT import PopularityViT, collate_fn as vit_collate


MODEL_CONFIGS = {
    "CNN": {
        "factory": PopularityCNN,
        "collate_fn": cnn_collate,
        "checkpoint": "best_CNN_fixed_model.pt",
        "batch_size": 16,
        "normalize": True,
    },
    "CNN_LSTM": {
        "factory": PopularityCNNLSTM,
        "collate_fn": cnn_lstm_collate,
        "checkpoint": "best_CNN_LSTM_fixed_model.pt",
        "batch_size": 32,
        "normalize": True,
    },
    "LSTM": {
        "factory": PopularityLSTM,
        "collate_fn": lstm_collate,
        "checkpoint": "best_LSTM_small_fixed_model.pt",
        "batch_size": 64,
        "normalize": False,
    },
    "ResNet": {
        # The checkpoint replaces every parameter, so evaluation need not
        # download ImageNet weights.
        "factory": lambda: PopularityResNet(weights=None),
        "collate_fn": resnet_collate,
        "checkpoint": "best_ResNet_fixed_model.pt",
        "batch_size": 16,
        "normalize": True,
    },
    "ViT": {
        "factory": lambda: PopularityViT(weights=None),
        "collate_fn": vit_collate,
        "checkpoint": "best_ViT_fixed_model.pt",
        "batch_size": 12,
        "normalize": True,
    },
}


def parse_args():
    user = os.environ.get("USER", "")
    scratch_root = Path("/net/scc1/scratch") / user
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or all NN checkpoints on the deterministic 15% "
            "holdout created by the corresponding training script."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model names (CNN CNN_LSTM LSTM ResNet ViT) or 'all'.",
    )
    parser.add_argument(
        "--mel-dir",
        type=Path,
        default=scratch_root / "mel_spectrograms",
        help="Directory containing *_mel.pt files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=scratch_root / "checkpoints",
    )
    parser.add_argument(
        "--local-scratch-dir",
        type=Path,
        default=None,
        help=(
            "If set, sync --mel-dir beneath this node-local directory before "
            "evaluation, as the training scripts do."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "nn_testing_fixed",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="For example cuda, cuda:0, or cpu.")
    return parser.parse_args()


def select_models(requested):
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(MODEL_CONFIGS)
    by_lower = {name.lower(): name for name in MODEL_CONFIGS}
    selected = []
    for value in requested:
        try:
            selected.append(by_lower[value.lower()])
        except KeyError as exc:
            choices = ", ".join(MODEL_CONFIGS)
            raise ValueError(f"unknown model {value!r}; choose from {choices}") from exc
    return selected


def make_test_loader(config, mel_dir, num_workers, device):
    dataset = MelPopularityDataset(
        str(mel_dir),
        load_popularity_by_id(),
        normalize=config["normalize"],
    )
    if len(dataset) < 3:
        raise RuntimeError(f"need at least 3 labeled mels, found {len(dataset)}")

    _, _, test_set = grouped_train_val_test_split(dataset)
    loader = DataLoader(
        test_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=config["collate_fn"],
    )
    return loader, len(dataset)


def load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint.get("epoch") if isinstance(checkpoint, dict) else None


def evaluate(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    with torch.inference_mode():
        for mels, batch_targets in loader:
            mels = mels.to(device, non_blocking=True)
            batch_predictions = model(mels)
            predictions.append(batch_predictions.detach().cpu().double())
            targets.append(batch_targets.double())

    predicted = torch.cat(predictions)
    actual = torch.cat(targets)
    errors = (predicted - actual) * 100.0
    actual_points = actual * 100.0
    predicted_points = predicted * 100.0
    mse = errors.square().mean().item()
    denominator = (actual_points - actual_points.mean()).square().sum()
    if denominator.item() == 0:
        r2 = float("nan")
    else:
        r2 = (
            1.0 - errors.square().sum().item() / denominator.item()
        )
    return {
        "test_samples": actual.numel(),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": errors.abs().mean().item(),
        "r2": r2,
        "prediction_mean": predicted_points.mean().item(),
        "prediction_std": predicted_points.std(unbiased=False).item(),
        "target_mean": actual_points.mean().item(),
        "target_std": actual_points.std(unbiased=False).item(),
    }


def write_results(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=True)
        handle.write("\n")
    fieldnames = ["model", *next(iter(results.values())).keys()]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in results.items():
            writer.writerow({"model": model_name, **metrics})
    return json_path, csv_path


def main():
    args = parse_args()
    selected = select_models(args.models)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if not args.mel_dir.is_dir():
        raise FileNotFoundError(f"mel directory not found: {args.mel_dir}")
    mel_dir = args.mel_dir
    if args.local_scratch_dir is not None:
        mel_dir = Path(
            sync_to_local_scratch(
                str(args.mel_dir),
                str(args.local_scratch_dir),
            )
        )

    print(f"evaluating {', '.join(selected)} on {device}", flush=True)
    results = {}
    for model_name in selected:
        config = MODEL_CONFIGS[model_name]
        loader, full_size = make_test_loader(
            config, mel_dir, args.num_workers, device
        )
        model = config["factory"]().to(device)
        checkpoint_path = args.checkpoint_dir / config["checkpoint"]
        epoch = load_checkpoint(model, checkpoint_path, device)
        metrics = evaluate(model, loader, device)
        metrics = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": epoch,
            "full_dataset_samples": full_size,
            **metrics,
        }
        results[model_name] = metrics
        print(
            f"{model_name:8s} | n={metrics['test_samples']:5d} | "
            f"MAE={metrics['mae']:.3f} | RMSE={metrics['rmse']:.3f} | "
            f"R2={metrics['r2']:.4f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    json_path, csv_path = write_results(results, args.output_dir)
    print(f"wrote {json_path} and {csv_path}", flush=True)


if __name__ == "__main__":
    main()
