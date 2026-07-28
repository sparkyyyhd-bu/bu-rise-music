"""Extract embeddings from every reproducible best neural-network checkpoint."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from audio_config import MAX_FRAMES
from training.data_utils import sync_to_local_scratch
from training.train_CNN import PopularityCNN
from training.train_CNN_LSTM import PopularityCNNLSTM
from training.train_LSTM import PopularityLSTM
from training.train_ResNet import PopularityResNet
from training.train_ViT import PopularityViT


REPO_ROOT = Path(__file__).resolve().parents[3]
USER = os.environ["USER"]
NETWORK_MEL_DIR = Path("/net/scc1/scratch") / USER / "mel_spectrograms"
LOCAL_SCRATCH_DIR = Path("/scratch") / USER
CHECKPOINT_DIR = Path("/net/scc1/scratch") / USER / "checkpoints"
TRACKS_CSV = (
    REPO_ROOT
    / "data"
    / "SpotGenTrack"
    / "Data Sources"
    / "spotify_tracks.csv"
)
MEL_SUFFIX = "_mel.pt"
NUM_WORKERS = 8

MODEL_CONFIGS = {
    "CNN": {
        "checkpoint": "best_CNN_fixed_model.pt",
        "factory": PopularityCNN,
        "dimensions": 64,
        "batch_size": 32,
        "normalize": True,
    },
    "CNN_LSTM": {
        "checkpoint": "best_CNN_LSTM_fixed_model.pt",
        "factory": PopularityCNNLSTM,
        "dimensions": 128,
        "batch_size": 32,
        "normalize": True,
    },
    "LSTM": {
        "checkpoint": "best_LSTM_small_fixed_model.pt",
        "factory": PopularityLSTM,
        "dimensions": 512,
        "batch_size": 64,
        "normalize": False,
    },
    "ResNet": {
        "checkpoint": "best_ResNet_fixed_model.pt",
        "factory": lambda: PopularityResNet(weights=None),
        "dimensions": 512,
        "batch_size": 32,
        "normalize": True,
    },
    "ViT": {
        "checkpoint": "best_ViT_fixed_model.pt",
        "factory": lambda: PopularityViT(weights=None),
        "dimensions": 768,
        "batch_size": 12,
        "normalize": True,
    },
}


class MelTrackDataset(Dataset):
    def __init__(self, mel_dir, labeled_ids, normalize):
        self.entries = [
            (entry.path, entry.name[: -len(MEL_SUFFIX)])
            for entry in os.scandir(mel_dir)
            if entry.is_file()
            and entry.name.endswith(MEL_SUFFIX)
            and entry.name[: -len(MEL_SUFFIX)] in labeled_ids
        ]
        self.entries.sort(key=lambda item: item[1])
        self.normalize = normalize
        if not self.entries:
            raise ValueError(f"no labeled {MEL_SUFFIX} files found in {mel_dir}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, track_id = self.entries[index]
        mel = torch.load(path, map_location="cpu", weights_only=True)
        if mel.dim() == 3:
            mel = mel.mean(dim=0)
        mel = mel.float()
        if self.normalize:
            mel = (mel - mel.mean()) / (mel.std() + 1e-5)
        return mel, track_id


def to_max_frames(mel):
    if mel.shape[-1] < MAX_FRAMES:
        return F.pad(mel, (0, MAX_FRAMES - mel.shape[-1]))
    return mel[..., :MAX_FRAMES]


def collate_mels(model_name):
    def collate(batch):
        mels, track_ids = zip(*batch)
        if model_name in {"CNN_LSTM", "ViT"}:
            padded = torch.stack([to_max_frames(mel) for mel in mels])
        else:
            max_frames = max(mel.shape[-1] for mel in mels)
            padded = torch.stack(
                [F.pad(mel, (0, max_frames - mel.shape[-1])) for mel in mels]
            )
        if model_name == "LSTM":
            inputs = padded.transpose(1, 2)
        else:
            inputs = padded.unsqueeze(1)
        return inputs, list(track_ids)

    return collate


def extract_embedding(model_name, model, x):
    if model_name == "CNN":
        x = model.pool(F.relu(model.bn1(model.conv1(x))))
        x = model.pool(F.relu(model.bn2(model.conv2(x))))
        x = model.pool(F.relu(model.bn3(model.conv3(x))))
        x = model.pool(F.relu(model.bn4(model.conv4(x))))
        x = model.global_pool(x).flatten(1)
        x = F.relu(model.fc1(x))
        return F.relu(model.fc2(x))
    if model_name == "CNN_LSTM":
        x = model.pool(F.relu(model.bn1(model.conv1(x))))
        x = model.pool(F.relu(model.bn2(model.conv2(x))))
        x = model.pool(F.relu(model.bn3(model.conv3(x))))
        x = model.pool(F.relu(model.bn4(model.conv4(x))))
        x = x.mean(dim=2).transpose(1, 2)
        _, (hidden, _) = model.lstm(x)
        return F.relu(model.fc1(hidden[-1]))
    if model_name == "LSTM":
        _, (hidden, _) = model.lstm(x)
        return hidden[-1]
    if model_name == "ResNet":
        return model.backbone(x)
    if model_name == "ViT":
        x = model.patch_embedding(x).flatten(2).transpose(1, 2)
        class_token = model.class_token.expand(x.shape[0], -1, -1)
        return model.encoder(torch.cat((class_token, x), dim=1))[:, 0]
    raise ValueError(f"unsupported model: {model_name}")


def load_model(model_name, device):
    config = MODEL_CONFIGS[model_name]
    checkpoint_path = CHECKPOINT_DIR / config["checkpoint"]
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    model = config["factory"]()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def extract_model(model_name, mel_dir, labeled_ids, device):
    config = MODEL_CONFIGS[model_name]
    dataset = MelTrackDataset(mel_dir, labeled_ids, config["normalize"])
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        collate_fn=collate_mels(model_name),
    )
    model = load_model(model_name, device)
    track_ids = []
    batches = []
    print(
        f"extracting {config['dimensions']} {model_name} features for "
        f"{len(dataset)} tracks on {device}",
        flush=True,
    )
    with torch.inference_mode():
        for batch_number, (inputs, batch_ids) in enumerate(loader, start=1):
            embeddings = extract_embedding(
                model_name, model, inputs.to(device, non_blocking=True)
            )
            batches.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
            track_ids.extend(batch_ids)
            if batch_number % 100 == 0:
                print(
                    f"{model_name}: processed "
                    f"{min(batch_number * config['batch_size'], len(dataset))}"
                    f"/{len(dataset)} tracks",
                    flush=True,
                )

    embedding_array = np.concatenate(batches)
    expected_shape = (len(dataset), config["dimensions"])
    if embedding_array.shape != expected_shape:
        raise ValueError(
            f"{model_name} produced {embedding_array.shape}, expected {expected_shape}"
        )
    output_path = CHECKPOINT_DIR / f"{model_name}_fixed_embeddings.npz"
    np.savez_compressed(
        output_path,
        track_ids=np.asarray(track_ids),
        embeddings=embedding_array,
    )
    print(
        f"saved {embedding_array.shape} to {output_path} "
        f"({output_path.stat().st_size / (1024**2):.1f} MiB)",
        flush=True,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=f"model names or all (choices: {', '.join(MODEL_CONFIGS)})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.models) == 1 and args.models[0].lower() == "all":
        selected = list(MODEL_CONFIGS)
    else:
        by_lower = {name.lower(): name for name in MODEL_CONFIGS}
        unknown = [name for name in args.models if name.lower() not in by_lower]
        if unknown:
            raise ValueError(f"unknown models: {', '.join(unknown)}")
        selected = [by_lower[name.lower()] for name in args.models]

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labeled_tracks = pd.read_csv(
        TRACKS_CSV, usecols=["id", "popularity"]
    ).dropna(subset=["id", "popularity"])
    labeled_ids = set(labeled_tracks["id"].astype(str))
    mel_dir = sync_to_local_scratch(
        str(NETWORK_MEL_DIR), str(LOCAL_SCRATCH_DIR)
    )
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for model_name in selected:
        extract_model(model_name, mel_dir, labeled_ids, device)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
