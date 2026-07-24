from torch.utils.data import Dataset
import torch
import pandas as pd
import os
from tqdm import tqdm 
import shutil
import torch.nn.functional as F


MEL_SUFFIX = "_mel.pt"
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tracks_csv = os.path.join(repo_root, "data", "SpotGenTrack", "Data Sources", "spotify_tracks.csv")


def spec_augment(
    mels,
    frequency_dim,
    time_dim,
    probability=0.8,
    max_frequency_width=10,
    max_time_width=80,
    frequency_masks=1,
    time_masks=1,
):
    """Apply independent frequency and time masks to each training example.

    Mel spectrograms are standardized to approximately zero mean, so zero is a
    neutral mask value. Call this only for training batches.
    """
    if mels.ndim < 3:
        raise ValueError("expected a batched mel spectrogram with at least 3 dimensions")

    frequency_dim %= mels.ndim
    time_dim %= mels.ndim
    if frequency_dim in (0, time_dim) or time_dim == 0:
        raise ValueError("batch, frequency, and time dimensions must be distinct")

    augmented = mels.clone()
    frequency_size = augmented.shape[frequency_dim]
    time_size = augmented.shape[time_dim]
    device = augmented.device

    for batch_index in range(augmented.shape[0]):
        if torch.rand((), device=device) >= probability:
            continue

        index = [slice(None)] * augmented.ndim
        index[0] = batch_index

        for _ in range(frequency_masks):
            frequency_width = int(
                torch.randint(
                    0,
                    min(max_frequency_width, frequency_size) + 1,
                    (),
                    device=device,
                ).item()
            )
            if not frequency_width:
                continue
            frequency_start = int(
                torch.randint(
                    0, frequency_size - frequency_width + 1, (), device=device
                ).item()
            )
            index[frequency_dim] = slice(
                frequency_start, frequency_start + frequency_width
            )
            augmented[tuple(index)] = 0.0
            index[frequency_dim] = slice(None)

        for _ in range(time_masks):
            time_width = int(
                torch.randint(
                    0, min(max_time_width, time_size) + 1, (), device=device
                ).item()
            )
            if not time_width:
                continue
            time_start = int(
                torch.randint(0, time_size - time_width + 1, (), device=device).item()
            )
            index[time_dim] = slice(time_start, time_start + time_width)
            augmented[tuple(index)] = 0.0

    return augmented


def augment_mels(mels):
    """Apply the shared strong spectrogram augmentation used for training."""
    mels = spec_augment(
        mels,
        frequency_dim=2,
        time_dim=3,
        probability=0.9,
        max_frequency_width=16,
        max_time_width=120,
        frequency_masks=2,
        time_masks=2,
    )
    max_shift = max(1, int(0.05 * mels.shape[-1]))
    shift = int(
        torch.randint(-max_shift, max_shift + 1, (), device=mels.device).item()
    )
    mels = torch.roll(mels, shifts=shift, dims=-1)
    return mels + 0.01 * torch.randn_like(mels)


def augment_lstm_mels(
    mels,
    probability=0.8,
    max_gain_change=0.05,
    max_offset_std=0.02,
    noise_std=0.01,
):
    """Mildly perturb (batch, time, mel) sequences without masking time spans.

    The same affine perturbation is used at every time step in an example so
    temporal structure stays intact. Noise is scaled to each example's
    standard deviation, and padded frames remain exactly zero.
    """
    if mels.ndim != 3:
        raise ValueError("expected LSTM input shaped (batch, time, mel)")

    augmented = mels.clone()
    valid_frames = augmented.ne(0).any(dim=2, keepdim=True)
    apply = (
        torch.rand(
            augmented.shape[0], 1, 1, device=augmented.device
        ) < probability
    )
    apply &= valid_frames.any(dim=1, keepdim=True)
    valid_values = valid_frames.expand_as(augmented)
    value_counts = valid_values.sum(dim=(1, 2), keepdim=True).clamp_min(1)
    means = (augmented * valid_values).sum(dim=(1, 2), keepdim=True) / value_counts
    variances = (
        ((augmented - means) ** 2 * valid_values).sum(
            dim=(1, 2), keepdim=True
        )
        / value_counts
    )
    scales = variances.sqrt().clamp_min(1e-6)

    gains = 1 + (
        2
        * torch.rand(
            augmented.shape[0], 1, 1, device=augmented.device
        )
        - 1
    ) * max_gain_change
    offsets = (
        2
        * torch.rand(
            augmented.shape[0], 1, 1, device=augmented.device
        )
        - 1
    ) * max_offset_std * scales
    noise = noise_std * scales * torch.randn_like(augmented)
    perturbed = augmented * gains + offsets + noise
    return torch.where(apply & valid_frames, perturbed, augmented)


def checkpoint_matches_model(model, checkpoint):
    """Check state-dict keys and tensor shapes without modifying the model."""
    if not isinstance(checkpoint, dict):
        return False, "checkpoint is not a dictionary"

    checkpoint_state = checkpoint.get("model_state_dict")
    if not isinstance(checkpoint_state, dict):
        return False, "checkpoint has no model_state_dict"

    model_state = model.state_dict()
    missing = sorted(model_state.keys() - checkpoint_state.keys())
    unexpected = sorted(checkpoint_state.keys() - model_state.keys())
    shape_mismatches = [
        f"{name}: checkpoint {tuple(checkpoint_state[name].shape)}, "
        f"model {tuple(model_state[name].shape)}"
        for name in sorted(model_state.keys() & checkpoint_state.keys())
        if checkpoint_state[name].shape != model_state[name].shape
    ]

    problems = []
    if missing:
        problems.append(f"missing keys: {', '.join(missing)}")
    if unexpected:
        problems.append(f"unexpected keys: {', '.join(unexpected)}")
    if shape_mismatches:
        problems.append(f"shape mismatches: {'; '.join(shape_mismatches)}")

    if problems:
        return False, " | ".join(problems)
    return True, "all parameter keys and shapes match"

class MelPopularityDataset(Dataset):
    def __init__(self, mel_dir, popularity_by_id, normalize=True):
        self.entries = []
        self.normalize = normalize
        for entry in os.scandir(mel_dir):
            if not entry.is_file() or not entry.name.endswith(MEL_SUFFIX):
                continue
            track_id = entry.name[: -len(MEL_SUFFIX)]
            popularity = popularity_by_id.get(track_id)
            if popularity is None:
                continue
            self.entries.append((entry.path, popularity))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, popularity = self.entries[index]
        mel = torch.load(path)
        if mel.dim() == 3:
            mel = mel.mean(dim=0)  # collapse stereo channels to mono
        if self.normalize:
            mel = (mel - mel.mean()) / (mel.std() + 1e-5)
        target = torch.tensor(popularity / 100.0, dtype=torch.float32)
        return mel, target
    
def load_popularity_by_id():
    df = pd.read_csv(tracks_csv, usecols=["id", "popularity"])
    df = df.dropna(subset=["id", "popularity"])
    return dict(zip(df["id"], df["popularity"]))

def sync_to_local_scratch(source_dir, local_root):
    """Copy source_dir onto the compute node's local scratch disk once, then
    reuse that copy on subsequent runs so training reads local disk instead
    of the network-mounted scratch space."""
    local_dir = os.path.join(local_root, os.path.basename(source_dir.rstrip("/")))
    done_marker = local_dir + ".sync_complete"
    if not os.path.exists(done_marker):
        os.makedirs(local_dir, exist_ok=True)
        filenames = [entry.name for entry in os.scandir(source_dir) if entry.is_file()]
        for name in tqdm(filenames, desc="syncing mel spectrograms to local scratch"):
            shutil.copy2(os.path.join(source_dir, name), os.path.join(local_dir, name))
        open(done_marker, "w").close()
    return local_dir
