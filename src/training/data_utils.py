from torch.utils.data import Dataset, Subset, random_split
import torch
import pandas as pd
import ast
import hashlib
import os
from tqdm import tqdm 
import shutil
import torch.nn.functional as F


MEL_SUFFIX = "_mel.pt"
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tracks_csv = os.path.join(repo_root, "data", "SpotGenTrack", "Data Sources", "spotify_tracks.csv")
FIXED_SPLIT_SEED = 0
FIXED_SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_MODES = ("fixed", "legacy")


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


def augment_vit_mels(mels):
    """Apply mild masking without overwhelming the pretrained ViT features."""
    return spec_augment(
        mels,
        frequency_dim=2,
        time_dim=3,
        probability=0.8,
        max_frequency_width=10,
        max_time_width=80,
        frequency_masks=1,
        time_masks=1,
    )


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
        # Split membership must depend on track identity, never filesystem order.
        self.entries.sort(
            key=lambda item: os.path.basename(item[0])[: -len(MEL_SUFFIX)]
        )
        self.track_ids = [
            os.path.basename(path)[: -len(MEL_SUFFIX)]
            for path, _ in self.entries
        ]

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


def _parse_artist_ids(value):
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        parsed = [value]
    if isinstance(parsed, str):
        parsed = [parsed]
    return sorted({str(artist) for artist in parsed if str(artist)})


def grouped_train_val_test_split(
    dataset,
    metadata_csv=tracks_csv,
    fractions=FIXED_SPLIT_FRACTIONS,
    seed=FIXED_SPLIT_SEED,
):
    """Return deterministic 70/15/15 subsets with no artist/album leakage.

    Tracks sharing an artist or album are unioned into connected components.
    Components are indivisible, so the resulting fractions are the closest
    deterministic allocation practical rather than necessarily exact.
    """
    from torch.utils.data import Subset

    if len(fractions) != 3 or any(fraction <= 0 for fraction in fractions):
        raise ValueError("fractions must contain three positive values")
    total_fraction = sum(fractions)
    fractions = tuple(fraction / total_fraction for fraction in fractions)
    track_ids = list(dataset.track_ids)
    if len(track_ids) < 3:
        raise RuntimeError("need at least three tracks for a train/validation/test split")
    if len(track_ids) != len(set(track_ids)):
        raise RuntimeError("mel dataset contains duplicate track IDs")

    metadata = pd.read_csv(
        metadata_csv,
        usecols=["id", "album_id", "artists_id"],
        dtype=str,
    ).dropna(subset=["id"])
    wanted = set(track_ids)
    metadata = metadata.loc[metadata["id"].isin(wanted)]
    found = set(metadata["id"])
    missing = sorted(wanted - found)
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"{len(missing)} mel tracks have no artist/album metadata"
            f" (first: {preview})"
        )

    parent = {track_id: track_id for track_id in track_ids}
    component_size = {track_id: 1 for track_id in track_ids}

    def find(track_id):
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if component_size[left_root] < component_size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        component_size[left_root] += component_size[right_root]

    entity_owner = {}
    for row in metadata.itertuples(index=False):
        entities = []
        if pd.notna(row.album_id) and row.album_id:
            entities.append(("album", row.album_id))
        entities.extend(("artist", artist) for artist in _parse_artist_ids(row.artists_id))
        for entity in entities:
            owner = entity_owner.setdefault(entity, row.id)
            union(row.id, owner)

    components = {}
    for track_id in track_ids:
        components.setdefault(find(track_id), []).append(track_id)

    # Largest-first allocation minimizes deviation while keeping the result
    # deterministic. The hash provides a stable tie-break independent of CSV
    # and filesystem ordering.
    def component_key(ids):
        digest = hashlib.sha256(
            f"{seed}:{min(ids)}".encode("utf-8")
        ).hexdigest()
        return (-len(ids), digest)

    groups = sorted(components.values(), key=component_key)
    targets = [fraction * len(track_ids) for fraction in fractions]
    split_ids = [set(), set(), set()]
    counts = [0, 0, 0]
    for group in groups:
        best_split = min(
            range(3),
            key=lambda candidate: (
                sum(
                    (
                        counts[index]
                        + (len(group) if index == candidate else 0)
                        - targets[index]
                    )
                    ** 2
                    for index in range(3)
                ),
                candidate,
            ),
        )
        split_ids[best_split].update(group)
        counts[best_split] += len(group)

    if any(not ids for ids in split_ids):
        raise RuntimeError(
            "artist/album grouping produced an empty split; cannot satisfy 70/15/15"
        )
    entity_splits = {}
    for row in metadata.itertuples(index=False):
        split_index = next(
            index for index, ids in enumerate(split_ids) if row.id in ids
        )
        entities = []
        if pd.notna(row.album_id) and row.album_id:
            entities.append(("album", row.album_id))
        entities.extend(("artist", artist) for artist in _parse_artist_ids(row.artists_id))
        for entity in entities:
            previous = entity_splits.setdefault(entity, split_index)
            if previous != split_index:
                raise RuntimeError(
                    f"{entity[0]} {entity[1]} leaked across grouped splits"
                )
    indices = [
        [index for index, track_id in enumerate(track_ids) if track_id in ids]
        for ids in split_ids
    ]
    fingerprint = hashlib.sha256(
        "\n".join(
            f"{name}:{track_id}"
            for name, ids in zip(("train", "validation", "test"), split_ids)
            for track_id in sorted(ids)
        ).encode("utf-8")
    ).hexdigest()[:16]
    print(
        "fixed artist/album-isolated split "
        f"train={counts[0]} ({counts[0] / len(track_ids):.1%}), "
        f"validation={counts[1]} ({counts[1] / len(track_ids):.1%}), "
        f"test={counts[2]} ({counts[2] / len(track_ids):.1%}), "
        f"fingerprint={fingerprint}"
    )
    return tuple(Subset(dataset, split_indices) for split_indices in indices)


def grouped_track_id_split(track_ids, **kwargs):
    """Return the shared grouped split as track-ID lists."""

    class _TrackIds:
        def __init__(self, ids):
            self.track_ids = list(ids)

        def __len__(self):
            return len(self.track_ids)

    source = _TrackIds(track_ids)
    subsets = grouped_train_val_test_split(source, **kwargs)
    return tuple(
        [source.track_ids[index] for index in subset.indices]
        for subset in subsets
    )


def legacy_track_id_split(
    track_ids,
    fractions=FIXED_SPLIT_FRACTIONS,
    seed=FIXED_SPLIT_SEED,
):
    """Return legacy deterministic-random split membership as track-ID lists."""
    ids = list(track_ids)
    subsets = deterministic_random_train_val_test_split(
        _TrackIdDataset(ids), fractions=fractions, seed=seed
    )
    return tuple(
        [ids[index] for index in subset.indices] for subset in subsets
    )


def canonical_legacy_track_id_split(metadata_csv=tracks_csv):
    """Split the sorted labeled Spotify track IDs for all legacy models."""
    metadata = pd.read_csv(
        metadata_csv, usecols=["id", "popularity"], dtype={"id": str}
    ).dropna(subset=["id", "popularity"])
    track_ids = sorted(metadata["id"].drop_duplicates().tolist())
    return legacy_track_id_split(track_ids)


class _TrackIdDataset:
    def __init__(self, track_ids):
        self.track_ids = list(track_ids)

    def __len__(self):
        return len(self.track_ids)


def deterministic_random_train_val_test_split(
    dataset,
    fractions=FIXED_SPLIT_FRACTIONS,
    seed=FIXED_SPLIT_SEED,
):
    """Return the reproducible random split used by legacy-mode training."""
    if len(fractions) != 3 or any(fraction <= 0 for fraction in fractions):
        raise ValueError("fractions must contain three positive values")
    total_fraction = sum(fractions)
    fractions = tuple(fraction / total_fraction for fraction in fractions)
    validation_size = max(1, int(fractions[1] * len(dataset)))
    test_size = max(1, int(fractions[2] * len(dataset)))
    train_size = len(dataset) - validation_size - test_size
    if train_size < 1:
        raise RuntimeError("need at least three tracks for a train/validation/test split")
    subsets = random_split(
        dataset,
        [train_size, validation_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    print(
        "legacy deterministic random split "
        f"train={train_size}, validation={validation_size}, test={test_size}"
    )
    return subsets


def train_val_test_split(dataset, mode="fixed"):
    """Dispatch to the fixed grouped split or deterministic legacy split."""
    if mode == "fixed":
        return grouped_train_val_test_split(dataset)
    if mode == "legacy":
        if not hasattr(dataset, "track_ids"):
            raise ValueError("legacy split requires dataset.track_ids")
        split_ids = tuple(map(set, canonical_legacy_track_id_split()))
        indices_by_id = {
            track_id: index
            for index, track_id in enumerate(dataset.track_ids)
        }
        subsets = tuple(
            Subset(
                dataset,
                [
                    indices_by_id[track_id]
                    for track_id in sorted(ids & indices_by_id.keys())
                ],
            )
            for ids in split_ids
        )
        print(
            "legacy CSV-anchored split "
            + ", ".join(
                f"{name}={len(subset)}"
                for name, subset in zip(
                    ("train", "validation", "test"), subsets
                )
            )
        )
        return subsets
    raise ValueError(f"unknown split mode {mode!r}; choose from {', '.join(SPLIT_MODES)}")
    
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
