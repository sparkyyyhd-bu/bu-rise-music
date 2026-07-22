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
    def __init__(self, mel_dir, popularity_by_id):
        self.entries = []
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
