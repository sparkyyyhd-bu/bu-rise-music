"""LAION-CLAP wrapper: checkpoint loading, text/audio embedding helpers.

Environment: Mac (query side, text encoder only in practice) and SCC (audio
embedding). This is the ONLY module that may import laion_clap or torch model
code -- keep the embedding model swappable behind this interface for later
thesis experiments.

Conventions:
- All returned embeddings are float32, L2-normalized, shape (N, D) or (D,).
- Audio is resampled to 48 kHz mono on load (CLAP's expected input).
- Long tracks are split into fixed windows, each window embedded, then pooled
  (mean or max, configurable) into a single track vector.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import numpy as np

log = logging.getLogger(__name__)

EMBED_DIM = 512


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return (x / norm).astype(np.float32)


def ensure_checkpoint(cfg: dict[str, Any]) -> Path:
    """Download the CLAP checkpoint if missing and return its path.

    Run this on a machine WITH internet (Mac, or the SCC login/transfer node).
    Compute nodes must find the file already present.
    """
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg["clap"]["checkpoint"]
    if ckpt_path.exists():
        return ckpt_path
    url = cfg["clap"]["checkpoint_url"]
    log.info("downloading CLAP checkpoint %s -> %s", url, ckpt_path)
    tmp = ckpt_path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(ckpt_path)
    return ckpt_path


def _pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    # MPS is intentionally not auto-selected: some HTSAT ops are flaky on MPS.
    # Set clap.device: mps in the config to opt in.
    return "cpu"


class ClapEncoder:
    """Lazy-loading CLAP encoder for text and audio.

    Construct with the full config dict; the model is loaded on first use so
    that CLI paths which never embed (e.g. --help) stay fast.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.sample_rate = int(cfg["clap"]["sample_rate"])
        self.window_seconds = float(cfg["chunking"]["window_seconds"])
        self.hop_seconds = float(cfg["chunking"]["hop_seconds"])
        self.pooling = cfg["chunking"]["pooling"]
        self.batch_size = int(cfg["embedding"]["batch_size"])
        self._model = None

    # -- model loading -----------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        import laion_clap
        import torch

        device = _pick_device(self.cfg["clap"].get("device", "auto"))
        ckpt_path = ensure_checkpoint(self.cfg)
        log.info("loading CLAP %s on %s", ckpt_path.name, device)
        model = laion_clap.CLAP_Module(
            enable_fusion=bool(self.cfg["clap"]["enable_fusion"]),
            amodel=self.cfg["clap"]["amodel"],
            device=device,
        )
        with torch.no_grad():
            model.load_ckpt(str(ckpt_path), verbose=False)
        model.eval()
        return model

    # -- text --------------------------------------------------------------

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a list of strings -> (N, 512) L2-normalized float32."""
        import torch

        with torch.no_grad():
            emb = self.model.get_text_embedding(list(texts), use_tensor=False)
        return l2_normalize(np.asarray(emb, dtype=np.float32))

    # -- audio -------------------------------------------------------------

    def load_audio(self, path: str | Path) -> np.ndarray:
        """Load any audio file as 48 kHz mono float32 in [-1, 1]."""
        import librosa

        wav, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return wav.astype(np.float32)

    def _windows(self, wav: np.ndarray) -> np.ndarray:
        """Split a waveform into fixed-length windows (padded at the tail)."""
        win = int(self.window_seconds * self.sample_rate)
        hop = int(self.hop_seconds * self.sample_rate)
        if len(wav) <= win:
            padded = np.zeros(win, dtype=np.float32)
            padded[: len(wav)] = wav
            return padded[None, :]
        starts = list(range(0, len(wav) - win + 1, hop))
        # Cover the tail if the last hop doesn't reach the end of the track.
        if starts[-1] + win < len(wav):
            starts.append(len(wav) - win)
        return np.stack([wav[s : s + win] for s in starts]).astype(np.float32)

    @staticmethod
    def _quantize(x: np.ndarray) -> np.ndarray:
        # int16 round-trip, matching how CLAP's training data was quantized.
        x = np.clip(x, -1.0, 1.0)
        return (x * 32767.0).astype(np.int16).astype(np.float32) / 32767.0

    def embed_audio_windows(self, wav: np.ndarray) -> np.ndarray:
        """Embed every window of a waveform -> (num_windows, 512), normalized."""
        import torch

        windows = self._quantize(self._windows(wav))
        chunks = []
        with torch.no_grad():
            for i in range(0, len(windows), self.batch_size):
                batch = windows[i : i + self.batch_size]
                emb = self.model.get_audio_embedding_from_data(
                    x=batch, use_tensor=False
                )
                chunks.append(np.asarray(emb, dtype=np.float32))
        return l2_normalize(np.concatenate(chunks, axis=0))

    def pool_windows(self, window_embs: np.ndarray) -> np.ndarray:
        """Pool window embeddings into one L2-normalized track vector."""
        if self.pooling == "mean":
            pooled = window_embs.mean(axis=0)
        elif self.pooling == "max":
            pooled = window_embs.max(axis=0)
        else:
            raise ValueError(f"unknown pooling {self.pooling!r} (use mean|max)")
        return l2_normalize(pooled)

    def embed_audio_file(self, path: str | Path) -> np.ndarray:
        """Load, window, embed, and pool one audio file -> (512,) vector."""
        wav = self.load_audio(path)
        return self.pool_windows(self.embed_audio_windows(wav))
