"""Cosine top-N retrieval over the precomputed index.

Environment: Mac (query side) and SCC. Plain numpy matmul -- at ~55k tracks a
(55k, 512) float32 matrix is ~110 MB and a single matmul takes milliseconds,
so no ANN library (FAISS etc.) is needed or wanted here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PlaylistIndex:
    """embeddings.npy + tracks.parquet (+ vocab.json), row-aligned."""

    embeddings: np.ndarray      # float32 (N, 512), L2-normalized
    tracks: pd.DataFrame        # row i describes embeddings[i]
    vocab: dict | None = None

    @classmethod
    def load(cls, index_dir: str | Path) -> "PlaylistIndex":
        index_dir = Path(index_dir)
        emb_path = index_dir / "embeddings.npy"
        parquet_path = index_dir / "tracks.parquet"
        if not emb_path.exists() or not parquet_path.exists():
            raise FileNotFoundError(
                f"no index at {index_dir} (need embeddings.npy + tracks.parquet; "
                "run scripts/embed_audio.py then scripts/build_index.py)"
            )
        embeddings = np.load(emb_path).astype(np.float32)
        tracks = pd.read_parquet(parquet_path)
        if len(tracks) != embeddings.shape[0]:
            raise ValueError(
                f"index misaligned: {embeddings.shape[0]} embeddings vs {len(tracks)} track rows"
            )
        vocab_path = index_dir / "vocab.json"
        vocab = json.loads(vocab_path.read_text()) if vocab_path.exists() else None
        return cls(embeddings=embeddings, tracks=tracks, vocab=vocab)

    def __len__(self) -> int:
        return self.embeddings.shape[0]


def cosine_top_n(
    query: np.ndarray, index: PlaylistIndex, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (row_indices, similarities) of the top-n most similar tracks.

    `query` must be a single L2-normalized (512,) vector; with normalized
    embeddings the dot product IS the cosine similarity.
    """
    sims = index.embeddings @ query.astype(np.float32)
    n = min(n, len(sims))
    top = np.argpartition(-sims, n - 1)[:n]
    top = top[np.argsort(-sims[top])]
    return top, sims[top]
