"""Synthetic index + fake encoder shared by the offline unit tests.

The fake index uses (near-)basis-vector embeddings so retrieval order is
fully deterministic; the fake encoder maps known caption strings to the same
basis vectors. No CLAP, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from playlistgen.retrieve import PlaylistIndex

DIM = 512


def basis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def make_index(n: int = 10) -> PlaylistIndex:
    embeddings = np.stack([basis(i) for i in range(n)])
    rows = []
    for i in range(n):
        rows.append(
            {
                "track_id": 1000 + i,
                "artist": f"artist{i}",
                "title": f"title{i}",
                "duration": 200.0 + i,
                "stream_url": f"https://example.com/{1000 + i}.mp3",
                "path": f"{i:02d}/{1000 + i}.mp3",
                "genres": ["reggae"] if i % 2 == 0 else ["techno"],
                "moods": ["calm"] if i < 5 else ["energetic"],
                "instruments": ["voice"] if i in (1, 3) else ["guitar"],
            }
        )
    vocab = {
        "genres": ["reggae", "techno"],
        "moods": ["calm", "energetic"],
        "instruments": ["voice", "guitar"],
    }
    return PlaylistIndex(embeddings=embeddings, tracks=pd.DataFrame(rows), vocab=vocab)


class FakeEncoder:
    """Maps caption strings to fixed vectors; no model behind it."""

    def __init__(self, mapping: dict[str, np.ndarray] | None = None):
        self.mapping = mapping or {}

    def embed_texts(self, texts):
        out = []
        for t in texts:
            if t in self.mapping:
                out.append(self.mapping[t])
            else:  # deterministic pseudo-vector for unknown text
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                v = rng.standard_normal(DIM).astype(np.float32)
                out.append(v / np.linalg.norm(v))
        return np.stack(out)
