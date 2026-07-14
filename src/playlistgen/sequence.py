"""Playlist sequencing: greedy nearest-neighbor path through embedding space.

Environment: Mac (query side) and SCC.

Seeded from the highest-scoring candidate, then repeatedly appends the
remaining candidate closest (cosine) to the current track. This trades a
little per-track relevance for smooth transitions -- adjacent tracks sound
alike, so the playlist "flows" instead of jumping between the query's modes.
Mentioned in --explain output.
"""

from __future__ import annotations

import numpy as np

from .rerank import Candidate
from .retrieve import PlaylistIndex


def sequence_playlist(
    candidates: list[Candidate], index: PlaylistIndex, length: int
) -> list[Candidate]:
    """Order the top candidates into a smooth listening sequence.

    `candidates` must be sorted by combined score (rerank output). The best
    `length` candidates are kept; ordering among them is the greedy NN path.
    """
    pool = candidates[: max(length, 0)]
    if len(pool) <= 2:
        return pool

    embs = np.stack([index.embeddings[c.row] for c in pool])  # (k, 512), normalized
    remaining = list(range(len(pool)))
    path = [remaining.pop(0)]  # seed: highest combined score
    while remaining:
        current = embs[path[-1]]
        sims = embs[remaining] @ current
        nxt = int(np.argmax(sims))
        path.append(remaining.pop(nxt))
    return [pool[i] for i in path]
