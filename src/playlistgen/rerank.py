"""Constraint-aware re-ranking of retrieved candidates.

Environment: Mac (query side) and SCC.

combined = alpha * cosine + beta * tag_jaccard

where tag_jaccard is the Jaccard overlap between the expanded query's
constraint tags (genres + moods + instruments) and the track's Jamendo tags.
With no constraints (e.g. --no-llm) the tag term is 0 for every track and the
ranking reduces to pure cosine order.

Vocal/instrumental: Jamendo has no explicit vocal/instrumental flag; the
`voice` instrument tag is the best available proxy. `instrumental` drops
tracks tagged with voice; `vocal` keeps only voice-tagged tracks when enough
candidates survive (tags are sparse, so this is applied softly to avoid
emptying the pool).

BPM: Jamendo's autotagging metadata carries no tempo, so bpm_range is NOT
enforced here. Do not run librosa tempo estimation at query time -- tempo
should be precomputed offline as an index enrichment step if needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expand import ExpandedQuery
from .retrieve import PlaylistIndex


@dataclass
class Candidate:
    row: int                    # row index into the PlaylistIndex
    cosine: float
    tag_overlap: float
    combined: float
    matched_tags: list[str]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rerank(
    rows: np.ndarray,
    cosines: np.ndarray,
    index: PlaylistIndex,
    query: ExpandedQuery | None,
    alpha: float,
    beta: float,
    keep_at_least: int = 1,
) -> list[Candidate]:
    """Score and sort retrieved candidates against the structured constraints."""
    constraint_tags: set[str] = set()
    vocals = "either"
    if query is not None:
        constraint_tags = set(query.genres) | set(query.moods) | set(query.instruments)
        vocals = query.vocals

    candidates: list[Candidate] = []
    for row, cos in zip(rows.tolist(), cosines.tolist()):
        t = index.tracks.iloc[row]
        track_tags = set(t["genres"]) | set(t["moods"]) | set(t["instruments"])
        overlap = _jaccard(constraint_tags, track_tags)
        candidates.append(
            Candidate(
                row=row,
                cosine=float(cos),
                tag_overlap=overlap,
                combined=alpha * float(cos) + beta * overlap,
                matched_tags=sorted(constraint_tags & track_tags),
            )
        )

    # Vocal/instrumental filter on the `voice` instrument tag (see docstring).
    if vocals != "either":
        def has_voice(c: Candidate) -> bool:
            return "voice" in set(index.tracks.iloc[c.row]["instruments"])

        wanted = [c for c in candidates if has_voice(c) == (vocals == "vocal")]
        if len(wanted) >= keep_at_least:
            candidates = wanted

    candidates.sort(key=lambda c: c.combined, reverse=True)
    return candidates
