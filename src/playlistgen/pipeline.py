"""The full expand -> retrieve -> rerank -> sequence pipeline.

Environment: Mac (query side). This module is the single implementation used
by BOTH the CLI (cli.py) and the web server (serve.py) -- no duplicated logic.
"""

from __future__ import annotations

import logging
from typing import Any

from .clap_model import ClapEncoder, l2_normalize
from .expand import ExpandedQuery, ExpansionError, MissingAPIKeyError, expand_query
from .rerank import diversify_artists, rerank
from .retrieve import PlaylistIndex, cosine_top_n
from .sequence import sequence_playlist

log = logging.getLogger(__name__)


class InadequateIndexError(RuntimeError):
    """The loaded index is a smoke sample, not a usable music catalog."""


def validate_index(index: PlaylistIndex, requested_length: int) -> None:
    """Fail clearly when a tiny smoke index would yield misleading results."""
    artists = index.tracks["artist"].fillna("").astype(str).str.strip().nunique()
    tags = {
        tag
        for field in ("genres", "moods", "instruments")
        for values in index.tracks[field]
        for tag in values
    }
    # Five artists and four searchable tags is deliberately a low bar. It
    # catches a tiny smoke artifact while supporting category-specific indexes.
    if requested_length >= 10 and (artists < 5 or len(tags) < 4):
        raise InadequateIndexError(
            f"The loaded index is only a smoke sample ({len(index)} tracks, "
            f"{artists} artists, {len(tags)} searchable tags), so it cannot match varied "
            "playlist prompts. Download a representative Jamendo audio set, "
            "embed it without --limit, and rebuild the index."
        )


def generate_playlist(
    prompt: str,
    n: int,
    cfg: dict[str, Any],
    index: PlaylistIndex,
    encoder: ClapEncoder,
    use_llm: bool = True,
    expansion_client: Any = None,
) -> dict[str, Any]:
    """Run the whole pipeline and return a JSON-serializable result dict.

    Raises MissingAPIKeyError if use_llm=True and no API key is configured
    (the caller decides how to surface that); any other expansion failure
    falls back to raw-prompt mode with a note.
    """
    validate_index(index, n)
    notes: list[str] = []
    if use_llm:
        try:
            query = expand_query(prompt, cfg, vocab=index.vocab, client=expansion_client)
        except MissingAPIKeyError:
            raise
        except ExpansionError as exc:
            log.warning("expansion failed, falling back to raw prompt: %s", exc)
            query = ExpandedQuery.from_raw_prompt(prompt)
            notes.append(f"LLM expansion failed ({exc}); used the raw prompt instead.")
    else:
        query = ExpandedQuery.from_raw_prompt(prompt)

    # Embed each caption, average, re-normalize -> one query vector.
    caption_embs = encoder.embed_texts(query.captions)
    query_vec = l2_normalize(caption_embs.mean(axis=0))

    top_n = max(int(cfg["retrieval"]["top_n"]), n)
    rows, cosines = cosine_top_n(query_vec, index, top_n)

    constraints = query if query.source == "llm" else None
    candidates = rerank(
        rows,
        cosines,
        index,
        constraints,
        alpha=float(cfg["retrieval"]["alpha"]),
        beta=float(cfg["retrieval"]["beta"]),
        keep_at_least=n,
    )
    max_per_artist = int(cfg["retrieval"].get("max_tracks_per_artist", 2))
    candidates = diversify_artists(candidates, index, n, max_per_artist)
    if query.bpm_range is not None:
        notes.append(
            "bpm_range is advisory only: Jamendo metadata has no tempo, so no "
            "BPM filter was applied (precompute tempo offline to enable it)."
        )
    notes.append(
        "Track order is a greedy nearest-neighbor path through CLAP space "
        "(seeded from the best match), trading a little relevance for flow."
    )

    ordered = sequence_playlist(candidates, index, n)

    playlist = []
    for rank, c in enumerate(ordered, start=1):
        t = index.tracks.iloc[c.row]
        playlist.append(
            {
                "rank": rank,
                "track_id": int(t["track_id"]),
                "artist": str(t["artist"]),
                "title": str(t["title"]),
                "duration": float(t["duration"]),
                "similarity": round(c.cosine, 4),
                "combined_score": round(c.combined, 4),
                "matched_tags": c.matched_tags,
                "tags": list(t["genres"]) + list(t["moods"]) + list(t["instruments"]),
                "stream_url": str(t["stream_url"]),
            }
        )

    return {
        "prompt": prompt,
        "playlist": playlist,
        "expansion": query.model_dump(),
        "notes": notes,
    }
