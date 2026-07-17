"""Live Jamendo API client for artist popularity.

Environment: Mac or SCC, offline-enrichment step only (not on the query
path). Needs network + a Jamendo client_id (devportal.jamendo.com).

Unlike jamendo.py (which parses the static MTG-Jamendo research dataset),
this module talks to the real api.jamendo.com/v3.0 service to back the
"prefer mainstream artists" ranking signal.

Jamendo does not return a numeric popularity value on a track/artist object;
it only supports sorting by popularity (order=popularity_total_desc). So the
score computed here is each artist's rank within the batch of ids requested,
normalized to [0, 1] (1.0 = most popular of the group). Artist ids are
queried in batches of BATCH_SIZE (Jamendo's documented max page size), and
ranks are concatenated batch-by-batch, so the result is only a full global
ranking when all ids fit in one batch; with more artists than that it is a
coarse per-batch approximation, which is the best signal the documented API
exposes without per-track stats fields.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import requests

API_BASE = "https://api.jamendo.com/v3.0"
BATCH_SIZE = 200  # Jamendo's documented max "limit"

log = logging.getLogger(__name__)


class JamendoAPIError(RuntimeError):
    pass


def fetch_artist_popularity(
    artist_ids: Iterable[int],
    client_id: str,
    *,
    request_delay: float = 0.2,
    session: requests.Session | None = None,
) -> dict[int, float]:
    """Return {artist_id: mainstream_score}, score in [0, 1] (1 = most popular)."""
    ids = sorted({int(a) for a in artist_ids})
    if not ids:
        return {}

    http = session or requests.Session()
    ranked: list[int] = []
    for start in range(0, len(ids), BATCH_SIZE):
        batch = ids[start : start + BATCH_SIZE]
        params = [
            ("client_id", client_id),
            ("format", "json"),
            ("limit", str(len(batch))),
            ("order", "popularity_total_desc"),
        ] + [("id", str(a)) for a in batch]

        resp = http.get(f"{API_BASE}/artists/", params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("headers", {}).get("status") != "success":
            raise JamendoAPIError(f"Jamendo API error: {payload.get('headers')}")

        batch_ranked = [int(a["id"]) for a in payload.get("results", [])]
        found = set(batch_ranked)
        missing = [a for a in batch if a not in found]
        if missing:
            log.warning("jamendo API returned no data for %d artist id(s); ranked last", len(missing))
        batch_ranked.extend(missing)
        ranked.extend(batch_ranked)

        if start + BATCH_SIZE < len(ids):
            time.sleep(request_delay)

    n = len(ranked)
    if n == 1:
        return {ranked[0]: 1.0}
    return {artist_id: 1.0 - (rank / (n - 1)) for rank, artist_id in enumerate(ranked)}
