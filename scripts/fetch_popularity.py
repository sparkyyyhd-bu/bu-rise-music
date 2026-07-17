#!/usr/bin/env python3
"""Fetch live Jamendo artist popularity -> index/artist_popularity.json.

Environment: Mac or SCC, needs network + JAMENDO_CLIENT_ID (get one at
devportal.jamendo.com, then put it in .env). Run once before
scripts/build_index.py (or re-run any time to refresh rankings):

    python scripts/fetch_popularity.py

This reads artist ids straight out of the MTG-Jamendo metadata TSV (same
source build_index.py uses), queries the real Jamendo API for a popularity
ranking (see playlistgen.jamendo_api), and writes a small
{artist_id: score} JSON file. build_index.py picks it up automatically if
present and adds a "mainstream_score" column to tracks.parquet; if this
script is never run, that column is simply absent and the "prefer mainstream
artists" preference is a no-op.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playlistgen.config import load_config  # noqa: E402
from playlistgen.jamendo import parse_autotagging_tsv  # noqa: E402
from playlistgen.jamendo_api import fetch_artist_popularity  # noqa: E402

log = logging.getLogger("fetch_popularity")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    client_id = os.environ.get("JAMENDO_CLIENT_ID")
    if not client_id:
        sys.exit(
            "JAMENDO_CLIENT_ID is not set. Get a client id at devportal.jamendo.com "
            "and add JAMENDO_CLIENT_ID=... to .env"
        )

    tracks = parse_autotagging_tsv(cfg["paths"]["metadata_tsv"])
    artist_ids = {t.artist_id for t in tracks}
    log.info("fetching popularity ranking for %d artists...", len(artist_ids))

    popularity = fetch_artist_popularity(artist_ids, client_id)

    index_dir = Path(cfg["paths"]["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)
    out_path = index_dir / "artist_popularity.json"
    out_path.write_text(json.dumps({str(k): v for k, v in popularity.items()}, indent=2))
    log.info("wrote %d artist scores -> %s", len(popularity), out_path)


if __name__ == "__main__":
    main()
