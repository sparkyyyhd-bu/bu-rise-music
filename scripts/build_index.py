#!/usr/bin/env python3
"""Merge embedding shards -> index/embeddings.npy + index/tracks.parquet.

Environment: SCC login node or any machine (CPU-only, no model needed). Run
after all embed_audio.py shards have finished:

    python scripts/build_index.py
    python scripts/build_index.py --strict   # fail if any shard is missing

Outputs (row i of tracks.parquet describes row i of embeddings.npy):
- embeddings.npy   float32 (N, 512), L2-normalized track vectors
- tracks.parquet   track_id, artist, title, duration, stream_url,
                   genres, moods, instruments, path
- vocab.json       tag vocabulary actually present in the index (used to
                   constrain LLM query expansion)
- skipped.json     tracks that failed to embed, with reasons
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playlistgen.config import load_config  # noqa: E402
from playlistgen.jamendo import attach_names, parse_autotagging_tsv, stream_url  # noqa: E402

log = logging.getLogger("build_index")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--strict", action="store_true", help="error out if the shard sequence has gaps")
    args = ap.parse_args()
    cfg = load_config(args.config)

    shards_dir = Path(cfg["paths"]["shards_dir"])
    manifests = sorted(shards_dir.glob("shard_*.json"))
    if not manifests:
        sys.exit(f"no shard manifests found in {shards_dir}; run scripts/embed_audio.py first")

    shard_ids = [int(re.search(r"shard_(\d+)\.json", m.name).group(1)) for m in manifests]
    expected = set(range(max(shard_ids) + 1))
    missing = sorted(expected - set(shard_ids))
    if missing:
        msg = f"missing shards: {missing[:20]}{'...' if len(missing) > 20 else ''}"
        if args.strict:
            sys.exit(msg)
        log.warning("%s (continuing without them)", msg)

    tracks = parse_autotagging_tsv(cfg["paths"]["metadata_tsv"])
    raw_meta = Path(cfg["paths"]["metadata_tsv"]).parent / "raw.meta.tsv"
    if raw_meta.exists():
        attach_names(tracks, raw_meta)
    by_id = {t.track_id: t for t in tracks}

    blocks: list[np.ndarray] = []
    rows: list[dict] = []
    skipped: list[dict] = []
    checkpoint = pooling = None
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        emb = np.load(manifest_path.with_suffix(".npy"))
        if emb.shape[0] != len(manifest["track_ids"]):
            sys.exit(f"{manifest_path}: {emb.shape[0]} vectors vs {len(manifest['track_ids'])} ids")
        # All shards must have been produced with the same settings.
        for key, seen in (("checkpoint", checkpoint), ("pooling", pooling)):
            if seen is not None and manifest[key] != seen:
                sys.exit(f"{manifest_path}: mixed {key} across shards ({manifest[key]} vs {seen})")
        checkpoint, pooling = manifest["checkpoint"], manifest["pooling"]
        skipped.extend(manifest.get("skipped", []))
        blocks.append(emb.astype(np.float32))
        for tid in manifest["track_ids"]:
            t = by_id[tid]
            rows.append(
                {
                    "track_id": tid,
                    "artist": t.artist_name or f"artist_{t.artist_id}",
                    "title": t.track_name or f"track_{tid}",
                    "duration": t.duration,
                    "stream_url": stream_url(tid),
                    "path": t.path,
                    "genres": t.genres,
                    "moods": t.moods,
                    "instruments": t.instruments,
                }
            )

    embeddings = np.concatenate(blocks, axis=0)
    df = pd.DataFrame(rows)
    assert len(df) == embeddings.shape[0], "row/embedding misalignment"

    # Safety: re-normalize (mean pooling already normalized, but be defensive).
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / np.maximum(norms, 1e-12)).astype(np.float32)

    index_dir = Path(cfg["paths"]["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", embeddings)
    df.to_parquet(index_dir / "tracks.parquet", index=False)

    vocab = {
        "genres": sorted({g for r in rows for g in r["genres"]}),
        "moods": sorted({m for r in rows for m in r["moods"]}),
        "instruments": sorted({i for r in rows for i in r["instruments"]}),
        "checkpoint": checkpoint,
        "pooling": pooling,
    }
    (index_dir / "vocab.json").write_text(json.dumps(vocab, indent=2))
    (index_dir / "skipped.json").write_text(json.dumps(skipped, indent=2))

    log.info(
        "index built: %d tracks (%d skipped) -> %s [checkpoint=%s pooling=%s]",
        len(df), len(skipped), index_dir, checkpoint, pooling,
    )


if __name__ == "__main__":
    main()
