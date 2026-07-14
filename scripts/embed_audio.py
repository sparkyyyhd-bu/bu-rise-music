#!/usr/bin/env python3
"""Batch CLAP audio embedding -> per-shard .npy + manifest files.

Environment: SCC compute nodes (GPU, via embed_audio.qsub as an SGE array
job over shards) or any machine for local runs/tests. Compute nodes have no
internet: download audio (scripts/download_jamendo.sh) and the CLAP
checkpoint (`playlistgen --download-checkpoint` or any first model load) on
the login node first.

The track list from the metadata TSV is sorted by track id and split into
fixed-size shards (embedding.shard_size in the config). One invocation
embeds one shard:

    python scripts/embed_audio.py --shard 3            # shard index (0-based)
    python scripts/embed_audio.py                      # uses $SGE_TASK_ID - 1
    python scripts/embed_audio.py --all                # plain-bash fallback: all shards
    python scripts/embed_audio.py --shard 0 --limit 20 # local smoke test

Robustness:
- each shard writes shards_dir/shard_XXXXX.npy + shard_XXXXX.json atomically
  (tmp file + rename), so a killed job never leaves a half-written shard;
- shards whose outputs already exist are skipped (safe to resubmit the array);
- undecodable/corrupt/missing files are logged into the manifest's "skipped"
  list instead of crashing the job.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playlistgen.clap_model import ClapEncoder  # noqa: E402
from playlistgen.config import load_config  # noqa: E402
from playlistgen.jamendo import parse_autotagging_tsv, resolve_audio_path  # noqa: E402

log = logging.getLogger("embed_audio")


def shard_slices(n_tracks: int, shard_size: int) -> int:
    return (n_tracks + shard_size - 1) // shard_size


def embed_shard(cfg: dict, shard: int, limit: int | None = None) -> Path | None:
    shards_dir = Path(cfg["paths"]["shards_dir"])
    shards_dir.mkdir(parents=True, exist_ok=True)
    npy_path = shards_dir / f"shard_{shard:05d}.npy"
    manifest_path = shards_dir / f"shard_{shard:05d}.json"
    if npy_path.exists() and manifest_path.exists():
        log.info("shard %d already complete, skipping", shard)
        return npy_path

    tracks = parse_autotagging_tsv(cfg["paths"]["metadata_tsv"])
    shard_size = int(cfg["embedding"]["shard_size"])
    n_shards = shard_slices(len(tracks), shard_size)
    if shard >= n_shards:
        log.warning("shard %d out of range (%d shards total)", shard, n_shards)
        return None
    shard_tracks = tracks[shard * shard_size : (shard + 1) * shard_size]
    if limit:
        shard_tracks = shard_tracks[:limit]

    encoder = ClapEncoder(cfg)
    audio_dir = cfg["paths"]["audio_dir"]

    vectors: list[np.ndarray] = []
    track_ids: list[int] = []
    skipped: list[dict] = []
    for i, t in enumerate(shard_tracks):
        path = resolve_audio_path(audio_dir, t.path)
        if path is None:
            skipped.append({"track_id": t.track_id, "error": "audio file not found"})
            continue
        try:
            vectors.append(encoder.embed_audio_file(path))
            track_ids.append(t.track_id)
        except Exception as exc:  # corrupt/undecodable audio must not kill the job
            log.warning("skipping track %d (%s): %s", t.track_id, path, exc)
            skipped.append({"track_id": t.track_id, "error": str(exc)})
        if (i + 1) % 25 == 0:
            log.info("shard %d: %d/%d tracks", shard, i + 1, len(shard_tracks))

    if not vectors:
        log.error("shard %d produced no embeddings (missing audio?)", shard)

    emb = (
        np.stack(vectors).astype(np.float32)
        if vectors
        else np.zeros((0, 512), dtype=np.float32)
    )
    manifest = {
        "shard": shard,
        "track_ids": track_ids,
        "skipped": skipped,
        "checkpoint": cfg["clap"]["checkpoint"],
        "pooling": cfg["chunking"]["pooling"],
        "window_seconds": cfg["chunking"]["window_seconds"],
        "hop_seconds": cfg["chunking"]["hop_seconds"],
    }

    # Atomic writes: never leave a half-written shard on disk.
    tmp_npy = npy_path.with_suffix(".npy.tmp")
    with open(tmp_npy, "wb") as fh:  # file handle: np.save won't append ".npy"
        np.save(fh, emb)
    os.replace(tmp_npy, npy_path)
    tmp_json = manifest_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(manifest))
    os.replace(tmp_json, manifest_path)
    log.info(
        "shard %d done: %d embedded, %d skipped -> %s",
        shard, len(track_ids), len(skipped), npy_path,
    )
    return npy_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="config YAML (default: configs/default.yaml)")
    ap.add_argument("--shard", type=int, default=None, help="0-based shard index (default: $SGE_TASK_ID - 1)")
    ap.add_argument("--all", action="store_true", help="embed every shard sequentially (no-cluster fallback)")
    ap.add_argument("--limit", type=int, default=None, help="only embed the first N tracks of the shard (testing)")
    ap.add_argument("--num-shards", action="store_true", help="print the total shard count and exit (for -t 1-N)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.num_shards:
        tracks = parse_autotagging_tsv(cfg["paths"]["metadata_tsv"])
        print(shard_slices(len(tracks), int(cfg["embedding"]["shard_size"])))
        return

    if args.all:
        tracks = parse_autotagging_tsv(cfg["paths"]["metadata_tsv"])
        n = shard_slices(len(tracks), int(cfg["embedding"]["shard_size"]))
        for shard in range(n):
            embed_shard(cfg, shard, args.limit)
        return

    shard = args.shard
    if shard is None:
        task_id = os.environ.get("SGE_TASK_ID")
        if task_id is None or task_id == "undefined":
            ap.error("pass --shard/--all, or run under an SGE array job ($SGE_TASK_ID)")
        shard = int(task_id) - 1  # SGE task ids are 1-based
    embed_shard(cfg, shard, args.limit)


if __name__ == "__main__":
    main()
