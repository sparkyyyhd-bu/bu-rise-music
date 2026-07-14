#!/usr/bin/env bash
# Download the MTG-Jamendo autotagging subset (low-quality MP3s) + metadata.
#
# Environment: run on a machine WITH internet — your Mac, or the SCC
# login/transfer node (NOT a compute node; compute nodes have no internet).
# Downloads are resumable: both the official downloader and the sample mode
# below verify sha256 checksums and skip files that already completed, so
# just re-run this script if it is interrupted.
#
# Usage:
#   scripts/download_jamendo.sh            # full dataset (~55k tracks of low-quality MP3)
#   scripts/download_jamendo.sh 00         # only tar shard 00 (~550 tracks) — local sample
#
# Paths come from configs/default.yaml (override with PLAYLISTGEN_PATHS_* env
# vars, e.g. PLAYLISTGEN_PATHS_AUDIO_DIR=/scratch/$USER/jamendo/audio).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MIRROR="https://cdn.freesound.org/mtg-jamendo"   # 'mtg-fast' mirror; the
# slower canonical server is https://essentia.upf.edu/documentation/datasets/mtg-jamendo

# Resolve configured paths through the playlistgen config module.
eval "$("$PYTHON" - <<'EOF'
from playlistgen.config import load_config
cfg = load_config()
print(f'AUDIO_DIR="{cfg["paths"]["audio_dir"]}"')
print(f'METADATA_TSV="{cfg["paths"]["metadata_tsv"]}"')
EOF
)"

TOOLS_DIR="${PLAYLISTGEN_TOOLS_DIR:-$REPO_ROOT/data/mtg-jamendo-dataset}"
SAMPLE_TAR="${1:-}"   # optional: single two-digit tar id (00..99) for sampling

mkdir -p "$AUDIO_DIR" "$(dirname "$METADATA_TSV")"

# 1. Official repo: download tooling + the metadata TSVs + checksums.
if [ ! -d "$TOOLS_DIR" ]; then
  git clone --depth 1 https://github.com/MTG/mtg-jamendo-dataset "$TOOLS_DIR"
fi

# 2. Metadata: autotagging subset tags + human-readable artist/track names.
cp -f "$TOOLS_DIR/data/autotagging.tsv" "$METADATA_TSV"
if [ -f "$TOOLS_DIR/data/raw.meta.tsv" ]; then
  cp -f "$TOOLS_DIR/data/raw.meta.tsv" "$(dirname "$METADATA_TSV")/raw.meta.tsv"
fi
echo "metadata -> $METADATA_TSV"

if [ -z "$SAMPLE_TAR" ]; then
  # 3a. Full dataset via the official, checksum-verified, resumable downloader.
  "$PYTHON" "$TOOLS_DIR/scripts/download/download.py" \
    --dataset raw_30s --type audio-low --from mtg-fast \
    --unpack --remove "$AUDIO_DIR"
else
  # 3b. Sample mode: fetch one tar directly (the official tool has no range
  # option) and verify it against the repo's published sha256 list.
  TAR_NAME="raw_30s_audio-low-${SAMPLE_TAR}.tar"
  SHA_FILE="$TOOLS_DIR/data/download/raw_30s_audio-low_sha256_tars.txt"
  EXPECTED=$(awk -v f="$TAR_NAME" '$2 == f {print $1}' "$SHA_FILE")
  [ -n "$EXPECTED" ] || { echo >&2 "unknown tar id: $SAMPLE_TAR"; exit 1; }

  TAR_PATH="$AUDIO_DIR/$TAR_NAME"
  if [ ! -f "$TAR_PATH" ]; then
    curl -fL --retry 5 -C - -o "$TAR_PATH.part" \
      "$MIRROR/raw_30s/audio-low/$TAR_NAME"
    mv "$TAR_PATH.part" "$TAR_PATH"
  fi
  ACTUAL=$(shasum -a 256 "$TAR_PATH" | awk '{print $1}')
  if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo >&2 "checksum mismatch for $TAR_NAME — deleting; re-run to retry"
    rm -f "$TAR_PATH"
    exit 1
  fi
  tar -xf "$TAR_PATH" -C "$AUDIO_DIR"
  rm -f "$TAR_PATH"
  echo "sample tar $TAR_NAME unpacked"
fi

echo "audio -> $AUDIO_DIR"
