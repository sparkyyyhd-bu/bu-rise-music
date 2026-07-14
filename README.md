# playlistgen

`playlistgen` turns a free-text description into a playable Jamendo playlist.
It expands the prompt into audio captions and structured constraints with
OpenAI, retrieves tracks with pretrained LAION-CLAP embeddings, re-ranks them
against Jamendo tags, and orders the result along a greedy nearest-neighbor
path through embedding space.

This is an ablation-friendly MIR research prototype. It does not train a model,
and `--no-llm` provides an offline/raw-prompt baseline.

## Architecture

Indexing is the expensive, one-time stage: MTG-Jamendo audio is split into
10-second windows, embedded by CLAP, pooled to one normalized vector per track,
and merged into `embeddings.npy` plus an aligned `tracks.parquet`. Querying only
embeds text and performs a NumPy matrix multiplication, tag re-ranking, and
sequencing. The CLI and web server call the same pipeline implementation.

## Requirements

- Python 3.10 or 3.11
- macOS/Linux for local querying
- Internet access for initial dataset/checkpoint downloads
- BU SCC access with SGE (`qsub`) for building the full index, or a sufficiently
  capable local machine for the plain-bash fallback
- An OpenAI API key only when using LLM expansion

Do not download from an SCC compute node. Download on a login/transfer node and
place the resulting files on scratch storage.

## Local setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[dev]'
```

Configuration lives in `configs/default.yaml`. Every value can be overridden
without editing the file:

```bash
export PLAYLISTGEN_PATHS_AUDIO_DIR=/path/to/jamendo/audio
export PLAYLISTGEN_PATHS_METADATA_TSV=/path/to/jamendo/autotagging.tsv
export PLAYLISTGEN_PATHS_SHARDS_DIR=/path/to/embedding-shards
export PLAYLISTGEN_PATHS_INDEX_DIR=/path/to/index
export PLAYLISTGEN_PATHS_CHECKPOINT_DIR=/path/to/checkpoints
```

The general environment variable form is `PLAYLISTGEN_<SECTION>_<KEY>`.
Relative paths are resolved from the repository root, not the current shell
directory.

## Build the index

### 1. Download metadata and audio

The downloader clones the official MTG-Jamendo tooling, resumes interrupted
downloads, and verifies published SHA-256 checksums:

```bash
scripts/download_jamendo.sh       # full autotagging audio set
scripts/download_jamendo.sh 00    # one archive for a local sample
```

The current index format uses the exact CLAP checkpoint named in
`configs/default.yaml`: `music_audioset_epoch_15_esc_90.14.pt`. Pre-fetch it on
an internet-connected host (or SCC login node):

```bash
.venv/bin/python - <<'PY'
from playlistgen.clap_model import ensure_checkpoint
from playlistgen.config import load_config
print(ensure_checkpoint(load_config()))
PY
```

### 2. Embed on the BU SCC

Put audio, shards, the index, and preferably the checkpoint on scratch via the
environment overrides above. Create the environment on the login node, then:

```bash
mkdir -p logs
N=$(.venv/bin/python scripts/embed_audio.py --num-shards)
qsub -t 1-$N scripts/embed_audio.qsub
```

`embed_audio.qsub` is an SGE array job (not SLURM): task 1 embeds shard 0, task
2 embeds shard 1, and so on. Adjust its Python/CUDA modules and GPU resource
directives to match `module avail` and the current SCC queue policy. Each shard
is written atomically and completed shards are skipped on resubmission. Missing
or corrupt audio is recorded in the shard manifest instead of aborting a job.

For a non-SGE machine, run all shards sequentially:

```bash
.venv/bin/python scripts/embed_audio.py --all
```

For a quick local smoke sample:

```bash
.venv/bin/python scripts/embed_audio.py --shard 0 --limit 20
```

### 3. Merge shards

After array jobs finish, merge on a login node (no model or GPU is needed):

```bash
.venv/bin/python scripts/build_index.py --strict
```

This creates `embeddings.npy`, `tracks.parquet`, `vocab.json`, and
`skipped.json` under the configured index directory. Rows in the NumPy and
Parquet files are aligned. Omit `--strict` only when deliberately building a
partial index from incomplete shards.

Copy the final index and the configured checkpoint to the Mac. Audio files are
not required for querying because playlist rows contain direct Jamendo stream
URLs.

## Generate playlists

Offline/raw-prompt baseline:

```bash
.venv/bin/playlistgen 'tropical beach sunset, laid-back, steel drums' \
  -n 20 --no-llm --explain
```

OpenAI-expanded query (`gpt-5.5` by default):

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your key.
.venv/bin/playlistgen 'warm nocturnal jazz for reading' -n 15 --explain
```

Never commit the API key; `.env` is ignored. Without the key, LLM mode exits
with a clear error. Use `--json` for machine-readable output. `--explain`
shows captions, constraints, dropped out-of-vocabulary tags, and the sequencing
tradeoff. BPM constraints remain advisory because base Jamendo metadata has no
tempo; tempo should be enriched offline for that ablation.

## Web player

```bash
.venv/bin/playlistgen-web
```

Open <http://127.0.0.1:8501>. The local-only FastAPI server exposes
`POST /generate`; the static page can generate, click-to-play, auto-advance,
and display the interpreted query. A failed stream is greyed out and skipped.
There is no authentication, persistence, or database, so do not expose this
development server publicly.

## Tests

```bash
.venv/bin/pytest -m 'not clap' -q  # fast unit/API suite
.venv/bin/pytest -q                # includes real-checkpoint CLAP smoke tests
```

Tests never use the network. The full suite expects the configured checkpoint
to have been downloaded already; the small WAV fixtures are committed under
`tests/fixtures/`.

## Useful script help

```bash
.venv/bin/python scripts/embed_audio.py --help
.venv/bin/python scripts/build_index.py --help
.venv/bin/playlistgen --help
```

The direct Jamendo stream URL is stored during index construction, so old
indexes may need rebuilding if Jamendo changes its playback endpoint.
