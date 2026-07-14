"""Configuration loading for playlistgen.

Environment: Mac and SCC. All paths and parameters come from configs/default.yaml;
any value can be overridden with an environment variable named
PLAYLISTGEN_<SECTION>_<KEY> (case-insensitive match on the YAML keys), e.g.

    PLAYLISTGEN_PATHS_AUDIO_DIR=/scratch/$USER/jamendo/audio
    PLAYLISTGEN_CHUNKING_POOLING=max

Never hardcode paths elsewhere in the codebase -- go through load_config().
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ENV_PREFIX = "PLAYLISTGEN"

# Repo root = two levels up from this file's package dir (src/playlistgen/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"

# Local developer convenience. Existing shell environment variables win, and
# the ignored .env file is never read or printed by application code.
load_dotenv(_REPO_ROOT / ".env", override=False)


def _coerce(raw: str, previous: Any) -> Any:
    """Coerce an env-var string to the type of the value it replaces."""
    if isinstance(previous, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(previous, int) and not isinstance(previous, bool):
        return int(raw)
    if isinstance(previous, float):
        return float(raw)
    return raw


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            env_name = f"{ENV_PREFIX}_{section}_{key}".upper()
            if env_name in os.environ:
                values[key] = _coerce(os.environ[env_name], value)
    return cfg


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config with env-var overrides applied.

    `path` defaults to configs/default.yaml in the repo, or the
    PLAYLISTGEN_CONFIG env var if set.
    """
    if path is None:
        path = os.environ.get(f"{ENV_PREFIX}_CONFIG", DEFAULT_CONFIG_PATH)
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg = _apply_env_overrides(cfg)
    # Resolve relative paths against the repo root so scripts work from any cwd.
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        if not p.is_absolute():
            cfg["paths"][key] = str(_REPO_ROOT / p)
    return cfg
