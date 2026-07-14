"""Shared test fixtures. No network access in tests (except -m clap, which
loads the local CLAP checkpoint and downloads it on first run)."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def cfg():
    from playlistgen.config import load_config

    return load_config()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES
