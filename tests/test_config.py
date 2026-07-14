"""Config loading + env override behavior. No network."""

import textwrap


def test_default_config_loads(cfg):
    assert cfg["clap"]["checkpoint"].endswith(".pt")
    assert cfg["chunking"]["pooling"] in ("mean", "max")
    assert 0 <= cfg["retrieval"]["alpha"] <= 1


def test_env_override(tmp_path, monkeypatch):
    from playlistgen.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text(
        textwrap.dedent(
            """
            paths:
              index_dir: /abs/index
            retrieval:
              top_n: 200
              alpha: 0.7
            chunking:
              pooling: mean
            """
        )
    )
    monkeypatch.setenv("PLAYLISTGEN_RETRIEVAL_TOP_N", "50")
    monkeypatch.setenv("PLAYLISTGEN_RETRIEVAL_ALPHA", "0.5")
    monkeypatch.setenv("PLAYLISTGEN_CHUNKING_POOLING", "max")
    cfg = load_config(p)
    assert cfg["retrieval"]["top_n"] == 50
    assert cfg["retrieval"]["alpha"] == 0.5
    assert cfg["chunking"]["pooling"] == "max"
    assert cfg["paths"]["index_dir"] == "/abs/index"
