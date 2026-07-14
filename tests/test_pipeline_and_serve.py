"""End-to-end pipeline + web API tests over the synthetic index. No network,
no CLAP -- the encoder and (where relevant) the LLM client are fakes."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from playlistgen.pipeline import generate_playlist
from playlistgen.expand import MissingAPIKeyError

from .helpers import FakeEncoder, basis, make_index

CFG = {
    "retrieval": {"top_n": 8, "alpha": 0.7, "beta": 0.3},
    "expansion": {"model": "gpt-5.5", "max_tokens": 1024},
}


def test_generate_playlist_no_llm():
    index = make_index(10)
    encoder = FakeEncoder({"give me track three": basis(3)})
    result = generate_playlist(
        "give me track three", 5, CFG, index, encoder, use_llm=False
    )
    assert len(result["playlist"]) == 5
    first = result["playlist"][0]
    assert first["rank"] == 1
    assert first["track_id"] == 1003, "raw prompt should hit its basis track first"
    assert first["similarity"] == pytest.approx(1.0, abs=1e-4)
    assert first["stream_url"].startswith("https://")
    assert result["expansion"]["source"] == "raw"
    # JSON-serializable contract for --json and the web API.
    json.dumps(result)


def test_generate_playlist_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    index = make_index(5)
    with pytest.raises(MissingAPIKeyError):
        generate_playlist("x", 3, CFG, index, FakeEncoder(), use_llm=True)


def test_generate_playlist_with_fake_llm():
    from .test_expand import FakeClient

    index = make_index(10)
    expansion = {
        "captions": ["cap-a", "cap-b"],
        "genres": ["techno"],
        "moods": [],
        "instruments": [],
        "bpm_range": [120, 130],
        "energy": "high",
        "vocals": "either",
    }
    encoder = FakeEncoder({"cap-a": basis(2), "cap-b": basis(2)})
    result = generate_playlist(
        "banging techno", 4, CFG, index, encoder,
        use_llm=True, expansion_client=FakeClient(expansion),
    )
    assert result["expansion"]["source"] == "llm"
    assert result["expansion"]["genres"] == ["techno"]
    # bpm note is surfaced since Jamendo has no tempo metadata.
    assert any("bpm" in n.lower() for n in result["notes"])
    ranks = [p["rank"] for p in result["playlist"]]
    assert ranks == [1, 2, 3, 4]


@pytest.fixture()
def web_client(monkeypatch, tmp_path):
    """serve.create_app with the index/encoder swapped for fakes."""
    import playlistgen.serve as serve

    monkeypatch.setattr(
        serve.PlaylistIndex, "load", classmethod(lambda cls, _dir: make_index(10))
    )
    monkeypatch.setattr(
        serve, "ClapEncoder", lambda cfg: FakeEncoder({"give me track three": basis(3)})
    )
    app = serve.create_app()
    return TestClient(app)


def test_serve_home_and_static(web_client):
    r = web_client.get("/")
    assert r.status_code == 200
    assert "playlistgen" in r.text
    assert web_client.get("/static/app.js").status_code == 200
    assert web_client.get("/static/style.css").status_code == 200


def test_serve_generate_no_llm(web_client):
    r = web_client.post(
        "/generate",
        json={"prompt": "give me track three", "n": 3, "use_llm": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["playlist"]) == 3
    assert body["playlist"][0]["track_id"] == 1003
    assert {"rank", "artist", "title", "similarity", "tags", "stream_url"} <= set(
        body["playlist"][0]
    )


def test_serve_generate_missing_key_is_400(web_client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = web_client.post("/generate", json={"prompt": "x", "n": 3, "use_llm": True})
    assert r.status_code == 400
    assert "no-llm" in r.json()["detail"].lower() or "OPENAI_API_KEY" in r.json()["detail"]


def test_serve_rejects_bad_request(web_client):
    assert web_client.post("/generate", json={"prompt": "", "n": 3}).status_code == 422
    assert web_client.post("/generate", json={"prompt": "x", "n": 0}).status_code == 422
