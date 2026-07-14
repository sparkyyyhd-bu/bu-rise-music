"""Query-expansion tests with a mocked OpenAI client. No network."""

from types import SimpleNamespace

import pytest

from playlistgen.expand import (
    ExpandedQuery,
    ExpansionError,
    MissingAPIKeyError,
    expand_query,
)

CFG = {"expansion": {"model": "gpt-5.5", "max_tokens": 1024}}

GOOD = {
    "captions": ["a laid-back reggae track with steel drums and warm bass"],
    "genres": ["reggae"],
    "moods": ["calm"],
    "instruments": ["guitar"],
    "bpm_range": [70, 95],
    "energy": "low",
    "vocals": "either",
}


class FakeClient:
    """Returns queued parsed payloads, records the requests it receives."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []
        self.responses = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        if payload is None:
            return SimpleNamespace(output_parsed=None)
        schema = kwargs["text_format"]
        return SimpleNamespace(output_parsed=schema.model_validate(payload))


def test_valid_json_first_try():
    client = FakeClient(GOOD)
    q = expand_query("chill reggae", CFG, client=client)
    assert q.genres == ["reggae"]
    assert q.bpm_range == (70, 95)
    assert q.source == "llm"
    assert len(client.calls) == 1
    # Vocab is embedded in the system prompt.
    assert "reggae" in client.calls[0]["instructions"]
    assert client.calls[0]["model"] == "gpt-5.5"


def test_out_of_vocab_tags_are_dropped_not_fatal():
    bad_tags = dict(GOOD, genres=["reggae", "hyperpolka"], instruments=["steeldrum"])
    client = FakeClient(bad_tags)
    q = expand_query("x", CFG, client=client)
    assert q.genres == ["reggae"]
    assert q.instruments == []
    assert set(q.dropped_tags) == {"hyperpolka", "steeldrum"}


def test_retry_once_with_error_appended_then_succeed():
    client = FakeClient(None, GOOD)
    q = expand_query("x", CFG, client=client)
    assert q.genres == ["reggae"]
    assert len(client.calls) == 2
    assert "previous response" in client.calls[1]["input"]


def test_two_failures_raise_expansion_error():
    client = FakeClient(None, None)
    with pytest.raises(ExpansionError):
        expand_query("x", CFG, client=client)
    assert len(client.calls) == 2


def test_api_failures_are_wrapped_as_expansion_error():
    client = FakeClient(RuntimeError("provider unavailable"), RuntimeError("again"))
    with pytest.raises(ExpansionError, match="again"):
        expand_query("x", CFG, client=client)


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="--no-llm"):
        expand_query("x", CFG)


def test_index_vocab_preferred_over_bundled():
    vocab = {"genres": ["zzz"], "moods": ["calm"], "instruments": ["guitar"]}
    client = FakeClient(dict(GOOD, genres=["zzz"]))
    q = expand_query("x", CFG, vocab=vocab, client=client)
    assert q.genres == ["zzz"]
    assert "zzz" in client.calls[0]["instructions"]


def test_raw_prompt_mode():
    q = ExpandedQuery.from_raw_prompt("lofi beats")
    assert q.captions == ["lofi beats"]
    assert q.source == "raw"
    assert q.vocals == "either"
