"""OpenAI query expansion: free-text prompt -> validated ExpandedQuery.

Environment: Mac (query side); needs network + OPENAI_API_KEY. One API
call per user query (plus at most one retry on invalid JSON). The CLI's
--no-llm flag bypasses this module entirely so the pipeline also works
offline and so LLM expansion can be ablated.

The API key is only ever read from the environment by the OpenAI SDK --
never log it, never echo it, never write it to disk.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from . import jamendo_vocab


class ExpandedQuery(BaseModel):
    """Structured interpretation of a free-text playlist prompt."""

    captions: list[str] = Field(min_length=1, max_length=8)
    genres: list[str] = []
    moods: list[str] = []
    instruments: list[str] = []
    bpm_range: tuple[int, int] | None = None
    energy: Literal["low", "medium", "high"] | None = None
    vocals: Literal["vocal", "instrumental", "either"] = "either"

    # Filled in locally, not by the model:
    dropped_tags: list[str] = []    # tags the LLM proposed that aren't in the vocab
    source: str = "llm"             # "llm" | "raw"

    @classmethod
    def from_raw_prompt(cls, prompt: str) -> "ExpandedQuery":
        """Fallback/--no-llm mode: embed the user's prompt verbatim, no constraints."""
        return cls(captions=[prompt], source="raw")


class ExpansionError(RuntimeError):
    """OpenAI did not produce a valid ExpandedQuery after one retry."""


class MissingAPIKeyError(RuntimeError):
    pass


class ExpansionPayload(BaseModel):
    """Exact schema requested from OpenAI Structured Outputs."""

    captions: list[str] = Field(min_length=1, max_length=8)
    genres: list[str]
    moods: list[str]
    instruments: list[str]
    # Use a constrained homogeneous array for OpenAI JSON Schema compatibility;
    # ExpandedQuery converts the parsed two-item list to its tuple field.
    bpm_range: list[int] | None = Field(min_length=2, max_length=2)
    energy: Literal["low", "medium", "high"] | None
    vocals: Literal["vocal", "instrumental", "either"]


SYSTEM_PROMPT = """\
You translate a user's free-text playlist request into retrieval inputs for a
CLAP (contrastive language-audio pretraining) music search system over the
MTG-Jamendo dataset.

Return data matching this schema:

{{
  "captions": [3-5 diverse audio-caption paraphrases of the request, each
               phrased the way CLAP training captions are phrased, e.g.
               "a laid-back reggae track with steel drums and warm bass".
               Vary instrumentation/mood/genre emphasis across captions but
               stay faithful to the request.],
  "genres": [0-5 tags, ONLY from the genre vocabulary below],
  "moods": [0-5 tags, ONLY from the mood/theme vocabulary below],
  "instruments": [0-5 tags, ONLY from the instrument vocabulary below],
  "bpm_range": [low, high] integers, or null if the request implies no tempo,
  "energy": "low" | "medium" | "high" | null,
  "vocals": "vocal" | "instrumental" | "either"
}}

Genre vocabulary: {genres}

Mood/theme vocabulary: {moods}

Instrument vocabulary: {instruments}
"""


def _vocab_lists(vocab: dict[str, Any] | None) -> tuple[list[str], list[str], list[str]]:
    """Prefer the index's own vocab.json; fall back to the bundled lists."""
    # An index may intentionally cover only one tag category, such as the
    # official Jamendo mood/theme subset. Empty lists are meaningful and must
    # not trigger the bundled full-dataset vocabulary fallback.
    if vocab is not None and all(k in vocab for k in ("genres", "moods", "instruments")):
        return vocab["genres"], vocab["moods"], vocab["instruments"]
    return jamendo_vocab.GENRES, jamendo_vocab.MOODS, jamendo_vocab.INSTRUMENTS


def _filter_vocab(query: ExpandedQuery, genres: list[str], moods: list[str],
                  instruments: list[str]) -> ExpandedQuery:
    """Drop hallucinated (out-of-vocab) tags rather than failing the whole call."""
    dropped: list[str] = []

    def keep(values: list[str], vocab: list[str]) -> list[str]:
        allowed = set(vocab)
        kept = [v for v in values if v in allowed]
        dropped.extend(v for v in values if v not in allowed)
        return kept

    query.genres = keep(query.genres, genres)
    query.moods = keep(query.moods, moods)
    query.instruments = keep(query.instruments, instruments)
    query.dropped_tags = dropped
    return query


def expand_query(
    prompt: str,
    cfg: dict[str, Any],
    vocab: dict[str, Any] | None = None,
    client: Any = None,
) -> ExpandedQuery:
    """One OpenAI call (plus at most one retry) -> validated ExpandedQuery.

    Raises MissingAPIKeyError / ExpansionError; the caller decides whether to
    fall back to ExpandedQuery.from_raw_prompt().
    `client` is injectable for tests; no network happens in the test suite.
    """
    if client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not set. Add it to .env or export it to enable LLM query "
                "expansion, or pass --no-llm to run with the raw prompt."
            )
        from openai import OpenAI

        client = OpenAI()

    genres, moods, instruments = _vocab_lists(vocab)
    system = SYSTEM_PROMPT.format(
        genres=", ".join(genres),
        moods=", ".join(moods),
        instruments=", ".join(instruments),
    )

    user_input = prompt
    last_error: Exception | None = None
    for _attempt in range(2):  # initial call + one retry with the error appended
        try:
            response = client.responses.parse(
                model=cfg["expansion"]["model"],
                max_output_tokens=int(cfg["expansion"]["max_tokens"]),
                instructions=system,
                input=user_input,
                text_format=ExpansionPayload,
            )
            if response.output_parsed is None:
                raise ValueError("OpenAI response contained no parsed structured output")
            query = ExpandedQuery.model_validate(response.output_parsed.model_dump())
            return _filter_vocab(query, genres, moods, instruments)
        except Exception as exc:
            last_error = exc
            user_input = (
                f"{prompt}\n\nThe previous response could not be parsed for the required "
                f"schema. Error: {exc}. Return a corrected structured response."
            )

    raise ExpansionError(f"query expansion failed after retry: {last_error}")
