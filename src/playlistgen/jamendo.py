"""MTG-Jamendo metadata parsing and URL helpers.

Environment: Mac and SCC. Pure-Python, no model dependencies.

The autotagging TSV (data/autotagging.tsv in the mtg-jamendo-dataset repo) has
a header row and one row per track:

    TRACK_ID  ARTIST_ID  ALBUM_ID  PATH  DURATION  TAGS...

where TAGS is a variable number of trailing tab-separated fields like
"genre--rock", "mood/theme--happy", "instrument--guitar".
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

# Direct-playable MP3 for a Jamendo track id. Verified 2026-07: returns
# audio/mpeg with HTTP range support (the pattern used by Jamendo's developer
# API and the mtg-jamendo demos; the dataset repo itself only links the
# track page https://www.jamendo.com/track/<id>).
STREAM_URL_TEMPLATE = "https://mp3l.jamendo.com/?trackid={track_id}&format=mp31&from=app-devsite"

TAG_CATEGORIES = ("genre", "mood/theme", "instrument")


def stream_url(track_id: int) -> str:
    return STREAM_URL_TEMPLATE.format(track_id=track_id)


@dataclass
class JamendoTrack:
    track_id: int                 # numeric Jamendo id (e.g. 202)
    artist_id: int
    album_id: int
    path: str                     # relative mp3 path from the TSV, e.g. "02/202.mp3"
    duration: float               # seconds
    genres: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)        # mood/theme tags
    instruments: list[str] = field(default_factory=list)
    artist_name: str = ""
    track_name: str = ""

    @property
    def all_tags(self) -> list[str]:
        return self.genres + self.moods + self.instruments


def _numeric_id(raw: str) -> int:
    # "track_0000202" -> 202 ; also accepts bare numbers.
    return int(raw.rsplit("_", 1)[-1])


def parse_autotagging_tsv(path: str | Path) -> list[JamendoTrack]:
    tracks: list[JamendoTrack] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        if not header or not header[0].upper().startswith("TRACK"):
            raise ValueError(f"{path} does not look like an autotagging TSV")
        for row in reader:
            if len(row) < 5:
                continue
            genres, moods, instruments = [], [], []
            for tag in row[5:]:
                # Tags look like "genre---rock" / "mood/theme---happy" /
                # "instrument---guitar" (triple-dash separator).
                category, sep, value = tag.strip().partition("---")
                if not sep:
                    continue
                if category == "genre":
                    genres.append(value)
                elif category == "mood/theme":
                    moods.append(value)
                elif category == "instrument":
                    instruments.append(value)
            tracks.append(
                JamendoTrack(
                    track_id=_numeric_id(row[0]),
                    artist_id=_numeric_id(row[1]),
                    album_id=_numeric_id(row[2]),
                    path=row[3],
                    duration=float(row[4]),
                    genres=genres,
                    moods=moods,
                    instruments=instruments,
                )
            )
    tracks.sort(key=lambda t: t.track_id)
    return tracks


def attach_names(tracks: list[JamendoTrack], raw_meta_tsv: str | Path) -> None:
    """Fill artist_name/track_name from the repo's data/raw.meta.tsv (optional)."""
    by_id = {t.track_id: t for t in tracks}
    with open(raw_meta_tsv, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            tid = _numeric_id(row["TRACK_ID"])
            t = by_id.get(tid)
            if t is not None:
                t.artist_name = row.get("ARTIST_NAME", "") or ""
                t.track_name = row.get("TRACK_NAME", "") or ""


def resolve_audio_path(audio_dir: str | Path, rel_path: str) -> Path | None:
    """Find the audio file for a TSV PATH entry.

    The full-quality archives unpack to `<dir>/<path>` (e.g. 02/202.mp3); the
    audio-low archives use a `.low.mp3` suffix (02/202.low.mp3). Try both.
    """
    audio_dir = Path(audio_dir)
    candidates = [
        audio_dir / rel_path,
        audio_dir / rel_path.replace(".mp3", ".low.mp3"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None
