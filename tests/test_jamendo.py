"""Jamendo metadata parsing tests. No network."""

from playlistgen.jamendo import (
    parse_autotagging_tsv,
    resolve_audio_path,
    stream_url,
)

TSV = """\
TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS
track_0000202\tartist_0000007\talbum_0000024\t02/202.mp3\t215.7\tgenre---rock\tgenre---pop\tmood/theme---happy\tinstrument---guitar
track_0000005\tartist_0000001\talbum_0000001\t05/5.mp3\t180.0\tgenre---techno
"""


def test_parse_autotagging_tsv(tmp_path):
    p = tmp_path / "autotagging.tsv"
    p.write_text(TSV)
    tracks = parse_autotagging_tsv(p)
    assert [t.track_id for t in tracks] == [5, 202], "sorted by id"
    t = tracks[1]
    assert t.genres == ["rock", "pop"]
    assert t.moods == ["happy"]
    assert t.instruments == ["guitar"]
    assert t.all_tags == ["rock", "pop", "happy", "guitar"]
    assert t.duration == 215.7
    assert t.path == "02/202.mp3"


def test_stream_url_pattern():
    url = stream_url(202)
    assert "trackid=202" in url and url.startswith("https://")


def test_resolve_audio_path_prefers_exact_then_low(tmp_path):
    (tmp_path / "02").mkdir()
    low = tmp_path / "02" / "202.low.mp3"
    low.write_bytes(b"x")
    assert resolve_audio_path(tmp_path, "02/202.mp3") == low
    exact = tmp_path / "02" / "202.mp3"
    exact.write_bytes(b"x")
    assert resolve_audio_path(tmp_path, "02/202.mp3") == exact
    assert resolve_audio_path(tmp_path, "99/999.mp3") is None
