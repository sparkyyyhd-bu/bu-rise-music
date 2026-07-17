"""Retrieval, rerank, and sequencing unit tests. No network, no CLAP."""

import numpy as np

from playlistgen.expand import ExpandedQuery
from playlistgen.rerank import Candidate, diversify_artists, rerank
from playlistgen.retrieve import cosine_top_n
from playlistgen.sequence import sequence_playlist

from .helpers import basis, make_index


def test_cosine_top_n_orders_by_similarity():
    index = make_index(10)
    query = (0.9 * basis(3) + 0.1 * basis(7)).astype(np.float32)
    query /= np.linalg.norm(query)
    rows, sims = cosine_top_n(query, index, 3)
    assert rows[0] == 3 and rows[1] == 7
    assert sims[0] > sims[1] > sims[2]
    assert len(rows) == 3


def test_cosine_top_n_caps_at_index_size():
    index = make_index(4)
    rows, _ = cosine_top_n(basis(0), index, 100)
    assert len(rows) == 4


def test_rerank_tag_overlap_boosts_matching_tracks():
    index = make_index(10)
    # rows 0 (reggae/calm) and 1 (techno/calm/voice), near-equal cosine.
    rows = np.array([0, 1])
    cosines = np.array([0.50, 0.49], dtype=np.float32)
    q = ExpandedQuery(captions=["x"], genres=["techno"], moods=[], instruments=[])
    out = rerank(rows, cosines, index, q, alpha=0.5, beta=0.5)
    assert out[0].row == 1, "tag match should outweigh 0.01 cosine deficit"
    assert "techno" in out[0].matched_tags
    # And with beta=0 the pure-cosine order wins.
    out = rerank(rows, cosines, index, q, alpha=1.0, beta=0.0)
    assert out[0].row == 0


def test_rerank_mainstream_score_boosts_popular_artist():
    index = make_index(10)
    index.tracks["mainstream_score"] = 0.0
    index.tracks.loc[1, "mainstream_score"] = 1.0  # row 1 is the "mainstream" artist
    rows = np.array([0, 1])
    cosines = np.array([0.50, 0.49], dtype=np.float32)

    # gamma=0: near-equal cosine keeps row 0 on top.
    out = rerank(rows, cosines, index, None, alpha=1.0, beta=0.0, gamma=0.0)
    assert out[0].row == 0

    # gamma>0: the mainstream_score bonus outweighs the 0.01 cosine deficit.
    out = rerank(rows, cosines, index, None, alpha=1.0, beta=0.0, gamma=0.5)
    assert out[0].row == 1


def test_rerank_mainstream_score_defaults_to_zero_when_column_missing():
    index = make_index(10)  # no mainstream_score column
    rows = np.array([0, 1])
    cosines = np.array([0.50, 0.49], dtype=np.float32)
    out = rerank(rows, cosines, index, None, alpha=1.0, beta=0.0, gamma=0.9)
    assert out[0].row == 0, "gamma should be a no-op without a mainstream_score column"


def test_rerank_without_constraints_is_cosine_order():
    index = make_index(6)
    rows = np.array([5, 2, 4])
    cosines = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    out = rerank(rows, cosines, index, None, alpha=0.7, beta=0.3)
    assert [c.row for c in out] == [5, 2, 4]
    assert all(c.tag_overlap == 0.0 for c in out)


def test_rerank_enforces_explicit_genre_when_enough_tracks_match():
    index = make_index(10)
    rows = np.arange(10)
    cosines = np.linspace(1.0, 0.1, 10).astype(np.float32)
    q = ExpandedQuery(captions=["x"], genres=["techno"])
    out = rerank(rows, cosines, index, q, alpha=0.7, beta=0.3, keep_at_least=4)
    assert len(out) == 5
    assert all("techno" in index.tracks.iloc[c.row]["genres"] for c in out)


def test_rerank_instrumental_filter_drops_voice_tracks():
    index = make_index(10)  # rows 1 and 3 have the "voice" instrument tag
    rows = np.arange(10)
    cosines = np.linspace(0.9, 0.1, 10).astype(np.float32)
    q = ExpandedQuery(captions=["x"], vocals="instrumental")
    out = rerank(rows, cosines, index, q, alpha=1.0, beta=0.0, keep_at_least=1)
    kept = {c.row for c in out}
    assert 1 not in kept and 3 not in kept

    q = ExpandedQuery(captions=["x"], vocals="vocal")
    out = rerank(rows, cosines, index, q, alpha=1.0, beta=0.0, keep_at_least=1)
    assert {c.row for c in out} == {1, 3}


def test_rerank_vocal_filter_is_soft_when_pool_too_small():
    index = make_index(10)
    rows = np.arange(10)
    cosines = np.linspace(0.9, 0.1, 10).astype(np.float32)
    q = ExpandedQuery(captions=["x"], vocals="vocal")
    # Only 2 voice tracks exist; demanding 5 keeps the unfiltered pool.
    out = rerank(rows, cosines, index, q, alpha=1.0, beta=0.0, keep_at_least=5)
    assert len(out) == 10


def test_sequence_greedy_nn_path():
    index = make_index(10)
    # Build a pool whose embeddings we control through index rows: use rows
    # 0..3 but give candidate cosines so 0 is the seed; the greedy path then
    # follows embedding similarity, which for basis vectors is degenerate --
    # so instead verify: seed first, all tracks kept exactly once.
    from playlistgen.rerank import Candidate

    pool = [
        Candidate(row=r, cosine=c, tag_overlap=0, combined=c, matched_tags=[])
        for r, c in zip([2, 0, 3, 1], [0.9, 0.8, 0.7, 0.6])
    ]
    out = sequence_playlist(pool, index, length=4)
    assert out[0].row == 2, "seed must be the highest-scoring candidate"
    assert sorted(c.row for c in out) == [0, 1, 2, 3]


def test_sequence_follows_embedding_neighbors():
    import pandas as pd
    from playlistgen.rerank import Candidate
    from playlistgen.retrieve import PlaylistIndex

    # 1-D line of angles: greedy NN from the seed walks monotonically.
    angles = np.array([0.0, 0.1, 0.2, 0.4, 0.8])
    emb = np.zeros((5, 512), dtype=np.float32)
    emb[:, 0] = np.cos(angles)
    emb[:, 1] = np.sin(angles)
    df = pd.DataFrame({"track_id": range(5)})
    index = PlaylistIndex(embeddings=emb, tracks=df)
    pool = [
        Candidate(row=r, cosine=1 - 0.01 * r, tag_overlap=0, combined=1 - 0.01 * r, matched_tags=[])
        for r in range(5)
    ]
    out = sequence_playlist(pool, index, length=5)
    assert [c.row for c in out] == [0, 1, 2, 3, 4]


def test_diversify_artists_caps_repetition():
    index = make_index(6)
    index.tracks.loc[[0, 1, 2, 3], "artist"] = "same artist"
    pool = [
        Candidate(row=r, cosine=1 - r / 10, tag_overlap=0, combined=1 - r / 10,
                  matched_tags=[])
        for r in range(6)
    ]
    out = diversify_artists(pool, index, length=4, max_per_artist=2)
    artists = [index.tracks.iloc[c.row]["artist"] for c in out]
    assert artists.count("same artist") == 2
    assert len(out) == 4
