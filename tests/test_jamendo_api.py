"""Jamendo live-API popularity ranking tests. No real network calls (fake session)."""

from playlistgen.jamendo_api import JamendoAPIError, fetch_artist_popularity


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    """Records requests and returns artists in a fixed popularity order."""

    def __init__(self, order):
        self.order = order  # artist ids, most popular first
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        requested_ids = {int(v) for k, v in params if k == "id"}
        results = [{"id": a} for a in self.order if a in requested_ids]
        return FakeResponse({"headers": {"status": "success"}, "results": results})


def test_fetch_artist_popularity_scores_by_rank():
    session = FakeSession(order=[30, 10, 20])  # 30 most popular, then 10, then 20
    scores = fetch_artist_popularity([10, 20, 30], "fake-client", session=session)
    assert scores[30] == 1.0
    assert scores[10] == 0.5
    assert scores[20] == 0.0


def test_fetch_artist_popularity_single_artist():
    session = FakeSession(order=[42])
    scores = fetch_artist_popularity([42], "fake-client", session=session)
    assert scores == {42: 1.0}


def test_fetch_artist_popularity_empty_input():
    session = FakeSession(order=[])
    assert fetch_artist_popularity([], "fake-client", session=session) == {}


def test_fetch_artist_popularity_missing_artist_ranked_last():
    # Jamendo has no record of artist 99; it should still get the lowest score.
    session = FakeSession(order=[10, 20])
    scores = fetch_artist_popularity([10, 20, 99], "fake-client", session=session)
    assert scores[10] == 1.0
    assert scores[20] == 0.5
    assert scores[99] == 0.0


def test_fetch_artist_popularity_raises_on_api_error():
    class FailingSession:
        def get(self, url, params, timeout):
            return FakeResponse({"headers": {"status": "failed", "code": 11}, "results": []})

    try:
        fetch_artist_popularity([1], "fake-client", session=FailingSession())
        assert False, "expected JamendoAPIError"
    except JamendoAPIError:
        pass
