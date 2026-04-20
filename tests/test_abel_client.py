"""Tests for Abel client helpers and example integration."""

import requests
import pandas as pd

from causal_edge.plugins.abel.client import AbelClient, normalize_public_node_id
from examples.causal_demo.engine import CausalDemoEngine, resolve_price_column


def test_normalize_public_node_id_ethusd():
    assert normalize_public_node_id("ETHUSD") == "ETHUSD.price"
    assert normalize_public_node_id("ETH") == "ETHUSD.price"
    assert normalize_public_node_id("ETHUSD_close") == "ETHUSD.price"


def test_resolve_price_column_prefers_close():
    df = pd.DataFrame({"close": [1.0], "price": [2.0], "volume": [3.0]})
    assert resolve_price_column(df, "price") == "close"
    assert resolve_price_column(df, "volume") == "volume"


def test_causal_demo_runs_with_default_parent_metadata():
    engine = CausalDemoEngine()
    positions, dates, prices = engine.compute_signals()
    assert len(positions) == len(dates) == len(prices)
    assert positions[-1] >= 0.0


def test_causal_demo_realistic_csv_mapping_demo(tmp_path):
    price_path = tmp_path / "ethusd.csv"
    pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=5),
            "close": [100, 101, 102, 103, 104],
            "volume": [10, 11, 12, 13, 14],
        }
    ).to_csv(price_path, index=False)
    df = pd.read_csv(price_path)
    assert resolve_price_column(df, "price") == "close"


def test_fetch_bars_uses_market_prod_base_url(monkeypatch):
    monkeypatch.delenv("ABEL_CAP_BASE_URL", raising=False)

    class StubSession:
        def __init__(self):
            self.calls = []

        def post(self, url, json=None, headers=None, timeout=20):
            self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return StubResponse({"data": []})

    class StubResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    session = StubSession()
    client = AbelClient(session=session)
    client.fetch_bars(
        symbols=["ETHUSD"],
        start=None,
        end=None,
        timeframe="1d",
        limit=10,
        fields=None,
        api_key="abel_test",
    )

    assert session.calls[0]["url"] == "https://cap.abel.ai/api/market/day_bar"
    assert session.calls[0]["json"]["symbols"] == ["ETHUSD"]
    assert session.calls[0]["headers"]["Authorization"] == "Bearer abel_test"
    assert "api-key" not in session.calls[0]["headers"]


def test_fetch_bars_uses_custom_base_url():
    class StubSession:
        def __init__(self):
            self.calls = []

        def post(self, url, json=None, headers=None, timeout=20):
            self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return StubResponse({"data": []})

    class StubResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    session = StubSession()
    client = AbelClient(cap_base_url="https://cap.custom.abel.ai/api/", session=session)
    client.fetch_bars(
        symbols=["ETHUSD"],
        start=None,
        end=None,
        timeframe="1d",
        limit=10,
        fields=None,
        api_key="abel_test",
    )

    assert session.calls[0]["url"] == "https://cap.custom.abel.ai/api/market/day_bar"


def test_fetch_bars_preserves_bearer_auth_header():
    class StubSession:
        def __init__(self):
            self.calls = []

        def post(self, url, json=None, headers=None, timeout=20):
            self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return StubResponse({"data": []})

    class StubResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    session = StubSession()
    client = AbelClient(session=session)
    client.fetch_bars(
        symbols=["ETHUSD"],
        start=None,
        end=None,
        timeframe="1d",
        limit=10,
        fields=None,
        api_key="Bearer abel_test",
    )

    assert session.calls[0]["headers"]["Authorization"] == "Bearer abel_test"
    assert "api-key" not in session.calls[0]["headers"]


def test_fetch_bars_strips_runtime_only_fields_from_market_request():
    class StubSession:
        def __init__(self):
            self.calls = []

        def post(self, url, json=None, headers=None, timeout=20):
            self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return StubResponse({"data": []})

    class StubResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    session = StubSession()
    client = AbelClient(session=session)
    client.fetch_bars(
        symbols=["ETHUSD"],
        start=None,
        end=None,
        timeframe="1d",
        limit=10,
        fields=["timestamp", "symbol", "close", "volume"],
        api_key="abel_test",
    )

    assert session.calls[0]["json"]["fields"] == ["close", "volume"]


def test_discover_uses_cap_prod_base_url(monkeypatch):
    monkeypatch.delenv("ABEL_CAP_BASE_URL", raising=False)

    class StubSession:
        def __init__(self):
            self.calls = []

        def post(self, url, json=None, headers=None, timeout=20):
            self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return StubResponse({"result": []})

    class StubResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    session = StubSession()
    client = AbelClient(session=session)
    client.discover_parents(node_id="ETHUSD", limit=5, api_key="abel_test")

    assert session.calls[0]["url"] == "https://cap.abel.ai/api/cap"
    assert session.calls[0]["json"]["verb"] == "traverse.parents"


def test_discover_uses_custom_base_url():
    class StubSession:
        def __init__(self):
            self.calls = []

        def post(self, url, json=None, headers=None, timeout=20):
            self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return StubResponse({"result": []})

    class StubResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    session = StubSession()
    client = AbelClient(cap_base_url="https://cap.custom.abel.ai/api/", session=session)
    client.discover_parents(node_id="ETHUSD", limit=5, api_key="abel_test")

    assert session.calls[0]["url"] == "https://cap.custom.abel.ai/api/cap"


def test_fetch_bars_falls_back_to_curl_on_windows_connection_reset(monkeypatch):
    class FailingSession:
        def post(self, url, json=None, headers=None, timeout=20):
            raise requests.exceptions.ConnectionError(
                "('Connection aborted.', ConnectionResetError(10054, 'reset', None, 10054, None))"
            )

    fallback_calls = []

    monkeypatch.setattr("causal_edge.plugins.abel.client.sys.platform", "win32")
    monkeypatch.setattr("causal_edge.plugins.abel.client.shutil.which", lambda name: "C:/Windows/System32/curl.exe")
    monkeypatch.setattr(
        "causal_edge.plugins.abel.client._post_with_curl",
        lambda **kwargs: fallback_calls.append(kwargs) or {"data": [{"symbol": "ETHUSD"}]},
    )

    client = AbelClient(session=FailingSession())
    rows = client.fetch_bars(
        symbols=["ETHUSD"],
        start=None,
        end=None,
        timeframe="1d",
        limit=10,
        fields=None,
        api_key="abel_test",
    )

    assert len(rows) == 1
    assert fallback_calls[0]["url"] == "https://cap.abel.ai/api/market/day_bar"


def test_fetch_bars_retries_on_429(monkeypatch):
    class RateLimitedResponse:
        status_code = 429
        headers = {"Retry-After": "0"}

        def raise_for_status(self):
            raise requests.exceptions.HTTPError("429 Too Many Requests")

    class OkResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"symbol": "ETHUSD"}]}

    class StubSession:
        def __init__(self):
            self.calls = 0

        def post(self, url, json=None, headers=None, timeout=20):
            self.calls += 1
            if self.calls == 1:
                return RateLimitedResponse()
            return OkResponse()

    slept = []
    monkeypatch.setattr("causal_edge.plugins.abel.client.time.sleep", lambda seconds: slept.append(seconds))

    session = StubSession()
    client = AbelClient(session=session)
    rows = client.fetch_bars(
        symbols=["ETHUSD"],
        start=None,
        end=None,
        timeframe="1d",
        limit=10,
        fields=None,
        api_key="abel_test",
    )

    assert len(rows) == 1
    assert session.calls == 2
    assert slept == [1.0]
