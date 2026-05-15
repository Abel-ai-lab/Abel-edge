"""Tests for Abel client helpers and example integration."""

import json
import requests
import pandas as pd

from abel_edge.plugins.abel.client import AbelClient, normalize_public_node_id
from examples.causal_demo.engine import CausalDemoEngine, GRAPH_PATH, resolve_price_column

CUSTOM_CAP_URL = "https://cap.file.abel.ai/api"


def _write_abel_env(path):
    path.write_text(
        f"ABEL_API_KEY=abel_test\nABEL_CAP_BASE_URL={CUSTOM_CAP_URL}/\n",
        encoding="utf-8",
    )


def test_normalize_public_node_id_ethusd():
    assert normalize_public_node_id("ETHUSD") == "ETHUSD.price"
    assert normalize_public_node_id("ETH") == "ETHUSD.price"
    assert normalize_public_node_id("ETHUSD_close") == "ETHUSD.price"


def test_resolve_price_column_prefers_close():
    df = pd.DataFrame({"close": [1.0], "price": [2.0], "volume": [3.0]})
    assert resolve_price_column(df, "price") == "close"
    assert resolve_price_column(df, "volume") == "volume"


def _build_causal_demo_context(tmp_path):
    with open(GRAPH_PATH, encoding="utf-8") as handle:
        graph = json.load(handle)

    dates = pd.bdate_range("2025-01-01", periods=40, tz="UTC")
    target = pd.DataFrame(
        {
            "timestamp": dates.astype(str),
            "close": [10.0 + 0.12 * i + ((i % 5) - 2) * 0.08 for i in range(len(dates))],
        }
    )
    primary_path = tmp_path / "tonusd.csv"
    target.to_csv(primary_path, index=False)

    feeds = {
        "primary": {
            "name": "primary",
            "kind": "bars",
            "adapter": "csv",
            "path": str(primary_path),
            "symbol": "TONUSD",
            "timeframe": "1d",
            "profile": "daily",
        }
    }
    for idx, item in enumerate(graph.get("parents", [])):
        ticker = item["ticker"] if isinstance(item, dict) else str(item)
        feed_path = tmp_path / f"{ticker.lower()}.csv"
        frame = pd.DataFrame(
            {
                "timestamp": dates.astype(str),
                "close": [
                    20.0 + idx + 0.07 * i + ((i + idx) % 4 - 1.5) * 0.05
                    for i in range(len(dates))
                ],
            }
        )
        frame.to_csv(feed_path, index=False)
        feeds[ticker] = {
            "name": ticker,
            "kind": "bars",
            "adapter": "csv",
            "path": str(feed_path),
            "symbol": ticker,
            "timeframe": "1d",
            "profile": "daily",
        }

    return {
        "_data_contract": {"profile": "daily"},
        "_runtime_profile": {
            "profile": "daily",
            "target": "TONUSD",
            "decision_event": "bar_close",
            "execution_delay_bars": 1,
            "return_basis": "close_to_close",
        },
        "_execution_constraints": {"position_bounds": [0.0, 1.0], "long_only": True},
        "_feeds": feeds,
    }


def test_causal_demo_runs_with_default_parent_metadata(tmp_path):
    engine = CausalDemoEngine(context=_build_causal_demo_context(tmp_path))
    compiled = engine.compute_runtime_output()
    assert len(compiled.positions) == len(compiled.decision_index) == len(compiled.close_prices)
    assert compiled.next_position[-1] >= 0.0


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


def test_client_uses_env_path_for_cap_base_url(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_CAP_BASE_URL", raising=False)
    env_path = tmp_path / "custom.env"
    _write_abel_env(env_path)

    client = AbelClient(env_path=env_path)

    assert client.cap_base_url == CUSTOM_CAP_URL


def test_fetch_bars_wrapper_uses_env_path_for_cap_base_url(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    monkeypatch.delenv("ABEL_CAP_BASE_URL", raising=False)
    env_path = tmp_path / "custom.env"
    _write_abel_env(env_path)
    observed = {}

    def fake_fetch_bars(self, **kwargs):
        observed["cap_base_url"] = self.cap_base_url
        return []

    monkeypatch.setattr(AbelClient, "fetch_bars", fake_fetch_bars)
    from abel_edge.plugins.abel import prices as prices_module

    prices_module.fetch_bars(symbols=["ETHUSD"], config={"env_path": str(env_path)})

    assert observed["cap_base_url"] == CUSTOM_CAP_URL


def test_discover_wrapper_uses_env_path_for_cap_base_url(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    monkeypatch.delenv("ABEL_CAP_BASE_URL", raising=False)
    env_path = tmp_path / "custom.env"
    _write_abel_env(env_path)
    observed = {}

    def fake_discover_parents(self, *, node_id, limit, api_key):
        observed["cap_base_url"] = self.cap_base_url
        return [{"node_id": "BTCUSD.price"}]

    monkeypatch.setattr(AbelClient, "discover_parents", fake_discover_parents)
    from abel_edge.plugins.abel import discover as discover_module

    discover_module.discover_graph_payload("ETHUSD", mode="parents", env_path=str(env_path))

    assert observed["cap_base_url"] == CUSTOM_CAP_URL


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

    monkeypatch.setattr("abel_edge.plugins.abel.client.sys.platform", "win32")
    monkeypatch.setattr("abel_edge.plugins.abel.client.shutil.which", lambda name: "C:/Windows/System32/curl.exe")
    monkeypatch.setattr(
        "abel_edge.plugins.abel.client._post_with_curl",
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
    monkeypatch.setattr("abel_edge.plugins.abel.client.time.sleep", lambda seconds: slept.append(seconds))

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
