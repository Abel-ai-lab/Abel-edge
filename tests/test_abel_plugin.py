"""Tests for the optional Abel plugin."""

import builtins
import os
from pathlib import Path

import pandas as pd

from causal_edge.plugins.abel.client import AbelClient, normalize_public_node_id, persist_api_key
from examples.causal_demo.engine import CausalDemoEngine, resolve_price_column


def test_normalize_public_node_id_ethusd():
    assert normalize_public_node_id("ETHUSD") == "ETHUSD.price"
    assert normalize_public_node_id("ETH") == "ETHUSD.price"
    assert normalize_public_node_id("ETHUSD_close") == "ETHUSD.price"


def test_persist_api_key(tmp_path):
    env_path = tmp_path / ".env"
    persist_api_key(env_path, "abel_123")
    assert "ABEL_API_KEY=abel_123" in env_path.read_text(encoding="utf-8")


def test_ensure_api_key_polls_until_authorized(tmp_path, monkeypatch):
    class StubSession:
        def __init__(self):
            self.poll_count = 0

        def get(self, url, timeout=20):
            if url.endswith("/authorize/agent"):
                return StubResponse(
                    {
                        "data": {
                            "authUrl": "https://example.com/auth",
                            "pollToken": "poll-123",
                        }
                    }
                )
            self.poll_count += 1
            if self.poll_count == 1:
                return StubResponse({"data": {"status": "pending"}})
            return StubResponse({"data": {"status": "authorized", "apiKey": "abel_key"}})

    class StubResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    opened_urls = []
    slept = []
    monkeypatch.setattr("webbrowser.open", opened_urls.append)
    monkeypatch.setattr("time.sleep", slept.append)
    client = AbelClient(session=StubSession())

    api_key = client.ensure_api_key(env_path=tmp_path / ".env", poll_interval=0.01)

    assert api_key == "abel_key"
    assert os.environ["ABEL_API_KEY"] == "abel_key"
    assert opened_urls == ["https://example.com/auth"]
    assert slept == [0.01]
    assert "ABEL_API_KEY=abel_key" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_ensure_api_key_waits_for_enter_in_tty(tmp_path, monkeypatch):
    class StubSession:
        def get(self, url, timeout=20):
            if url.endswith("/authorize/agent"):
                return StubResponse(
                    {
                        "data": {
                            "authUrl": "https://example.com/auth",
                            "pollToken": "poll-123",
                        }
                    }
                )
            return StubResponse({"data": {"status": "authorized", "apiKey": "abel_key_tty"}})

    class StubResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class StubStdin:
        def isatty(self):
            return True

    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    monkeypatch.setattr("sys.stdin", StubStdin())
    monkeypatch.setattr("webbrowser.open", lambda url: True)
    prompted = []
    monkeypatch.setattr(builtins, "input", lambda: prompted.append(True) or "")
    client = AbelClient(session=StubSession())

    api_key = client.ensure_api_key(env_path=tmp_path / ".env", poll_interval=0.01)

    assert api_key == "abel_key_tty"
    assert prompted == [True]


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


def test_fetch_bars_uses_market_prod_base_url():
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


def test_discover_uses_cap_prod_base_url():
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
