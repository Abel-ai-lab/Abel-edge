"""Tests for the optional Abel plugin."""

import os

import pandas as pd

from causal_edge.plugins.abel.client import AbelClient, normalize_public_node_id
from causal_edge.plugins.abel.credentials import (
    MissingAbelApiKeyError,
    require_api_key,
    resolve_api_key,
    resolve_cap_base_url,
)
from examples.causal_demo.engine import CausalDemoEngine, resolve_price_column


def test_normalize_public_node_id_ethusd():
    assert normalize_public_node_id("ETHUSD") == "ETHUSD.price"
    assert normalize_public_node_id("ETH") == "ETHUSD.price"
    assert normalize_public_node_id("ETHUSD_close") == "ETHUSD.price"


def test_resolve_api_key_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ABEL_API_KEY", "Bearer abel_env")
    (tmp_path / ".env").write_text("ABEL_API_KEY=abel_file\n", encoding="utf-8")

    api_key = resolve_api_key(env_path=tmp_path / ".env")

    assert api_key == "abel_env"


def test_resolve_api_key_reads_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    (tmp_path / ".env").write_text("CAP_API_KEY=Bearer cap_file\n", encoding="utf-8")

    api_key = resolve_api_key(env_path=tmp_path / ".env")

    assert api_key == "cap_file"


def test_require_api_key_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)

    try:
        require_api_key(env_path=tmp_path / ".env")
    except MissingAbelApiKeyError as e:
        assert "ABEL_API_KEY" in str(e)
    else:
        raise AssertionError("Expected MissingAbelApiKeyError")


def test_resolve_cap_base_url_uses_public_default(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_CAP_BASE_URL", raising=False)

    base_url = resolve_cap_base_url(env_path=tmp_path / ".env")

    assert base_url == "https://cap.abel.ai/api"


def test_resolve_cap_base_url_reads_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ABEL_CAP_BASE_URL", "https://cap.custom.abel.ai/api/")

    base_url = resolve_cap_base_url(env_path=tmp_path / ".env")

    assert base_url == "https://cap.custom.abel.ai/api"


def test_resolve_cap_base_url_reads_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_CAP_BASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "ABEL_CAP_BASE_URL=https://cap.file.abel.ai/api/\n", encoding="utf-8"
    )

    base_url = resolve_cap_base_url(env_path=tmp_path / ".env")

    assert base_url == "https://cap.file.abel.ai/api"


def test_resolve_api_key_does_not_mutate_process_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    (tmp_path / ".env").write_text("ABEL_API_KEY=abel_file\n", encoding="utf-8")

    api_key = resolve_api_key(env_path=tmp_path / ".env")

    assert api_key == "abel_file"
    assert os.getenv("ABEL_API_KEY") is None


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
