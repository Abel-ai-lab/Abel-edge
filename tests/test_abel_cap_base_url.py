"""Tests for Abel CAP base URL resolution across wrappers."""

from abel_edge.plugins.abel.client import AbelClient

CUSTOM_CAP_URL = "https://cap.file.abel.ai/api"


def _write_abel_env(path):
    path.write_text(
        f"ABEL_API_KEY=abel_test\nABEL_CAP_BASE_URL={CUSTOM_CAP_URL}/\n",
        encoding="utf-8",
    )


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
