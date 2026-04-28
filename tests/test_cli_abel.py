"""CLI tests for Abel-specific commands and failure paths."""

from pathlib import Path

from click.testing import CliRunner

from abel_edge.cli import main


def test_login_json_output(monkeypatch):
    from abel_edge.plugins.abel import auth as auth_module

    def _login_with_oauth(**kwargs):
        kwargs["notify"]("Open this URL to authorize Abel access:\nhttps://example.com/auth")
        kwargs["on_handoff"](
            {
                "status": "awaiting_authorization",
                "auth_url": "https://example.com/auth",
                "env_path": ".env",
                "opened_browser": False,
                "result_url": None,
                "poll_token": "poll-123",
                "poll_interval_seconds": 2.0,
                "timeout_seconds": 300,
            }
        )
        kwargs["on_pending"](
            {
                "status": "waiting_for_authorization",
                "polls": 1,
                "poll_interval_seconds": 2.0,
                "timeout_seconds": 300,
            }
        )
        return {
            "status": "authorized",
            "api_key": "abel_login",
            "env_path": ".env",
            "auth_url": "https://example.com/auth",
            "opened_browser": False,
            "stored": True,
        }

    monkeypatch.setattr(auth_module, "login_with_oauth", _login_with_oauth)
    result = CliRunner().invoke(main, ["login", "--json"])

    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert any('"status": "awaiting_authorization"' in line for line in lines)
    assert any('"status": "waiting_for_authorization"' in line for line in lines)
    assert '"status": "authorized"' in lines[-1]
    assert "abel_login" not in result.output
    assert "Open this URL to authorize Abel access:" in result.output
    assert "https://example.com/auth" in result.output


def test_login_print_token_for_existing_key(monkeypatch):
    from abel_edge.plugins.abel import auth as auth_module

    monkeypatch.setattr(
        auth_module,
        "login_with_oauth",
        lambda **kwargs: {
            "status": "already_configured",
            "api_key": "abel_existing",
            "env_path": ".env",
            "auth_url": None,
            "opened_browser": False,
            "stored": False,
            "source": "env_var",
            "source_path": None,
        },
    )
    result = CliRunner().invoke(main, ["login", "--print-token"])

    assert result.exit_code == 0, result.output
    assert "Reusing existing Abel auth from the current process environment." in result.output
    assert "abel_existing" in result.output


def test_login_reports_shared_auth_reuse(monkeypatch):
    from abel_edge.plugins.abel import auth as auth_module

    monkeypatch.setattr(
        auth_module,
        "login_with_oauth",
        lambda **kwargs: {
            "status": "already_configured",
            "api_key": "abel_existing",
            "env_path": ".env",
            "auth_url": None,
            "opened_browser": False,
            "stored": False,
            "source": "shared_auth_file",
            "source_path": "/tmp/shared/.env.skill",
        },
    )
    result = CliRunner().invoke(main, ["login"])

    assert result.exit_code == 0, result.output
    assert "Reusing existing Abel auth from shared file: /tmp/shared/.env.skill" in result.output


def test_discover_ethusd_parents(monkeypatch, tmp_path):
    class StubClient:
        def discover_parents(self, *, node_id, limit, api_key):
            assert node_id == "ETHUSD"
            assert limit == 20
            assert api_key == "abel_test"
            return [{"node_id": "BTCUSD.price"}, {"node_id": "SOLUSD.price"}]

    from abel_edge.plugins.abel import discover as discover_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(discover_module, "require_api_key", lambda env_path=".env": "abel_test")
    monkeypatch.setattr(discover_module, "AbelClient", StubClient)
    result = CliRunner().invoke(main, ["discover", "ETHUSD", "--limit", "50"])

    assert result.exit_code == 0, result.output
    assert "parents:" in result.output
    assert "ticker: BTCUSD" in result.output
    assert "field: price" in result.output


def test_discover_ethusd_markov_blanket(monkeypatch, tmp_path):
    class StubClient:
        def markov_blanket(self, *, node_id, limit, api_key):
            assert node_id == "ETHUSD"
            assert limit == 12
            return [
                {"node_id": "BTCUSD.price", "roles": ["parent"]},
                {"node_id": "SOLUSD.price", "roles": ["spouse"]},
            ]

    from abel_edge.plugins.abel import discover as discover_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(discover_module, "require_api_key", lambda env_path=".env": "abel_test")
    monkeypatch.setattr(discover_module, "AbelClient", StubClient)
    result = CliRunner().invoke(main, ["discover", "ETHUSD", "--mode", "mb", "--limit", "12"])

    assert result.exit_code == 0, result.output
    assert "markov_blanket:" in result.output
    assert "roles: [parent]" in result.output
    assert "roles: [spouse]" in result.output


def test_discover_json_preserves_target_node_and_field_aware_items(monkeypatch, tmp_path):
    class StubClient:
        def discover_parents(self, *, node_id, limit, api_key):
            assert node_id == "ETHUSD.price"
            return [{"node_id": "BTCUSD.price"}]

        def markov_blanket(self, *, node_id, limit, api_key):
            assert node_id == "ETHUSD.price"
            return [{"node_id": "ETHUSD.volume", "roles": ["sibling"]}]

    from abel_edge.plugins.abel import discover as discover_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(discover_module, "require_api_key", lambda env_path=".env": "abel_test")
    monkeypatch.setattr(discover_module, "AbelClient", StubClient)
    result = CliRunner().invoke(main, ["discover", "ETHUSD.price", "--mode", "all", "--json"])

    assert result.exit_code == 0, result.output
    assert '"target_node": "ETHUSD.price"' in result.output
    assert '"node_id": "BTCUSD.price"' in result.output
    assert '"node_id": "ETHUSD.volume"' in result.output


def test_discover_missing_api_key_fails(monkeypatch, tmp_path):
    from abel_edge.plugins.abel import discover as discover_module
    from abel_edge.plugins.abel.credentials import MissingAbelApiKeyError

    monkeypatch.chdir(tmp_path)

    def _raise(env_path=".env"):
        raise MissingAbelApiKeyError("Abel API key not found.")

    monkeypatch.setattr(discover_module, "require_api_key", _raise)
    result = CliRunner().invoke(main, ["discover", "ETHUSD"])

    assert result.exit_code != 0
    assert "Abel API key not found." in result.output
    assert "ABEL_CAP_BASE_URL" in result.output


def test_discover_uses_causal_abel_skill_auth_file(monkeypatch, tmp_path):
    class StubClient:
        def discover_parents(self, *, node_id, limit, api_key):
            assert node_id == "ETHUSD"
            assert api_key == "abel_skill"
            return [{"node_id": "BTCUSD.price"}]

    from abel_edge.plugins.abel import discover as discover_module

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        monkeypatch.delenv("ABEL_API_KEY", raising=False)
        monkeypatch.delenv("CAP_API_KEY", raising=False)
        monkeypatch.delenv("ABEL_AUTH_ENV_FILE", raising=False)
        Path(".agents/skills/causal-abel").mkdir(parents=True)
        Path(".agents/skills/causal-abel/.env.skill").write_text(
            "ABEL_API_KEY=abel_skill\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(discover_module, "AbelClient", StubClient)
        result = runner.invoke(main, ["discover", "ETHUSD"])

        assert result.exit_code == 0, result.output
        assert "ticker: BTCUSD" in result.output


def test_run_with_abel_source_missing_api_key_fails(monkeypatch, tmp_path):
    from abel_edge.engine import trader as trader_module
    from abel_edge.engine.base import StrategyEngine

    class DemoEngine(StrategyEngine):
        def compute_signals(self):
            bars = self.load_bars(limit=2)
            prices = bars["close"].to_numpy()
            dates = bars["timestamp"]
            return prices * 0.0, dates, prices

        def get_latest_signal(self):
            return {"position": 0.0}

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        monkeypatch.delenv("ABEL_API_KEY", raising=False)
        monkeypatch.delenv("CAP_API_KEY", raising=False)
        monkeypatch.delenv("ABEL_CAP_BASE_URL", raising=False)
        monkeypatch.setattr(trader_module, "_load_engine", lambda engine_path: DemoEngine)
        Path("strategies.yaml").write_text(
            """
settings:
  price_data:
    default_adapter: abel
strategies:
  - id: demo
    name: Demo
    asset: ETHUSD
    color: '#123456'
    engine: strategies.demo.engine
    trade_log: data/demo.csv
""",
            encoding="utf-8",
        )
        Path("data").mkdir()
        result = runner.invoke(main, ["run", "--strategy", "demo"])

        assert result.exit_code != 0
        assert "Abel API key not found." in result.output
        assert "price_data.adapter to 'csv'" in result.output
