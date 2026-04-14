"""CLI entry point tests."""

from pathlib import Path

from click.testing import CliRunner

from causal_edge.cli import main


DEMO_ENGINE = (
    "from causal_edge.engine.base import StrategyEngine\n"
    "class DemoSignalEngine(StrategyEngine):\n"
    "    def compute_signals(self):\n"
    "        raise NotImplementedError\n"
    "    def get_latest_signal(self):\n"
    "        return {'position': 0.0}\n"
)


def _write_demo_project(*, paper_log=False, cta=False, backtest_csv=None, paper_csv=None):
    config = [
        "settings: {}",
        "strategies:",
        "  - id: demo_signal",
        '    name: "Demo Signal"',
        "    asset: ETHUSD",
        '    color: "#2563EB"',
        "    engine: strategies.demo_signal.engine",
        "    trade_log: data/trade_log_demo_signal.csv",
    ]
    if paper_log:
        config.append("    paper_log: data/paper_log_demo_signal.csv")
    config.append('    thesis: "Signal thesis"')
    if cta:
        config.append('    cta_text: "Start tracking this signal"')

    Path("strategies.yaml").write_text("\n".join(config) + "\n", encoding="utf-8")
    Path("strategies").mkdir()
    Path("strategies/__init__.py").write_text("", encoding="utf-8")
    Path("strategies/demo_signal").mkdir(parents=True)
    Path("strategies/demo_signal/__init__.py").write_text("", encoding="utf-8")
    Path("strategies/demo_signal/engine.py").write_text(DEMO_ENGINE, encoding="utf-8")
    Path("data").mkdir()
    if backtest_csv is not None:
        Path("data/trade_log_demo_signal.csv").write_text(backtest_csv, encoding="utf-8")
    if paper_csv is not None:
        Path("data/paper_log_demo_signal.csv").write_text(paper_csv, encoding="utf-8")


def test_help():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "causal-edge" in result.output


def test_version():
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_status_empty():
    """Status with empty strategies.yaml should show 0 strategies."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("strategies.yaml").write_text("settings: {}\nstrategies: []\n", encoding="utf-8")
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "Strategies: 0" in result.output


def test_init_creates_project(tmp_path):
    """init should create a project directory with expected files."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["init", "myproject"])
        assert result.exit_code == 0, result.output
        root = Path("myproject")
        assert root.is_dir()
        assert (root / "strategies.yaml").exists()
        assert (root / "strategies" / "sma_crossover" / "engine.py").exists()
        assert (root / "strategies" / "sma_crossover" / "__init__.py").exists()
        assert (root / "data").is_dir()
        assert (root / ".env.example").exists()
        assert (root / "CLAUDE.md").exists()
        assert (root / "AGENTS.md").exists()


def test_init_fails_if_dir_exists(tmp_path):
    """init should fail with a clear error if the directory already exists."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(main, ["init", "myproject"])
        result = runner.invoke(main, ["init", "myproject"])
        assert result.exit_code != 0
        assert "already exists" in result.output


def test_run_empty():
    """Run with no strategies should print message, not crash."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("strategies.yaml").write_text("settings: {}\nstrategies: []\n", encoding="utf-8")
        result = runner.invoke(main, ["run"])
        assert result.exit_code == 0
        assert "No strategies" in result.output


def test_login_json_output(monkeypatch):
    from causal_edge.plugins.abel import auth as auth_module

    def _login_with_oauth(**kwargs):
        kwargs["on_handoff"](
            {
                "status": "awaiting_authorization",
                "auth_url": "https://example.com/auth",
                "env_path": ".env",
                "opened_browser": False,
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

    monkeypatch.setattr(
        auth_module,
        "login_with_oauth",
        _login_with_oauth,
    )

    result = CliRunner().invoke(main, ["login", "--json"])

    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert '"status": "awaiting_authorization"' in lines[0]
    assert '"status": "authorized"' in lines[-1]
    assert "abel_login" not in result.output


def test_login_print_token_for_existing_key(monkeypatch):
    from causal_edge.plugins.abel import auth as auth_module

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
        },
    )

    result = CliRunner().invoke(main, ["login", "--print-token"])

    assert result.exit_code == 0, result.output
    assert "Abel API key already configured." in result.output
    assert "abel_existing" in result.output


def test_dashboard_empty():
    """Dashboard with no strategies should generate HTML without error."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create minimal strategies.yaml
        Path("strategies.yaml").write_text("settings: {}\nstrategies: []\n", encoding="utf-8")
        result = runner.invoke(main, ["dashboard"])
        assert result.exit_code == 0
        assert Path("dashboard.html").exists()


def test_dashboard_renders_paper_trading_section(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _write_demo_project(
            paper_log=True,
            backtest_csv=(
                "date,asset_return,pnl,position,cum_return,source\n"
                "2024-01-01,0.00,0.00,0.00,0.00,backfill\n"
                "2024-01-02,0.02,0.01,0.50,0.01,backfill\n"
            ),
            paper_csv=(
                "date,asset_return,pnl,position,source,close,next_position\n"
                "2024-01-03,0.03,0.01,0.50,live,101.0,1.00\n"
            ),
        )

        result = runner.invoke(main, ["dashboard", "--output", "dashboard.html"])

        assert result.exit_code == 0, result.output
        html = Path("dashboard.html").read_text(encoding="utf-8")
        assert "Paper Trading" in html
        assert "Tracking started" in html
        assert "Live through: 2024-01-03" in html
        assert "Live Rows" in html
        assert "showSectionTab('demo_signal', 'paper'" in html


def test_signal_demo_renders_single_signal_page(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _write_demo_project(
            paper_log=True,
            cta=True,
            backtest_csv=(
                "date,asset_return,pnl,position,cum_return,source\n"
                "2024-01-01,0.00,0.00,0.00,0.00,backfill\n"
                "2024-01-02,0.02,0.01,0.50,0.01,backfill\n"
            ),
        )

        result = runner.invoke(
            main, ["signal-demo", "--strategy", "demo_signal", "--output", "signal-demo.html"]
        )

        assert result.exit_code == 0, result.output
        html = Path("signal-demo.html").read_text(encoding="utf-8")
        assert "ETHUSD" in html
        assert "Live Signal: Track Lightly" in html
        assert "Paper Trading" in html
        assert "Strategy vs Hold" in html
        assert "signal-track-ethusd.html" in html
        assert "Watchlist" in html
        assert "showSectionTab('strategy')" in html
        assert "Strategy" in html
        assert "Abel Causal Graph" in html
        assert "Live Rows" in html


def test_signal_demo_surfaces_live_tracking_status(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _write_demo_project(
            paper_log=True,
            cta=True,
            backtest_csv=(
                "date,asset_return,pnl,position,cum_return,source\n"
                "2024-01-01,0.00,0.00,0.00,0.00,backfill\n"
                "2024-01-02,0.02,0.01,0.50,0.01,backfill\n"
            ),
            paper_csv=(
                "date,asset_return,pnl,position,source,close,next_position\n"
                "2024-01-03,0.03,0.01,0.50,live,101.0,1.00\n"
            ),
        )

        result = runner.invoke(
            main, ["signal-demo", "--strategy", "demo_signal", "--output", "signal-demo.html"]
        )

        assert result.exit_code == 0, result.output
        html = Path("signal-demo.html").read_text(encoding="utf-8")
        assert "Paper Trading" in html
        assert "Live Signal: Hold" in html or "Live Signal: Observe" in html
        assert "Strategy vs Hold" in html
        assert "Live through" in html
        assert "Abel Causal Graph" in html


def test_signal_demo_missing_id_fails():
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("strategies.yaml").write_text(
            "settings: {}\nstrategies:\n  - id: only_one\n    name: 'Only One'\n    asset: ETHUSD\n    color: '#2563EB'\n    engine: strategies.only_one.engine\n    trade_log: data/only.csv\n",
            encoding="utf-8",
        )
        result = runner.invoke(main, ["signal-demo", "--strategy", "missing"])
        assert result.exit_code != 0
        assert "Strategy 'missing' not found" in result.output


def test_dashboard_rejects_strategy_option():
    runner = CliRunner()
    result = runner.invoke(main, ["dashboard", "--strategy", "demo_signal"])
    assert result.exit_code != 0
    assert "No such option: --strategy" in result.output


def test_tracking_strategy_renders_empty_state(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _write_demo_project(
            backtest_csv=(
                "date,asset_return,pnl,position,cum_return,source\n"
                "2024-01-01,0.00,0.00,0.00,0.00,backfill\n"
            ),
        )

        result = runner.invoke(
            main, ["tracking", "--strategy", "demo_signal", "--output", "tracking.html"]
        )

        assert result.exit_code == 0, result.output
        html = Path("tracking.html").read_text(encoding="utf-8")
        assert "Tracking View" in html
        assert "No live tracking data yet" in html
        assert "Tracking Launch Context" in html


def test_tracking_strategy_renders_live_rows(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _write_demo_project(
            paper_log=True,
            backtest_csv=(
                "date,asset_return,pnl,position,cum_return,source\n"
                "2024-01-01,0.00,0.00,0.00,0.00,backfill\n"
                "2024-01-02,0.02,0.01,0.50,0.01,backfill\n"
            ),
            paper_csv=(
                "date,asset_return,pnl,position,source,close,next_position\n"
                "2024-01-03,0.03,0.01,0.50,live,101.0,1.00\n"
            ),
        )

        result = runner.invoke(
            main, ["tracking", "--strategy", "demo_signal", "--output", "tracking.html"]
        )

        assert result.exit_code == 0, result.output
        html = Path("tracking.html").read_text(encoding="utf-8")
        assert "Live Rows" in html
        assert "2024-01-03" in html
        assert "101.00" in html


def test_validate_empty():
    """Validate with no strategies should print message, not crash."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("strategies.yaml").write_text("settings: {}\nstrategies: []\n", encoding="utf-8")
        result = runner.invoke(main, ["validate"])
        assert result.exit_code == 0
        assert "No strategies" in result.output


def test_validate_csv_missing_file():
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--csv", "missing.csv"])
    assert result.exit_code != 0
    assert "CSV not found: missing.csv" in result.output


def test_discover_ethusd_parents(monkeypatch, tmp_path):
    class StubClient:
        def discover_parents(self, *, node_id, limit, api_key):
            assert node_id == "ETHUSD"
            assert limit == 20
            assert api_key == "abel_test"
            return [{"node_id": "BTCUSD.price"}, {"node_id": "SOLUSD.price"}]

    from causal_edge.plugins.abel import discover as discover_module

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

    from causal_edge.plugins.abel import discover as discover_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(discover_module, "require_api_key", lambda env_path=".env": "abel_test")
    monkeypatch.setattr(discover_module, "AbelClient", StubClient)
    result = CliRunner().invoke(main, ["discover", "ETHUSD", "--mode", "mb", "--limit", "12"])
    assert result.exit_code == 0, result.output
    assert "markov_blanket:" in result.output
    assert "roles: [parent]" in result.output
    assert "roles: [spouse]" in result.output


def test_discover_missing_api_key_fails(monkeypatch, tmp_path):
    from causal_edge.plugins.abel import discover as discover_module
    from causal_edge.plugins.abel.credentials import MissingAbelApiKeyError

    monkeypatch.chdir(tmp_path)

    def _raise(env_path=".env"):
        raise MissingAbelApiKeyError("Abel API key not found.")

    monkeypatch.setattr(discover_module, "require_api_key", _raise)
    result = CliRunner().invoke(main, ["discover", "ETHUSD"])

    assert result.exit_code != 0
    assert "Abel API key not found." in result.output
    assert "ABEL_CAP_BASE_URL" in result.output


def test_run_with_abel_source_missing_api_key_fails(monkeypatch, tmp_path):
    from causal_edge.engine.base import StrategyEngine
    from causal_edge.engine import trader as trader_module

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
    default_source: abel
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
        assert "price_data.source to 'csv'" in result.output
