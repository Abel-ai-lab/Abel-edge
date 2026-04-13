"""CLI entry point tests."""

from pathlib import Path

from click.testing import CliRunner

from causal_edge.cli import main


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


def test_dashboard_empty():
    """Dashboard with no strategies should generate HTML without error."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create minimal strategies.yaml
        Path("strategies.yaml").write_text("settings: {}\nstrategies: []\n", encoding="utf-8")
        result = runner.invoke(main, ["dashboard"])
        assert result.exit_code == 0
        assert Path("dashboard.html").exists()


def test_dashboard_strategy_renders_single_signal_page(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("strategies.yaml").write_text(
            """
settings: {}
strategies:
  - id: demo_signal
    name: "Demo Signal"
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.demo_signal.engine
    trade_log: data/trade_log_demo_signal.csv
    thesis: "Signal thesis"
    cta_text: "Start tracking this signal"
""".strip()
            + "\n",
            encoding="utf-8",
        )
        Path("strategies").mkdir()
        Path("strategies/__init__.py").write_text("", encoding="utf-8")
        Path("strategies/demo_signal").mkdir(parents=True)
        Path("strategies/demo_signal/__init__.py").write_text("", encoding="utf-8")
        Path("strategies/demo_signal/engine.py").write_text(
            "from causal_edge.engine.base import StrategyEngine\n"
            "class DemoSignalEngine(StrategyEngine):\n"
            "    def compute_signals(self):\n"
            "        raise NotImplementedError\n"
            "    def get_latest_signal(self):\n"
            "        return {'position': 0.0}\n",
            encoding="utf-8",
        )
        Path("data").mkdir()
        Path("data/trade_log_demo_signal.csv").write_text(
            "date,asset_return,pnl,position,cum_return,source\n"
            "2024-01-01,0.00,0.00,0.00,0.00,backfill\n"
            "2024-01-02,0.02,0.01,0.50,0.01,backfill\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            main, ["dashboard", "--strategy", "demo_signal", "--output", "signal-demo.html"]
        )

        assert result.exit_code == 0, result.output
        html = Path("signal-demo.html").read_text(encoding="utf-8")
        assert "ETHUSD" in html
        assert "Current Signal" in html
        assert "Start tracking this signal" in html
        assert "Backtest vs Ticker Trend" in html
        assert "signal-track-ethusd.html" in html


def test_dashboard_strategy_surfaces_live_tracking_status(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("strategies.yaml").write_text(
            """
settings: {}
strategies:
  - id: demo_signal
    name: "Demo Signal"
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.demo_signal.engine
    trade_log: data/trade_log_demo_signal.csv
    thesis: "Signal thesis"
    cta_text: "Start tracking this signal"
""".strip()
            + "\n",
            encoding="utf-8",
        )
        Path("strategies").mkdir()
        Path("strategies/__init__.py").write_text("", encoding="utf-8")
        Path("strategies/demo_signal").mkdir(parents=True)
        Path("strategies/demo_signal/__init__.py").write_text("", encoding="utf-8")
        Path("strategies/demo_signal/engine.py").write_text(
            "from causal_edge.engine.base import StrategyEngine\n"
            "class DemoSignalEngine(StrategyEngine):\n"
            "    def compute_signals(self):\n"
            "        raise NotImplementedError\n"
            "    def get_latest_signal(self):\n"
            "        return {'position': 0.0}\n",
            encoding="utf-8",
        )
        Path("data").mkdir()
        Path("data/trade_log_demo_signal.csv").write_text(
            "date,asset_return,pnl,position,cum_return,source,close,next_position\n"
            "2024-01-01,0.00,0.00,0.00,0.00,backfill,,\n"
            "2024-01-02,0.02,0.01,0.50,0.01,backfill,,\n"
            "2024-01-03,0.03,0.01,0.50,0.02,live,101.0,1.00\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            main, ["dashboard", "--strategy", "demo_signal", "--output", "signal-demo.html"]
        )

        assert result.exit_code == 0, result.output
        html = Path("signal-demo.html").read_text(encoding="utf-8")
        assert "Tracking started" in html
        assert "Continue tracking" in html
        assert "Tracking Snapshot" in html


def test_dashboard_strategy_missing_id_fails():
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("strategies.yaml").write_text(
            "settings: {}\nstrategies:\n  - id: only_one\n    name: 'Only One'\n    asset: ETHUSD\n    color: '#2563EB'\n    engine: strategies.only_one.engine\n    trade_log: data/only.csv\n",
            encoding="utf-8",
        )
        result = runner.invoke(main, ["dashboard", "--strategy", "missing"])
        assert result.exit_code != 0
        assert "Strategy 'missing' not found" in result.output


def test_tracking_strategy_renders_empty_state(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("strategies.yaml").write_text(
            """
settings: {}
strategies:
  - id: demo_signal
    name: "Demo Signal"
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.demo_signal.engine
    trade_log: data/trade_log_demo_signal.csv
    thesis: "Signal thesis"
""".strip()
            + "\n",
            encoding="utf-8",
        )
        Path("strategies").mkdir()
        Path("strategies/__init__.py").write_text("", encoding="utf-8")
        Path("strategies/demo_signal").mkdir(parents=True)
        Path("strategies/demo_signal/__init__.py").write_text("", encoding="utf-8")
        Path("strategies/demo_signal/engine.py").write_text(
            "from causal_edge.engine.base import StrategyEngine\n"
            "class DemoSignalEngine(StrategyEngine):\n"
            "    def compute_signals(self):\n"
            "        raise NotImplementedError\n"
            "    def get_latest_signal(self):\n"
            "        return {'position': 0.0}\n",
            encoding="utf-8",
        )
        Path("data").mkdir()
        Path("data/trade_log_demo_signal.csv").write_text(
            "date,asset_return,pnl,position,cum_return,source\n"
            "2024-01-01,0.00,0.00,0.00,0.00,backfill\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            main, ["tracking", "--strategy", "demo_signal", "--output", "tracking.html"]
        )

        assert result.exit_code == 0, result.output
        html = Path("tracking.html").read_text(encoding="utf-8")
        assert "Tracking View" in html
        assert "No live tracking data yet" in html
        assert "Tracking Launch Context" in html


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
        def ensure_api_key(self, *, env_path=".env"):
            assert env_path == ".env"
            return "abel_test"

        def discover_parents(self, *, node_id, limit, api_key):
            assert node_id == "ETHUSD"
            assert limit == 20
            assert api_key == "abel_test"
            return [{"node_id": "BTCUSD.price"}, {"node_id": "SOLUSD.price"}]

    from causal_edge.plugins.abel import discover as discover_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(discover_module, "AbelClient", StubClient)
    result = CliRunner().invoke(main, ["discover", "ETHUSD", "--limit", "50"])
    assert result.exit_code == 0, result.output
    assert "parents:" in result.output
    assert "ticker: BTCUSD" in result.output
    assert "field: price" in result.output


def test_discover_ethusd_markov_blanket(monkeypatch, tmp_path):
    class StubClient:
        def ensure_api_key(self, *, env_path=".env"):
            return "abel_test"

        def markov_blanket(self, *, node_id, limit, api_key):
            assert node_id == "ETHUSD"
            assert limit == 12
            return [
                {"node_id": "BTCUSD.price", "roles": ["parent"]},
                {"node_id": "SOLUSD.price", "roles": ["spouse"]},
            ]

    from causal_edge.plugins.abel import discover as discover_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(discover_module, "AbelClient", StubClient)
    result = CliRunner().invoke(main, ["discover", "ETHUSD", "--mode", "mb", "--limit", "12"])
    assert result.exit_code == 0, result.output
    assert "markov_blanket:" in result.output
    assert "roles: [parent]" in result.output
    assert "roles: [spouse]" in result.output
