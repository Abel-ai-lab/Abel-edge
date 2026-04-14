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


def test_status_prefers_local_overlay_config():
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("strategies.yaml").write_text("settings: {}\nstrategies: []\n", encoding="utf-8")
        Path("strategies.local.yaml").write_text(
            "\n".join(
                [
                    "settings: {}",
                    "strategies:",
                    "  - id: local_demo",
                    '    name: "Local Demo"',
                    "    asset: ETHUSD",
                    '    color: "#2563EB"',
                    "    engine: strategies.local_demo.engine",
                    "    trade_log: data/local_demo.csv",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "Strategies: 1" in result.output
        assert "Local Demo" in result.output


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
