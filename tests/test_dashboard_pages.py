from pathlib import Path

from click.testing import CliRunner

from causal_edge.cli import main
from tests.dashboard_test_utils import DEMO_ENGINE, write_demo_project


def test_dashboard_empty():
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("strategies.yaml").write_text("settings: {}\nstrategies: []\n", encoding="utf-8")
        result = runner.invoke(main, ["dashboard"])
        assert result.exit_code == 0
        assert Path("dashboard.html").exists()


def test_dashboard_renders_paper_trading_section(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_demo_project(
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


def test_dashboard_keeps_backtest_summary_when_paper_data_exists(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_demo_project(
            paper_log=True,
            backtest_csv=(
                "date,asset_return,pnl,position,cum_return,source\n"
                "2024-01-01,0.00,0.00,0.00,0.00,backfill\n"
                + "".join(
                    f"2024-01-{day:02d},0.10,0.10,1.00,0.00,backfill\n" for day in range(2, 31)
                )
            ),
            paper_csv=(
                "date,asset_return,pnl,position,source,close,next_position\n"
                "2024-01-31,0.01,0.01,0.50,live,101.0,0.00\n"
            ),
        )
        result = runner.invoke(main, ["dashboard", "--output", "dashboard.html"])
        assert result.exit_code == 0, result.output
        html = Path("dashboard.html").read_text(encoding="utf-8")
        assert "+1486.3%" in html
        assert "+1.0%" in html
        assert html.index("+1486.3%") < html.index("+1.0%")


def test_dashboard_hides_legacy_paper_price_chart_without_asset_returns(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_demo_project(
            backtest_csv=(
                "date,pnl,position,cum_pnl,source\n"
                "2024-01-01,0.01,0.50,0.01,backfill\n"
                "2024-01-02,0.00,0.00,0.01,live\n"
                "2024-01-03,0.01,0.25,0.02,live\n"
            ),
        )
        result = runner.invoke(main, ["dashboard", "--output", "dashboard.html"])
        assert result.exit_code == 0, result.output
        html = Path("dashboard.html").read_text(encoding="utf-8")
        assert "Backtest vs ETHUSD" not in html
        assert "Legacy live rows do not include asset price returns" in html
        assert "tracking-asset-demo_signal" not in html


def test_dashboard_uses_price_data_overlay_when_trade_log_lacks_asset_returns(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("strategies.yaml").write_text(
            (
                "settings:\n  price_data:\n    default_source: csv\nstrategies:\n"
                '  - id: demo_signal\n    name: "Demo Signal"\n    asset: ETHUSD\n'
                '    color: "#2563EB"\n    engine: strategies.demo_signal.engine\n'
                "    trade_log: data/trade_log_demo_signal.csv\n    price_data:\n"
                '      source: csv\n      path: data/prices.csv\n    thesis: "Signal thesis"\n'
            ),
            encoding="utf-8",
        )
        Path("strategies").mkdir()
        Path("strategies/__init__.py").write_text("", encoding="utf-8")
        Path("strategies/demo_signal").mkdir(parents=True)
        Path("strategies/demo_signal/__init__.py").write_text("", encoding="utf-8")
        Path("strategies/demo_signal/engine.py").write_text(DEMO_ENGINE, encoding="utf-8")
        Path("data").mkdir()
        Path("data/trade_log_demo_signal.csv").write_text(
            "date,pnl,position,cum_pnl,source\n2024-01-01,0.01,0.50,0.01,backfill\n2024-01-02,0.02,0.50,0.03,backfill\n",
            encoding="utf-8",
        )
        Path("data/prices.csv").write_text(
            "timestamp,close\n2024-01-01,100\n2024-01-02,110\n", encoding="utf-8"
        )
        result = runner.invoke(main, ["dashboard", "--output", "dashboard.html"])
        assert result.exit_code == 0, result.output
        html = Path("dashboard.html").read_text(encoding="utf-8")
        assert "Backtest vs ETHUSD" in html


def test_signal_demo_renders_single_signal_page(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_demo_project(
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
        assert "Strategy Equity" in html or "Strategy vs Hold" in html
        assert "signal-track-ethusd.html" in html
        assert "Watchlist" in html
        assert "showSectionTab('strategy')" in html
        assert "Strategy" in html
        assert "Abel Causal Graph" in html
        assert "Live Rows" in html


def test_signal_demo_hides_hold_language_without_backtest_asset_returns(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_demo_project(
            backtest_csv=(
                "date,pnl,position,cum_pnl,source\n"
                "2024-01-01,0.00,0.00,0.00,backfill\n"
                "2024-01-02,0.02,0.50,0.02,backfill\n"
            ),
        )
        result = runner.invoke(
            main, ["signal-demo", "--strategy", "demo_signal", "--output", "signal-demo.html"]
        )
        assert result.exit_code == 0, result.output
        html = Path("signal-demo.html").read_text(encoding="utf-8")
        assert "Strategy Equity" in html
        assert "Strategy vs Hold" not in html


def test_signal_demo_surfaces_live_tracking_status(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_demo_project(
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
        write_demo_project(
            backtest_csv=(
                "date,asset_return,pnl,position,cum_return,source\n2024-01-01,0.00,0.00,0.00,0.00,backfill\n"
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
        write_demo_project(
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


def test_tracking_hides_legacy_price_chart_without_asset_returns(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        write_demo_project(
            backtest_csv=(
                "date,pnl,position,cum_pnl,source\n"
                "2024-01-01,0.01,0.50,0.01,backfill\n"
                "2024-01-02,0.00,0.00,0.01,live\n"
                "2024-01-03,0.01,0.25,0.02,live\n"
            ),
        )
        result = runner.invoke(
            main, ["tracking", "--strategy", "demo_signal", "--output", "tracking.html"]
        )
        assert result.exit_code == 0, result.output
        html = Path("tracking.html").read_text(encoding="utf-8")
        assert "Legacy live rows do not include asset price returns" in html
        assert (
            "Legacy logs do not include backtest asset returns for a launch-context chart." in html
        )
        assert "tracking-preview" not in html
        assert "tracking-asset" not in html
        assert "Position Since Tracking Began" in html
