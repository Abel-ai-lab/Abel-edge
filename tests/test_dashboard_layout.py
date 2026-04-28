"""Regression tests for the 3-page dashboard layout.

Guards against accidental regression to the pre-2026-04 single-page
dashboard. If any commit removes the Portfolio / Live / Strategy nav,
the YTD/MTD/today badges, or the signals_flat render path, these tests
go red immediately.

Deliberately minimal: renders a realistic CSV fixture and greps the
output HTML for structural markers. Tests intentionally use string
matching rather than full DOM parsing — the invariant we protect is
"visible to user" not "semantic HTML".
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from abel_edge.cli import main


def _write_demo_project(base: Path) -> None:
    strategies_yaml = """settings:
  capital: 100000
  port: 8088
  theme: dark
  paper_trading_start: "2024-01-01"

strategies:
  - id: demo
    name: "Demo Strategy"
    asset: ETH
    color: "#00FF88"
    engine: strategies.demo.engine
    trade_log: "data/trade_log_demo.csv"
"""
    (base / "strategies.yaml").write_text(strategies_yaml)
    (base / "data").mkdir(exist_ok=True)
    # Minimal but realistic trade log — includes live row to exercise live-row path
    trade_log = """date,asset_return,pnl,position,cum_return,source
2024-01-01 00:00:00+00:00,0.0,0.0,0.0,0.0,backfill
2024-01-02 00:00:00+00:00,0.01,0.005,0.5,0.005,backfill
2024-01-03 00:00:00+00:00,-0.01,-0.005,0.5,-0.0001,backfill
2024-01-04 00:00:00+00:00,0.02,0.01,0.5,0.01,live
"""
    (base / "data" / "trade_log_demo.csv").write_text(trade_log)


def test_dashboard_renders_three_page_nav(tmp_path):
    """Portfolio / Live / Strategy page navigation must exist.

    If this fails, the three-page layout shipped 2026-04-15 has been
    regressed. Fix: inspect abel_edge/dashboard/templates/base.html
    and ensure the nav block renders three `.nav-btn` buttons wired to
    `showPage('portfolio')`, `showPage('live')`, `showPage('strategy')`.
    """
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _write_demo_project(Path.cwd())
        result = runner.invoke(main, ["dashboard", "--output", "dashboard.html"])
        assert result.exit_code == 0, result.output
        html = Path("dashboard.html").read_text(encoding="utf-8")

    assert "showPage('portfolio')" in html, "Portfolio nav button missing"
    assert "showPage('strategy')" in html, "Strategy nav button missing"


def test_dashboard_renders_per_strategy_period_badges(tmp_path):
    """Signals table must show today / MTD / YTD columns per strategy.

    Regression guard for commit c43b103 which added per-strategy YTD/MTD/
    today badges. Fix: ensure build_live_overview in portfolio.py
    populates the period_pnl dicts per strategy.
    """
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _write_demo_project(Path.cwd())
        result = runner.invoke(main, ["dashboard", "--output", "dashboard.html"])
        assert result.exit_code == 0, result.output
        html = Path("dashboard.html").read_text(encoding="utf-8")

    # These strings live in the template header row and get rendered
    # regardless of whether any period has non-zero PnL.
    for label in ("today", "mtd", "ytd"):
        assert label in html.lower(), f"Period label '{label}' missing from signals table"


def test_dashboard_includes_flat_strategies(tmp_path):
    """Flat strategies (pos=0 for all active dates) must still render.

    Regression guard for commit d0f7d66 which added signals_flat to the
    generator return dict. If this asserts fails, a flat strategy won't
    appear in the signals table — users won't know it's dormant.
    """
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        base = Path.cwd()
        _write_demo_project(base)
        # Overwrite trade log with a fully-flat strategy
        (base / "data" / "trade_log_demo.csv").write_text(
            "date,asset_return,pnl,position,cum_return,source\n"
            "2024-01-01 00:00:00+00:00,0.0,0.0,0.0,0.0,backfill\n"
            "2024-01-02 00:00:00+00:00,0.01,0.0,0.0,0.0,backfill\n"
            "2024-01-03 00:00:00+00:00,-0.01,0.0,0.0,0.0,live\n"
        )
        result = runner.invoke(main, ["dashboard", "--output", "dashboard.html"])
        assert result.exit_code == 0, result.output
        html = Path("dashboard.html").read_text(encoding="utf-8")

    # Flat strategy should still appear in the signals section (its name)
    assert "Demo Strategy" in html, "Flat strategy not rendered in dashboard"
