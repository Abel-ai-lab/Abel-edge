"""Dashboard HTML generator — reads config + prepared strategy payloads."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from causal_edge.config import load_config
from causal_edge.dashboard._helpers import fmt_dollar, fmt_pnl_pct
from causal_edge.dashboard.components import (
    asset_index_from_returns,
    asset_price_chart,
    compute_metrics,
    equity_chart,
    position_chart,
)
from causal_edge.dashboard.live_overview import build_live_overview
from causal_edge.dashboard.strategy_data import prepare_strategy, tracked_ticker_item

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Component names stay imported here so structural tests can verify the dashboard surface.
_REGISTERED_COMPONENTS = (
    compute_metrics,
    equity_chart,
    asset_price_chart,
    asset_index_from_returns,
    position_chart,
)


def _build_env(*, autoescape: bool) -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=autoescape,
    )
    env.globals["fmt_pnl_pct"] = fmt_pnl_pct
    env.globals["fmt_dollar"] = fmt_dollar
    return env


def generate(config_path: str | None, output_path: str, *, bars_loader=None) -> None:
    """Generate dashboard.html from config and strategy payloads."""
    cfg = load_config(config_path)
    strategies = [
        prepare_strategy(s, settings=cfg["settings"], bars_loader=bars_loader)
        for s in cfg["strategies"]
    ]
    live_overview = build_live_overview(strategies, cfg["strategies"], cfg["settings"])
    env = _build_env(autoescape=True)
    template = env.get_template("base.html")
    html = template.render(
        strategies=strategies,
        live_overview=live_overview,
        settings=cfg["settings"],
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        initial_strategy_id=None,
        initial_section="backtest",
    )
    Path(output_path).write_text(html, encoding="utf-8")


def generate_strategy_dashboard(
    config_path: str | None,
    output_path: str,
    strategy_id: str,
    *,
    bars_loader=None,
    initial_section: str = "paper",
) -> None:
    """Generate the standard dashboard surface focused on one strategy."""
    cfg = load_config(config_path)
    strategies_cfg = [s for s in cfg["strategies"] if s["id"] == strategy_id]
    if not strategies_cfg:
        available = ", ".join(s["id"] for s in cfg["strategies"]) or "none"
        raise ValueError(
            f"Strategy '{strategy_id}' not found in strategies.yaml. Available: {available}"
        )

    strategies = [
        prepare_strategy(strategies_cfg[0], settings=cfg["settings"], bars_loader=bars_loader)
    ]
    live_overview = build_live_overview(strategies, strategies_cfg, cfg["settings"])
    env = _build_env(autoescape=True)
    template = env.get_template("base.html")
    html = template.render(
        strategies=strategies,
        live_overview=live_overview,
        settings=cfg["settings"],
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        initial_strategy_id=strategy_id,
        initial_section=initial_section,
    )
    Path(output_path).write_text(html, encoding="utf-8")


def generate_signal_demo(
    config_path: str | None, output_path: str, strategy_id: str, *, bars_loader=None
) -> None:
    """Generate a single-strategy Signal Demo page."""
    cfg = load_config(config_path)
    strategies_cfg = [s for s in cfg["strategies"] if s["id"] == strategy_id]
    if not strategies_cfg:
        available = ", ".join(s["id"] for s in cfg["strategies"]) or "none"
        raise ValueError(
            f"Strategy '{strategy_id}' not found in strategies.yaml. Available: {available}"
        )

    selected_strategy = prepare_strategy(
        strategies_cfg[0], settings=cfg["settings"], bars_loader=bars_loader
    )
    tracked_tickers = [
        tracked_ticker_item(s, settings=cfg["settings"], bars_loader=bars_loader)
        for s in cfg["strategies"]
    ]
    env = _build_env(autoescape=True)
    template = env.get_template("signal_demo.html")
    html = template.render(
        selected_strategy=selected_strategy,
        tracked_tickers=tracked_tickers,
        settings=cfg["settings"],
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    Path(output_path).write_text(html, encoding="utf-8")


def generate_tracking_page(
    config_path: str | None, output_path: str, strategy_id: str, *, bars_loader=None
) -> None:
    """Compatibility wrapper: tracking now reuses the dashboard surface."""
    generate_strategy_dashboard(
        config_path,
        output_path,
        strategy_id,
        bars_loader=bars_loader,
        initial_section="paper",
    )
