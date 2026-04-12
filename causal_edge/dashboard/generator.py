"""Dashboard HTML generator — reads config + trade logs, renders Jinja2 templates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from causal_edge.config import load_config
from causal_edge.dashboard._helpers import fmt_pnl_pct, fmt_dollar
from causal_edge.dashboard.components import (
    asset_price_chart,
    asset_index_from_returns,
    compute_metrics,
    equity_chart,
    position_chart,
)
from causal_edge.validation.metrics import detect_profile, load_profile

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_trade_log(path: str) -> pd.DataFrame | None:
    """Load trade log CSV, return None if not found."""
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None


def _signal_label(position: float) -> str:
    if position >= 0.75:
        return "Hold"
    if position >= 0.25:
        return "Track Lightly"
    return "Observe"


def _signal_summary(position: float) -> str:
    if position >= 0.75:
        return "The model sees strong enough alignment to keep this signal in a hold state."
    if position >= 0.25:
        return "The model sees partial alignment, so this stays on a light tracking footing."
    return "The model does not yet have enough conviction to move beyond observation."


def _strategy_copy(s_cfg: dict, metrics: dict | None = None, latest_position: float = 0.0) -> dict:
    metrics = metrics or {}
    return {
        "thesis": s_cfg.get(
            "thesis",
            f"{s_cfg['asset']} is being tracked because its causal drivers are showing a usable setup.",
        ),
        "audience": s_cfg.get("audience", "Beginner signal followers"),
        "cta_text": s_cfg.get("cta_text", "Start tracking this signal"),
        "why_now": s_cfg.get(
            "why_now",
            "The latest driver mix is coherent enough to keep this ticker on the shortlist.",
        ),
        "risk_note": s_cfg.get(
            "risk_note",
            "Treat this as a tracked signal, not an automatic buy instruction.",
        ),
        "tracking_href": s_cfg.get("tracking_href", f"signal-track-{s_cfg['asset'].lower()}.html"),
        "signal_label_rules": s_cfg.get(
            "signal_label_rules",
            "Observe < 0.25, Track Lightly 0.25-0.74, Hold >= 0.75",
        ),
        "signal_label": _signal_label(latest_position),
        "signal_summary": _signal_summary(latest_position),
        "summary_cards": [
            {"label": "Backtest Return", "value": fmt_pnl_pct(metrics.get("cum_return", 0.0))},
            {"label": "Max Drawdown", "value": f"-{metrics.get('max_dd', 0.0) * 100:.1f}%"},
            {"label": "Win Rate", "value": f"{metrics.get('win_rate', 0.0) * 100:.0f}%"},
            {"label": "Days Tested", "value": str(metrics.get("n_days", 0))},
        ],
    }


def _filter_tracking_rows(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return None
    if "source" not in df.columns:
        return None
    tracked = df[df["source"].astype(str).str.lower() == "live"].copy()
    if len(tracked) == 0:
        return None
    return tracked


def _prepare_strategy(s_cfg: dict) -> dict:
    """Prepare data dict for a single strategy."""
    df = _load_trade_log(s_cfg["trade_log"])
    if df is None or len(df) == 0:
        return {
            "id": s_cfg["id"],
            "name": s_cfg["name"],
            "color": s_cfg["color"],
            "asset": s_cfg["asset"],
            "has_data": False,
            "metrics": {},
            "equity_json": "{}",
            "asset_price_json": "{}",
            **_strategy_copy(s_cfg),
        }

    pnl = df["pnl"].values.astype(float)
    asset_returns = (
        df["asset_return"].values.astype(float) if "asset_return" in df.columns else None
    )
    positions = (
        df["position"].values.astype(float) if "position" in df.columns else np.zeros(len(pnl))
    )
    dates = pd.DatetimeIndex(df["date"])
    cum_return = np.cumprod(1.0 + pnl) - 1.0

    profile_name = detect_profile(pnl, dates, asset_returns=asset_returns)
    profile = load_profile(profile_name)
    periods_per_year = profile.get("validation", {}).get("periods_per_year", 252)

    metrics = compute_metrics(pnl, periods_per_year=periods_per_year)

    latest_position = float(positions[-1]) if len(positions) > 0 else 0.0
    asset_index = asset_index_from_returns(asset_returns if asset_returns is not None else pnl)

    return {
        "id": s_cfg["id"],
        "name": s_cfg["name"],
        "color": s_cfg["color"],
        "asset": s_cfg["asset"],
        "has_data": True,
        "metrics": metrics,
        "equity_json": equity_chart(
            dates,
            cum_return,
            s_cfg["name"],
            s_cfg["color"],
            asset_index=asset_index,
            asset=s_cfg["asset"],
        ),
        "asset_price_json": asset_price_chart(
            dates, asset_returns, s_cfg["asset"], s_cfg["color"]
        ),
        "position_json": position_chart(dates, positions, s_cfg["name"], s_cfg["color"]),
        "latest_date": str(dates[-1].date()) if len(dates) > 0 else "N/A",
        "latest_position": latest_position,
        **_strategy_copy(s_cfg, metrics=metrics, latest_position=latest_position),
    }


def _prepare_tracking_strategy(s_cfg: dict) -> dict:
    base = _prepare_strategy(s_cfg)
    full_df = _load_trade_log(s_cfg["trade_log"])
    df = _filter_tracking_rows(full_df)
    if df is None:
        preview_df = full_df.tail(30).copy() if full_df is not None and len(full_df) > 0 else None
        return {
            **base,
            "has_tracking_data": False,
            "has_tracking_preview": preview_df is not None,
            "tracking_metrics": {},
            "tracking_equity_json": "{}",
            "tracking_asset_price_json": "{}",
            "tracking_preview_json": (
                asset_price_chart(
                    pd.DatetimeIndex(preview_df["date"]),
                    preview_df["asset_return"].values.astype(float),
                    s_cfg["asset"],
                    s_cfg["color"],
                )
                if preview_df is not None and "asset_return" in preview_df.columns
                else "{}"
            ),
        }

    live_start = df["date"].iloc[0]
    preload = (
        full_df[full_df["date"] < live_start].tail(30).copy() if full_df is not None else None
    )
    tracking_df = (
        pd.concat([preload, df], ignore_index=True)
        if preload is not None and len(preload) > 0
        else df.copy()
    )

    pnl = df["pnl"].values.astype(float)
    asset_returns = (
        df["asset_return"].values.astype(float)
        if "asset_return" in df.columns
        else np.zeros(len(pnl))
    )
    positions = (
        df["position"].values.astype(float) if "position" in df.columns else np.zeros(len(pnl))
    )
    dates = pd.DatetimeIndex(df["date"])
    cum_return = np.cumprod(1.0 + pnl) - 1.0
    tracking_metrics = compute_metrics(pnl)

    return {
        **base,
        "has_tracking_data": True,
        "has_tracking_preview": preload is not None and len(preload) > 0,
        "tracking_start_date": str(dates[0].date()),
        "tracking_latest_date": str(dates[-1].date()),
        "tracking_latest_position": float(positions[-1]) if len(positions) > 0 else 0.0,
        "tracking_preload_days": int(len(preload)) if preload is not None else 0,
        "tracking_metrics": tracking_metrics,
        "tracking_equity_json": equity_chart(
            dates, cum_return, f"{s_cfg['asset']} Tracking", s_cfg["color"]
        ),
        "tracking_asset_price_json": asset_price_chart(
            dates, asset_returns, s_cfg["asset"], s_cfg["color"]
        ),
        "tracking_preview_json": asset_price_chart(
            pd.DatetimeIndex(tracking_df["date"]),
            tracking_df["asset_return"].values.astype(float),
            s_cfg["asset"],
            s_cfg["color"],
        ),
    }


def generate(config_path: str, output_path: str, strategy_id: str | None = None) -> None:
    """Generate dashboard.html from config and trade logs.

    Args:
        config_path: Path to strategies.yaml
        output_path: Path to write dashboard.html
    """
    cfg = load_config(config_path)
    strategies_cfg = cfg["strategies"]
    if strategy_id:
        strategies_cfg = [s for s in strategies_cfg if s["id"] == strategy_id]
        if not strategies_cfg:
            available = ", ".join(s["id"] for s in cfg["strategies"]) or "none"
            raise ValueError(
                f"Strategy '{strategy_id}' not found in strategies.yaml. Available: {available}"
            )

    strategies = [_prepare_strategy(s) for s in strategies_cfg]
    selected_strategy = strategies[0] if strategy_id and strategies else None

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )
    # Register helper functions
    env.globals["fmt_pnl_pct"] = fmt_pnl_pct
    env.globals["fmt_dollar"] = fmt_dollar

    template = env.get_template("base.html")
    html = template.render(
        strategies=strategies,
        selected_strategy=selected_strategy,
        settings=cfg["settings"],
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    Path(output_path).write_text(html, encoding="utf-8")


def generate_tracking_page(config_path: str, output_path: str, strategy_id: str) -> None:
    cfg = load_config(config_path)
    strategies_cfg = [s for s in cfg["strategies"] if s["id"] == strategy_id]
    if not strategies_cfg:
        available = ", ".join(s["id"] for s in cfg["strategies"]) or "none"
        raise ValueError(
            f"Strategy '{strategy_id}' not found in strategies.yaml. Available: {available}"
        )

    strategy = _prepare_tracking_strategy(strategies_cfg[0])

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
    )
    env.globals["fmt_pnl_pct"] = fmt_pnl_pct
    env.globals["fmt_dollar"] = fmt_dollar

    template = env.get_template("tracking.html")
    html = template.render(
        selected_strategy=strategy,
        settings=cfg["settings"],
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    Path(output_path).write_text(html, encoding="utf-8")
