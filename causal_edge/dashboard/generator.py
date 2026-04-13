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
from causal_edge.dashboard.story import strategy_story
from causal_edge.dashboard.rows import filter_backtest_rows, filter_tracking_rows, paper_rows
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
            {"label": "Days Tested", "value": str(metrics.get("n_days", 0))},
            {"label": "Signal State", "value": _signal_label(latest_position)},
        ],
    }


def _attach_tracking_status(strategy: dict, full_df: pd.DataFrame | None, s_cfg: dict) -> dict:
    tracked = filter_tracking_rows(full_df)
    if tracked is None:
        return {
            **strategy,
            **strategy_story(s_cfg),
            "has_tracking_data": False,
            "tracking_status": "Tracking not started",
            "tracking_status_detail": "Run paper trading to start appending live rows.",
            "tracking_summary_cards": [],
            "display_signal_label": strategy.get(
                "signal_label", _signal_label(strategy.get("latest_position", 0.0))
            ),
            "display_signal_source": "Backtest",
        }

    pnl = tracked["pnl"].values.astype(float)
    next_positions = (
        tracked["next_position"].values.astype(float)
        if "next_position" in tracked.columns
        else tracked["position"].values.astype(float)
    )
    tracking_metrics = compute_metrics(pnl)
    latest_live_date = pd.to_datetime(tracked["date"].iloc[-1]).date()
    latest_close = tracked["close"].iloc[-1] if "close" in tracked.columns else None
    latest_signal_position = float(next_positions[-1]) if len(next_positions) > 0 else 0.0
    return {
        **strategy,
        **strategy_story(s_cfg),
        "has_tracking_data": True,
        "tracking_status": "Tracking started",
        "tracking_status_detail": f"Live paper rows are available through {latest_live_date}.",
        "tracking_latest_date": str(latest_live_date),
        "tracking_latest_close": None if pd.isna(latest_close) else float(latest_close),
        "tracking_latest_signal_position": latest_signal_position,
        "tracking_signal_label": _signal_label(latest_signal_position),
        "display_signal_label": _signal_label(latest_signal_position),
        "display_signal_source": "Live",
        "tracking_summary_cards": [
            {
                "label": "Tracked Return",
                "value": fmt_pnl_pct(tracking_metrics.get("cum_return", 0.0)),
            },
            {"label": "Days Tracked", "value": str(tracking_metrics.get("n_days", 0))},
            {"label": "Next Position", "value": f"{latest_signal_position:.2f}"},
        ],
    }


def _tracked_ticker_item(s_cfg: dict) -> dict:
    strategy = _prepare_strategy(s_cfg)
    return {
        "id": strategy["id"],
        "asset": strategy["asset"],
        "tracking_href": strategy["tracking_href"],
        "status": strategy.get("tracking_status", "Tracking not started"),
        "signal_label": strategy.get(
            "display_signal_label", strategy.get("signal_label", "Observe")
        ),
        "latest_date": strategy.get("tracking_latest_date")
        or strategy.get("latest_date")
        or "N/A",
        "has_tracking_data": strategy.get("has_tracking_data", False),
    }


def _prepare_strategy(s_cfg: dict) -> dict:
    """Prepare data dict for a single strategy."""
    full_df = _load_trade_log(s_cfg["trade_log"])
    df = filter_backtest_rows(full_df)
    if df is None or len(df) == 0:
        return _attach_tracking_status(
            {
                "id": s_cfg["id"],
                "name": s_cfg["name"],
                "color": s_cfg["color"],
                "asset": s_cfg["asset"],
                "has_data": False,
                "metrics": {},
                "equity_json": "{}",
                "asset_price_json": "{}",
                "paper_rows": paper_rows(
                    full_df, signal_label=_signal_label, fmt_pnl_pct=fmt_pnl_pct
                ),
                **_strategy_copy(s_cfg),
            },
            full_df,
            s_cfg,
        )

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

    return _attach_tracking_status(
        {
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
            "paper_rows": paper_rows(full_df, signal_label=_signal_label, fmt_pnl_pct=fmt_pnl_pct),
            **_strategy_copy(s_cfg, metrics=metrics, latest_position=latest_position),
        },
        full_df,
        s_cfg,
    )


def _prepare_tracking_strategy(s_cfg: dict) -> dict:
    base = _prepare_strategy(s_cfg)
    full_df = _load_trade_log(s_cfg["trade_log"])
    df = filter_tracking_rows(full_df)
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
            "tracking_rows": [],
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
    next_positions = (
        df["next_position"].values.astype(float)
        if "next_position" in df.columns
        else positions.copy()
    )
    dates = pd.DatetimeIndex(df["date"])
    cum_return = np.cumprod(1.0 + pnl) - 1.0
    tracking_metrics = compute_metrics(pnl)
    live_signal_position = float(next_positions[-1]) if len(next_positions) > 0 else 0.0

    return {
        **base,
        **_strategy_copy(s_cfg, metrics=tracking_metrics, latest_position=live_signal_position),
        "has_tracking_data": True,
        "has_tracking_preview": preload is not None and len(preload) > 0,
        "tracking_start_date": str(dates[0].date()),
        "tracking_latest_date": str(dates[-1].date()),
        "tracking_latest_position": float(positions[-1]) if len(positions) > 0 else 0.0,
        "tracking_latest_signal_position": live_signal_position,
        "tracking_preload_days": int(len(preload)) if preload is not None else 0,
        "tracking_metrics": tracking_metrics,
        "tracking_equity_json": equity_chart(
            dates, cum_return, f"{s_cfg['asset']} Tracking", s_cfg["color"]
        ),
        "tracking_asset_price_json": asset_price_chart(
            dates, asset_returns, s_cfg["asset"], s_cfg["color"]
        ),
        "tracking_position_json": position_chart(
            dates, next_positions, f"{s_cfg['asset']} Signal", s_cfg["color"]
        ),
        "tracking_preview_json": asset_price_chart(
            pd.DatetimeIndex(tracking_df["date"]),
            tracking_df["asset_return"].values.astype(float),
            s_cfg["asset"],
            s_cfg["color"],
        ),
        "tracking_rows": paper_rows(df, signal_label=_signal_label, fmt_pnl_pct=fmt_pnl_pct),
    }


def generate(config_path: str, output_path: str) -> None:
    """Generate dashboard.html from config and trade logs."""
    cfg = load_config(config_path)
    strategies = [_prepare_strategy(s) for s in cfg["strategies"]]

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
        settings=cfg["settings"],
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    Path(output_path).write_text(html, encoding="utf-8")


def generate_signal_demo(config_path: str, output_path: str, strategy_id: str) -> None:
    """Generate a single-strategy Signal Demo page."""
    cfg = load_config(config_path)
    strategies_cfg = [s for s in cfg["strategies"] if s["id"] == strategy_id]
    if not strategies_cfg:
        available = ", ".join(s["id"] for s in cfg["strategies"]) or "none"
        raise ValueError(
            f"Strategy '{strategy_id}' not found in strategies.yaml. Available: {available}"
        )

    selected_strategy = _prepare_strategy(strategies_cfg[0])
    tracking_strategy = _prepare_tracking_strategy(strategies_cfg[0])
    selected_strategy = {
        **selected_strategy,
        "has_tracking_data": tracking_strategy.get("has_tracking_data", False),
        "has_tracking_preview": tracking_strategy.get("has_tracking_preview", False),
        "tracking_start_date": tracking_strategy.get("tracking_start_date"),
        "tracking_latest_date": tracking_strategy.get("tracking_latest_date"),
        "tracking_latest_position": tracking_strategy.get("tracking_latest_position"),
        "tracking_latest_signal_position": tracking_strategy.get(
            "tracking_latest_signal_position"
        ),
        "tracking_preload_days": tracking_strategy.get("tracking_preload_days", 0),
        "tracking_metrics": tracking_strategy.get("tracking_metrics", {}),
        "tracking_asset_price_json": tracking_strategy.get("tracking_asset_price_json", "{}"),
        "tracking_position_json": tracking_strategy.get("tracking_position_json", "{}"),
        "tracking_preview_json": tracking_strategy.get("tracking_preview_json", "{}"),
        "tracking_rows": tracking_strategy.get("tracking_rows", []),
    }
    tracked_tickers = [_tracked_ticker_item(s) for s in cfg["strategies"]]

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )
    env.globals["fmt_pnl_pct"] = fmt_pnl_pct
    env.globals["fmt_dollar"] = fmt_dollar

    template = env.get_template("signal_demo.html")
    html = template.render(
        selected_strategy=selected_strategy,
        tracked_tickers=tracked_tickers,
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
