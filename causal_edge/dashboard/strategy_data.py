"""Strategy data preparation for dashboard-style pages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_edge.dashboard._helpers import fmt_pnl_pct
from causal_edge.dashboard.components import (
    asset_index_from_returns,
    asset_price_chart,
    compute_metrics,
    equity_chart,
    position_chart,
)
from causal_edge.dashboard.price_overlay import fetch_price_overlay
from causal_edge.dashboard.rows import filter_backtest_rows, filter_tracking_rows, paper_rows
from causal_edge.dashboard.story import strategy_story
from causal_edge.validation.metrics import detect_profile, load_profile


def _load_trade_log(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
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


def _normalize_paper_df(
    backtest_df: pd.DataFrame | None, paper_df: pd.DataFrame | None
) -> pd.DataFrame | None:
    if paper_df is not None and len(paper_df) > 0:
        normalized = paper_df.copy()
        if "source" not in normalized.columns:
            normalized["source"] = "live"
        else:
            normalized["source"] = normalized["source"].fillna("live")
        return normalized.sort_values("date").reset_index(drop=True)
    return filter_tracking_rows(backtest_df)


def _load_strategy_frames(s_cfg: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    backtest_full = _load_trade_log(s_cfg["trade_log"])
    paper_full = _load_trade_log(s_cfg.get("paper_log"))
    return backtest_full, _normalize_paper_df(backtest_full, paper_full)


def _build_tracking_payload(
    base: dict,
    backtest_df: pd.DataFrame | None,
    paper_df: pd.DataFrame | None,
    s_cfg: dict,
    *,
    settings: dict,
    bars_loader,
) -> dict:
    if paper_df is None or len(paper_df) == 0:
        preview_df = (
            backtest_df.tail(30).copy()
            if backtest_df is not None and len(backtest_df) > 0
            else None
        )
        return {
            **base,
            **strategy_story(s_cfg),
            "has_tracking_data": False,
            "has_tracking_preview": preview_df is not None
            and "asset_return" in preview_df.columns,
            "has_tracking_preview_chart": preview_df is not None
            and "asset_return" in preview_df.columns,
            "has_tracking_asset_chart": False,
            "tracking_signal_chart_title": "Signal Changes Since Tracking Began",
            "tracking_status": "Tracking not started",
            "tracking_status_detail": "Run paper trading to start appending live rows.",
            "tracking_latest_date": None,
            "tracking_latest_close": None,
            "tracking_latest_position": None,
            "tracking_latest_signal_position": None,
            "tracking_preload_days": int(len(preview_df)) if preview_df is not None else 0,
            "tracking_metrics": {},
            "tracking_equity_json": "{}",
            "tracking_asset_price_json": "{}",
            "tracking_position_json": "{}",
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
            "tracking_summary_cards": [],
            "display_signal_label": base.get(
                "signal_label", _signal_label(base.get("latest_position", 0.0))
            ),
            "display_signal_source": "Backtest",
        }

    paper_df = paper_df.sort_values("date").reset_index(drop=True)
    pnl = paper_df["pnl"].values.astype(float)
    has_asset_returns = "asset_return" in paper_df.columns
    asset_returns = (
        paper_df["asset_return"].values.astype(float) if has_asset_returns else np.zeros(len(pnl))
    )
    has_next_positions = "next_position" in paper_df.columns
    positions = (
        paper_df["position"].values.astype(float)
        if "position" in paper_df.columns
        else np.zeros(len(pnl))
    )
    next_positions = (
        paper_df["next_position"].values.astype(float)
        if "next_position" in paper_df.columns
        else positions.copy()
    )
    dates = pd.DatetimeIndex(paper_df["date"])
    tracking_metrics = compute_metrics(pnl)
    live_signal_position = float(next_positions[-1]) if len(next_positions) > 0 else 0.0
    latest_close = paper_df["close"].iloc[-1] if "close" in paper_df.columns else None
    preload = (
        backtest_df.tail(30).copy() if backtest_df is not None and len(backtest_df) > 0 else None
    )
    preview_df = (
        pd.concat([preload, paper_df], ignore_index=True)
        if preload is not None and len(preload) > 0
        else paper_df.copy()
    )
    cum_return = np.cumprod(1.0 + pnl) - 1.0

    price_overlay = fetch_price_overlay(
        s_cfg, settings, bars_loader, start=dates[0], end=dates[-1]
    )
    if not has_asset_returns and price_overlay is not None:
        has_asset_returns = True
        asset_returns = price_overlay["returns"]
        latest_close = float(price_overlay["close"][-1])

    preview_price_overlay = None
    if preview_df is not None and len(preview_df) > 0:
        preview_dates = pd.DatetimeIndex(preview_df["date"])
        preview_price_overlay = fetch_price_overlay(
            s_cfg,
            settings,
            bars_loader,
            start=preview_dates[0],
            end=preview_dates[-1],
        )

    return {
        **base,
        **strategy_story(s_cfg),
        "signal_label": _signal_label(live_signal_position),
        "signal_summary": _signal_summary(live_signal_position),
        "has_tracking_data": True,
        "has_tracking_preview": preview_df is not None and len(preview_df) > len(paper_df),
        "has_tracking_preview_chart": (
            preview_price_overlay is not None
            or (
                preview_df is not None
                and len(preview_df) > len(paper_df)
                and "asset_return" in preview_df.columns
            )
        ),
        "has_tracking_asset_chart": has_asset_returns,
        "tracking_signal_chart_title": (
            "Signal Changes Since Tracking Began"
            if has_next_positions
            else "Position Since Tracking Began"
        ),
        "tracking_status": "Tracking started",
        "tracking_status_detail": f"Live paper rows are available through {dates[-1].date()}.",
        "tracking_start_date": str(dates[0].date()),
        "tracking_latest_date": str(dates[-1].date()),
        "tracking_latest_close": None if pd.isna(latest_close) else float(latest_close),
        "tracking_latest_position": float(positions[-1]) if len(positions) > 0 else 0.0,
        "tracking_latest_signal_position": live_signal_position,
        "tracking_preload_days": int(len(preload)) if preload is not None else 0,
        "tracking_metrics": tracking_metrics,
        "tracking_equity_json": equity_chart(
            dates, cum_return, f"{s_cfg['asset']} Tracking", s_cfg["color"]
        ),
        "tracking_asset_price_json": (
            asset_price_chart(
                price_overlay["dates"] if price_overlay is not None else dates,
                asset_returns,
                s_cfg["asset"],
                s_cfg["color"],
            )
            if has_asset_returns
            else "{}"
        ),
        "tracking_position_json": position_chart(
            dates, next_positions, f"{s_cfg['asset']} Signal", s_cfg["color"]
        ),
        "tracking_preview_json": (
            asset_price_chart(
                preview_price_overlay["dates"]
                if preview_price_overlay is not None
                else pd.DatetimeIndex(preview_df["date"]),
                preview_price_overlay["returns"]
                if preview_price_overlay is not None
                else preview_df["asset_return"].values.astype(float),
                s_cfg["asset"],
                s_cfg["color"],
            )
            if preview_price_overlay is not None or "asset_return" in preview_df.columns
            else "{}"
        ),
        "tracking_rows": paper_rows(paper_df, signal_label=_signal_label, fmt_pnl_pct=fmt_pnl_pct),
        "tracking_summary_cards": [
            {
                "label": "Tracked Return",
                "value": fmt_pnl_pct(tracking_metrics.get("cum_return", 0.0)),
            },
            {"label": "Sharpe", "value": f"{tracking_metrics.get('sharpe', 0.0):.2f}"},
            {
                "label": "Max Drawdown",
                "value": fmt_pnl_pct(-1.0 * tracking_metrics.get("max_dd", 0.0)),
            },
            {
                "label": "Win Rate",
                "value": f"{tracking_metrics.get('win_rate', 0.0) * 100:.0f}%",
            },
            {"label": "Days Tracked", "value": str(tracking_metrics.get("n_days", 0))},
            {"label": "Next Position", "value": f"{live_signal_position:.2f}"},
        ],
        "display_signal_label": _signal_label(live_signal_position),
        "display_signal_source": "Live",
    }


def prepare_strategy(s_cfg: dict, settings: dict | None = None, bars_loader=None) -> dict:
    settings = settings or {}
    backtest_full, paper_df = _load_strategy_frames(s_cfg)
    backtest_df = filter_backtest_rows(backtest_full)

    if backtest_df is None or len(backtest_df) == 0:
        base = {
            "id": s_cfg["id"],
            "name": s_cfg["name"],
            "color": s_cfg["color"],
            "asset": s_cfg["asset"],
            "has_data": False,
            "has_backtest_asset_chart": False,
            "metrics": {},
            "equity_json": "{}",
            "asset_price_json": "{}",
            "position_json": "{}",
            "latest_date": "N/A",
            "latest_position": 0.0,
            **_strategy_copy(s_cfg),
        }
        return _build_tracking_payload(
            base,
            backtest_df,
            paper_df,
            s_cfg,
            settings=settings,
            bars_loader=bars_loader,
        )

    pnl = backtest_df["pnl"].values.astype(float)
    asset_returns = (
        backtest_df["asset_return"].values.astype(float)
        if "asset_return" in backtest_df.columns
        else None
    )
    dates = pd.DatetimeIndex(backtest_df["date"])
    price_overlay = None
    if asset_returns is None and len(dates) > 0:
        price_overlay = fetch_price_overlay(
            s_cfg, settings, bars_loader, start=dates[0], end=dates[-1]
        )
        if price_overlay is not None:
            asset_returns = price_overlay["returns"]

    has_backtest_asset_chart = asset_returns is not None
    positions = (
        backtest_df["position"].values.astype(float)
        if "position" in backtest_df.columns
        else np.zeros(len(pnl))
    )
    cum_return = np.cumprod(1.0 + pnl) - 1.0
    profile_name = detect_profile(pnl, dates, asset_returns=asset_returns)
    profile = load_profile(profile_name)
    periods_per_year = profile.get("validation", {}).get("periods_per_year", 252)
    metrics = compute_metrics(pnl, periods_per_year=periods_per_year)
    latest_position = float(positions[-1]) if len(positions) > 0 else 0.0
    asset_index = asset_index_from_returns(asset_returns) if asset_returns is not None else None

    base = {
        "id": s_cfg["id"],
        "name": s_cfg["name"],
        "color": s_cfg["color"],
        "asset": s_cfg["asset"],
        "has_data": True,
        "has_backtest_asset_chart": has_backtest_asset_chart,
        "metrics": metrics,
        "equity_json": equity_chart(
            dates,
            cum_return,
            s_cfg["name"],
            s_cfg["color"],
            asset_index=asset_index,
            asset=s_cfg["asset"],
            asset_dates=(price_overlay["dates"] if price_overlay is not None else None),
        ),
        "asset_price_json": (
            asset_price_chart(
                price_overlay["dates"] if price_overlay is not None else dates,
                asset_returns,
                s_cfg["asset"],
                s_cfg["color"],
            )
            if asset_returns is not None
            else "{}"
        ),
        "position_json": position_chart(dates, positions, s_cfg["name"], s_cfg["color"]),
        "latest_date": str(dates[-1].date()),
        "latest_position": latest_position,
        **_strategy_copy(s_cfg, metrics=metrics, latest_position=latest_position),
    }
    return _build_tracking_payload(
        base,
        backtest_df,
        paper_df,
        s_cfg,
        settings=settings,
        bars_loader=bars_loader,
    )


def tracked_ticker_item(s_cfg: dict, settings: dict | None = None, bars_loader=None) -> dict:
    strategy = prepare_strategy(s_cfg, settings=settings, bars_loader=bars_loader)
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
