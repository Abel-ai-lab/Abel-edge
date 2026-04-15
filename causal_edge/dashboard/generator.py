"""Dashboard HTML generator — reads config + trade logs, renders Jinja2 template.

Three-page layout: Live (default) / Portfolio / Strategy drill-down.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from causal_edge.config import load_config
from causal_edge.dashboard._helpers import fmt_pnl_pct, fmt_dollar
from causal_edge.dashboard.components import (
    compute_metrics,
    yearly_metrics,
    live_metrics,
    equity_chart,
    drawdown_chart,
    rolling_sharpe_chart,
    daily_pnl_chart,
    monthly_heatmap,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_trade_log(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None


def _prepare_strategy(s_cfg: dict) -> dict:
    df = _load_trade_log(s_cfg["trade_log"])
    empty = {
        "id": s_cfg["id"], "name": s_cfg["name"],
        "color": s_cfg["color"], "asset": s_cfg["asset"],
        "has_data": False, "metrics": {},
        "health_dots": "", "health_class": "",
    }
    if df is None or len(df) == 0:
        return empty

    pnl = df["pnl"].values.astype(float)
    positions = (df["position"].values.astype(float)
                 if "position" in df.columns else np.zeros(len(pnl)))
    source = (df["source"].values if "source" in df.columns
              else np.array(["backfill"] * len(pnl)))
    asset_close = (df["asset_close"].values.astype(float)
                   if "asset_close" in df.columns else None)
    dates = pd.DatetimeIndex(df["date"])
    cum_pnl = np.cumsum(pnl)
    name = s_cfg["name"]
    color = s_cfg["color"]

    # Validation
    validation = None
    try:
        from causal_edge.validation.gate import validate_strategy
        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            vdf = pd.DataFrame({
                "date": dates, "pnl": pnl, "position": positions,
            })
            vdf.to_csv(f.name, index=False)
            vr = validate_strategy(f.name, positions_col="position")
            Path(f.name).unlink(missing_ok=True)
            validation = {
                "verdict": vr["verdict"],
                "score": vr["score"],
                "failures": vr.get("failures", []),
                "triangle": vr.get("triangle", {}),
                "metrics": {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in vr.get("metrics", {}).items()
                    if k in (
                        "sharpe", "lo_adjusted", "ic", "omega",
                        "max_dd", "dsr", "pbo", "oos_is",
                        "neg_roll_frac", "loss_years", "total_pnl",
                        "calmar", "bootstrap_p", "ic_stability",
                        "sharpe_lo_ratio", "sortino", "ic_hit_rate",
                    )
                },
            }
    except Exception:
        pass

    # Health dots (live vs backtest degradation)
    health_dots, health_class = "", ""
    bt_sharpe = compute_metrics(pnl)["sharpe"]
    live_data = live_metrics(dates, pnl, positions, source, asset_close)
    if live_data and live_data["n_days"] > 10 and bt_sharpe > 0:
        ratio = live_data["sharpe"] / bt_sharpe
        n_dots = max(1, min(5, int(ratio * 5)))
        health_dots = "\u25cf" * n_dots + "\u25cb" * (5 - n_dots)
        health_class = (
            "health-good" if ratio > 0.7
            else "health-warn" if ratio > 0.4
            else "health-bad"
        )
    elif live_data and live_data["n_days"] > 0:
        health_dots = "\u25cb" * 5
        health_class = "health-warn"

    return {
        "id": s_cfg["id"],
        "name": name,
        "color": color,
        "asset": s_cfg["asset"],
        "badge": s_cfg.get("badge", ""),
        "has_data": True,
        "metrics": compute_metrics(pnl),
        "live": live_data,
        "validation": validation,
        "yearly": yearly_metrics(dates, pnl),
        "health_dots": health_dots,
        "health_class": health_class,
        # Charts
        "equity_json": equity_chart(dates, cum_pnl, name, color),
        "drawdown_json": drawdown_chart(dates, cum_pnl, name),
        "rolling_sharpe_json": rolling_sharpe_chart(dates, pnl, name),
        "daily_pnl_json": daily_pnl_chart(dates, pnl, name),
        "monthly_json": monthly_heatmap(dates, pnl, name),
        "latest_date": str(dates[-1].date()),
        "latest_position": float(positions[-1]),
        "latest_pnl": float(pnl[-1]),
        "prev_position": float(positions[-2]) if len(positions) > 1 else 0,
    }


def _sparkline(values: list[float], width: int = 7) -> str:
    """Unicode block sparkline."""
    blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    if not values or all(v == 0 for v in values):
        return "\u2581" * width
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1
    return "".join(
        blocks[min(8, max(0, int((v - mn) / rng * 7) + 1))]
        for v in values[-width:]
    )


def _build_portfolio(strategies: list[dict], settings: dict) -> dict:
    """Aggregate portfolio-level metrics for the Live page."""
    now = datetime.now()
    capital = settings.get("capital", 100_000)

    # Collect latest signals
    signals_active = []
    signals_flat = []
    today_pnl = 0.0
    n_active = 0

    for s in strategies:
        if not s["has_data"]:
            continue
        pos = s["latest_position"]
        pnl_today = s["latest_pnl"]
        changed = abs(pos - s["prev_position"]) > 0.01
        today_pnl += pnl_today

        sig = {
            "name": s["name"],
            "color": s["color"],
            "position": pos,
            "today_pnl": pnl_today,
            "changed": changed,
            "id": s["id"],
        }
        if abs(pos) > 0.01:
            signals_active.append(sig)
            n_active += 1
        else:
            signals_flat.append(sig)

    signals_active.sort(key=lambda x: x["position"], reverse=True)

    # MTD and live-total PnL (sum across strategies, live rows only)
    mtd_pnl = 0.0
    total_pnl = 0.0
    current_month = now.month
    current_year = now.year

    for s in strategies:
        if not s["has_data"]:
            continue
        df = _load_trade_log(
            next(
                (sc["trade_log"] for sc in _strat_cfgs
                 if sc["id"] == s["id"]),
                "",
            )
        )
        if df is not None:
            df["date"] = pd.to_datetime(df["date"])
            # Dedup (strategy, date) — same day can have both backfill and live rows
            # (Step 6b writes both). Prefer live when both exist.
            df_u = df.copy()
            df_u["_date_str"] = df_u["date"].dt.strftime("%Y-%m-%d")
            if "source" in df_u.columns:
                df_u["_src_rank"] = df_u["source"].map({"live": 1, "backfill": 0}).fillna(0)
                df_u = df_u.sort_values(["_date_str", "_src_rank"])
            df_u = df_u.drop_duplicates(subset=["_date_str"], keep="last")
            # YTD: current year. Treats backfill as live-equivalent because
            # pure-function engines would have produced identical signals on
            # each of those dates (paper trading has been running all along).
            ytd_mask = df_u["date"].dt.year == current_year
            total_pnl += df_u.loc[ytd_mask, "pnl"].sum()
            # MTD: current month, deduped
            mtd_mask = ytd_mask & (df_u["date"].dt.month == current_month)
            mtd_pnl += df_u.loc[mtd_mask, "pnl"].sum()

    # Recent days + ledger: both dedup (strategy, date). Extracted to portfolio.py
    # so this file stays under the 400-line structural limit.
    from causal_edge.dashboard.portfolio import build_recent_days
    recent_days, pnl_history = build_recent_days(
        strategies, _strat_cfgs, _load_trade_log,
    )
    if recent_days:
        recent_days[0]["spark"] = _sparkline(list(reversed(pnl_history)))

    # Asset prices (latest from trade logs)
    prices = {}
    seen_assets = set()
    for s in strategies:
        if not s["has_data"] or s["asset"] in seen_assets:
            continue
        df = _load_trade_log(
            next(
                (sc["trade_log"] for sc in _strat_cfgs
                 if sc["id"] == s["id"]),
                "",
            )
        )
        if df is not None and "asset_close" in df.columns:
            price = df["asset_close"].iloc[-1]
            if price > 0:
                prices[s["asset"]] = price
                seen_assets.add(s["asset"])

    # Stale check
    latest_dates = []
    for s in strategies:
        if s["has_data"]:
            latest_dates.append(pd.Timestamp(s["latest_date"]))
    if latest_dates:
        most_recent = max(latest_dates)
        stale_hours = (pd.Timestamp(now) - most_recent).total_seconds() / 3600
        if stale_hours < 1:
            stale_label = "just now"
        elif stale_hours < 24:
            stale_label = f"{int(stale_hours)}h ago"
        else:
            stale_label = f"{int(stale_hours / 24)}d ago"
    else:
        stale_hours = 999
        stale_label = "unknown"

    # Live performance summary per strategy
    live_perf = []
    for s in strategies:
        if not s["has_data"]:
            continue
        cfg_match = next(
            (sc for sc in _strat_cfgs if sc["id"] == s["id"]), None,
        )
        if not cfg_match:
            continue
        df = _load_trade_log(cfg_match["trade_log"])
        if df is None or "source" not in df.columns:
            continue
        live = df[df["source"] == "live"]
        if len(live) < 2:
            continue
        lp = live["pnl"].values.astype(float)
        lpos = live["position"].values.astype(float) if "position" in live.columns else np.zeros(len(lp))
        cum = np.cumsum(lp)
        eq = np.exp(cum)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0
        std = np.std(lp, ddof=1) if len(lp) > 1 else 0
        sharpe = float(np.mean(lp) / std * np.sqrt(252)) if std > 0 else 0
        active = lp[np.abs(lp) > 1e-10]
        win_rate = float(np.mean(active > 0)) if len(active) > 0 else 0
        live_perf.append({
            "name": s["name"],
            "color": s["color"],
            "id": s["id"],
            "days": len(lp),
            "sharpe": round(sharpe, 2),
            "pnl": round(float(cum[-1]), 4),
            "max_dd": round(max_dd, 4),
            "win_rate": round(win_rate, 3),
            "active_days": int(np.sum(np.abs(lpos) > 0.01)),
        })
    live_perf.sort(key=lambda x: x["sharpe"], reverse=True)

    from causal_edge.dashboard.portfolio import build_ledger
    since = settings.get("paper_trading_start")  # e.g. "2026-03-01"
    ledger_days = int(settings.get("ledger_days", 30))
    ledger = build_ledger(
        strategies, _strat_cfgs, _load_trade_log,
        since_date=since, n_days=ledger_days,
    )

    return {
        "today_pnl": today_pnl,
        "mtd_pnl": mtd_pnl,
        "total_pnl": total_pnl,
        "n_active": n_active,
        "n_flat": len(signals_flat),
        "n_strats": len([s for s in strategies if s["has_data"]]),
        "signals_active": signals_active,
        "recent_days": recent_days,
        "prices": prices,
        "stale_hours": stale_hours,
        "stale_label": stale_label,
        "live_perf": live_perf,
        "ledger": ledger,
    }


# Module-level for _build_portfolio to access
_strat_cfgs: list[dict] = []


def generate(config_path: str, output_path: str) -> None:
    """Generate the dashboard HTML."""
    global _strat_cfgs
    cfg = load_config(config_path)
    strat_list = cfg["strategies"]
    _strat_cfgs = strat_list

    if len(strat_list) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(
            max_workers=min(len(strat_list), 8)
        ) as pool:
            strategies = list(pool.map(_prepare_strategy, strat_list))
    else:
        strategies = [_prepare_strategy(s) for s in strat_list]

    portfolio = _build_portfolio(strategies, cfg.get("settings", {}))

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False,
    )
    env.globals["fmt_pnl_pct"] = fmt_pnl_pct
    env.globals["fmt_dollar"] = fmt_dollar

    template = env.get_template("base.html")
    html = template.render(
        strategies=strategies,
        portfolio=portfolio,
        settings=cfg.get("settings", {}),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    Path(output_path).write_text(html)
