"""Portfolio-level aggregators (ledger, recent days) extracted from generator.py.

Kept separate so generator.py stays under the 400-line structural limit
(tests/test_structure.py::TestFileSizeLimit).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def build_recent_days(
    strategies: list[dict],
    strat_cfgs: list[dict],
    load_trade_log: Callable[[str], pd.DataFrame | None],
    primary_portfolio_id: str | None = None,
) -> tuple[list[dict], list[float]]:
    """Build last-7-days summary aligned with the primary portfolio's PnL.

    When ``primary_portfolio_id`` is provided AND that strategy has a trade
    log, the sparkline uses ITS daily ``pnl`` (already weighted across
    components by the composite engine), and ``n_active`` counts how many
    of its declared ``components`` had a non-flat position that day. This
    keeps the Recent Days widget on the same equity curve as the hero
    metrics (Sharpe / MaxDD / cum_return) — i.e. SSOT alignment.

    Falls back to the legacy "raw sum across all strategies" behavior only
    when no primary portfolio is configured.
    """
    primary_cfg = next(
        (c for c in strat_cfgs if c["id"] == primary_portfolio_id),
        None,
    )
    primary_df = (
        load_trade_log(primary_cfg["trade_log"])
        if primary_cfg is not None
        else None
    )

    if primary_cfg is not None and primary_df is not None:
        return _recent_days_primary(
            primary_cfg, primary_df, strat_cfgs, load_trade_log,
        )

    return _recent_days_legacy_sum(strategies, strat_cfgs, load_trade_log)


def _recent_days_primary(
    primary_cfg: dict,
    primary_df: pd.DataFrame,
    strat_cfgs: list[dict],
    load_trade_log: Callable[[str], pd.DataFrame | None],
) -> tuple[list[dict], list[float]]:
    df = primary_df.tail(60).copy()
    df["_date_str"] = df["date"].apply(lambda x: str(pd.Timestamp(x).date()))
    df = df.drop_duplicates(subset=["_date_str"], keep="last")
    df = df.sort_values("_date_str")

    component_ids = list(primary_cfg.get("components", []))
    component_dfs: dict[str, pd.DataFrame] = {}
    for cid in component_ids:
        ccfg = next((c for c in strat_cfgs if c["id"] == cid), None)
        cdf = load_trade_log(ccfg["trade_log"]) if ccfg else None
        if cdf is None or "date" not in cdf.columns:
            continue
        tmp = cdf.copy()
        tmp["_date_str"] = tmp["date"].apply(
            lambda x: str(pd.Timestamp(x).date())
        )
        tmp = tmp.drop_duplicates(subset=["_date_str"], keep="last")
        component_dfs[cid] = tmp.set_index("_date_str")

    sorted_dates = list(df["_date_str"])[-7:][::-1]
    recent_days: list[dict] = []
    pnl_history: list[float] = []
    for d in sorted_dates:
        row = df[df["_date_str"] == d].iloc[-1]
        pnl = float(row.get("pnl", 0.0))
        n_active = 0
        for cid, cdf in component_dfs.items():
            if d in cdf.index:
                pos = float(cdf.loc[d].get("position", 0.0) or 0.0)
                if abs(pos) > 0.01:
                    n_active += 1
        pnl_history.append(pnl)
        recent_days.append({
            "date": d[5:],
            "pnl": pnl,
            "n_active": n_active,
            "spark": "",
        })
    return recent_days, pnl_history


def _recent_days_legacy_sum(
    strategies: list[dict],
    strat_cfgs: list[dict],
    load_trade_log: Callable[[str], pd.DataFrame | None],
) -> tuple[list[dict], list[float]]:
    all_dates: set[str] = set()
    strat_pnl_by_date: dict[str, dict[str, float]] = {}

    for s in strategies:
        if not s["has_data"]:
            continue
        cfg = next((c for c in strat_cfgs if c["id"] == s["id"]), None)
        df = load_trade_log(cfg["trade_log"]) if cfg else None
        if df is None:
            continue

        df_tail = df.tail(60).copy()
        df_tail["_date_str"] = df_tail["date"].apply(
            lambda x: str(pd.Timestamp(x).date())
        )
        df_tail = df_tail.drop_duplicates(subset=["_date_str"], keep="last")
        for _, row in df_tail.iterrows():
            d = row["_date_str"]
            all_dates.add(d)
            if d not in strat_pnl_by_date:
                strat_pnl_by_date[d] = {"pnl": 0.0, "n_active": 0}
            strat_pnl_by_date[d]["pnl"] += row["pnl"]
            if abs(row.get("position", 0)) > 0.01:
                strat_pnl_by_date[d]["n_active"] += 1

    sorted_dates = sorted(all_dates, reverse=True)[:7]
    recent_days = []
    pnl_history = []
    for d in sorted_dates:
        info = strat_pnl_by_date.get(d, {"pnl": 0, "n_active": 0})
        pnl_history.append(info["pnl"])
        recent_days.append({
            "date": d[5:],
            "pnl": info["pnl"],
            "n_active": info["n_active"],
            "spark": "",
        })
    return recent_days, pnl_history


def build_ledger(strategies: list[dict], strat_cfgs: list[dict],
                 load_trade_log: Callable[[str], pd.DataFrame | None],
                 since_date: str | None = None,
                 n_days: int = 30,
                 primary_portfolio_id: str | None = None) -> list[dict]:
    """Build per-strategy ledger for the Live page.

    Args:
        since_date: ISO date string ("2026-03-01"). If provided, show every
            date from that day forward — this is the correct semantic when
            paper trading went live on a known date. Configured via
            strategies.yaml `settings.paper_trading_start`.
        n_days: fallback when since_date is None — show last N trading days.
        primary_portfolio_id: when set AND that strategy has a trade log,
            ``total_pnl`` is taken from the primary portfolio's own daily
            ``pnl`` column (already weighted across components) rather than
            naively summed across every strategy's daily PnL%, which has no
            financial meaning. SSOT-aligned with hero metrics.

    Each entry: {date, total_pnl, entries: [{name, color, position, pnl,
    cum_pnl, action, action_class}, ...]}. Flat rows with no pnl are skipped
    to reduce noise. Dedups (strategy, date), preferring live over backfill.
    """
    primary_pnl_by_date: dict[str, float] = {}
    if primary_portfolio_id is not None:
        pcfg = next(
            (c for c in strat_cfgs if c["id"] == primary_portfolio_id), None,
        )
        pdf = load_trade_log(pcfg["trade_log"]) if pcfg else None
        if pdf is not None and "date" in pdf.columns:
            tmp = pdf.copy()
            tmp["_date_str"] = tmp["date"].apply(
                lambda x: str(pd.Timestamp(x).date())
            )
            if "source" in tmp.columns:
                tmp["_src_rank"] = (
                    tmp["source"].map({"live": 1, "backfill": 0}).fillna(0)
                )
                tmp = tmp.sort_values(["_date_str", "_src_rank"])
            tmp = tmp.drop_duplicates(subset=["_date_str"], keep="last")
            primary_pnl_by_date = dict(zip(tmp["_date_str"], tmp["pnl"]))
    strat_rows: dict[str, dict] = {}
    for s in strategies:
        if not s["has_data"]:
            continue
        cfg = next((c for c in strat_cfgs if c["id"] == s["id"]), None)
        if not cfg:
            continue
        df = load_trade_log(cfg["trade_log"])
        if df is None:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["_date_str"] = df["date"].dt.strftime("%Y-%m-%d")
        if "source" in df.columns:
            df["_src_rank"] = (
                df["source"].map({"live": 1, "backfill": 0}).fillna(0)
            )
            df = df.sort_values(["_date_str", "_src_rank"])
        df = (
            df.drop_duplicates(subset=["_date_str"], keep="last")
            .sort_values("date")
        )
        if since_date is not None:
            cutoff = pd.Timestamp(since_date)
            recent = df[df["date"] >= cutoff].copy().reset_index(drop=True)
        else:
            recent = df.tail(max(n_days * 2, 60)).copy().reset_index(drop=True)
        if len(recent) == 0:
            continue
        # live_cum uses the same simple-return compounding as trade_log
        # cum_return, so the ledger "Cum" column matches persisted PnL.
        recent["live_cum"] = (1.0 + recent["pnl"].astype(float)).cumprod() - 1.0
        strat_rows[s["id"]] = {
            "name": s["name"],
            "color": s["color"],
            "df": recent,
        }

    all_dates: set[str] = set()
    for info in strat_rows.values():
        all_dates.update(str(d.date()) for d in info["df"]["date"])

    ledger = []
    sorted_dates = sorted(all_dates, reverse=True)
    if since_date is None:
        sorted_dates = sorted_dates[:n_days]
    for date_str in sorted_dates:
        entries = []
        total_pnl_day = 0.0
        for _sid, info in strat_rows.items():
            df = info["df"]
            day_rows = df[df["date"].dt.strftime("%Y-%m-%d") == date_str]
            if len(day_rows) == 0:
                continue
            row = day_rows.iloc[-1]
            pos = float(row["position"])
            pnl_val = float(row["pnl"])
            cum = float(row["live_cum"])
            total_pnl_day += pnl_val

            idx = df.index.get_loc(day_rows.index[-1])
            prev_pos = float(df.iloc[idx - 1]["position"]) if idx > 0 else 0.0

            if abs(pos) < 0.01 and abs(prev_pos) < 0.01:
                action, action_class = "—", "ledger-action-flat"
            elif abs(prev_pos) < 0.01 and abs(pos) > 0.01:
                action, action_class = "→ LONG", "ledger-action-change"
            elif abs(prev_pos) > 0.01 and abs(pos) < 0.01:
                action, action_class = "→ EXIT", "ledger-action-change"
            elif abs(pos - prev_pos) > 0.01:
                direction = "↑" if pos > prev_pos else "↓"
                action = f"{direction} {pos:.2f}"
                action_class = "ledger-action-change"
            else:
                action, action_class = "= hold", "ledger-action-long"

            # Skip rows that are truly idle: flat today AND flat yesterday AND
            # no pnl. This keeps HOLD, OPEN, EXIT, and ADJUST rows visible —
            # an exit (pos 0.07 → 0) would otherwise be hidden and look like
            # "never had a position", which is worse than showing a zero row.
            changed = abs(pos - prev_pos) > 0.01
            if abs(pos) < 0.01 and abs(pnl_val) < 1e-8 and not changed:
                continue

            entries.append({
                "name": info["name"],
                "color": info["color"],
                "position": pos,
                "pnl": pnl_val,
                "cum_pnl": cum,
                "action": action,
                "action_class": action_class,
            })

        entries.sort(key=lambda x: abs(x["pnl"]), reverse=True)
        if entries:
            day_total = (
                float(primary_pnl_by_date[date_str])
                if date_str in primary_pnl_by_date
                else total_pnl_day
            )
            ledger.append({
                "date": date_str,
                "total_pnl": day_total,
                "entries": entries,
            })
    return ledger
