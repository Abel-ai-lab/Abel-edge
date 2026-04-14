"""Live overview aggregations for the dashboard overview page."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from causal_edge.dashboard.components import compute_metrics


def _load_live_rows(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None
    if len(df) == 0:
        return None
    live = df.copy()
    if "source" in live.columns:
        live = live[live["source"].astype(str).str.lower() == "live"].copy()
    else:
        live["source"] = "live"
    if len(live) == 0:
        return None
    live = live.sort_values("date").copy()
    live["_date_str"] = live["date"].dt.strftime("%Y-%m-%d")
    return live.drop_duplicates(subset=["_date_str"], keep="last").reset_index(drop=True)


def _signal_position(row: pd.Series) -> float:
    if "next_position" in row and not pd.isna(row["next_position"]):
        return float(row["next_position"])
    return float(row.get("position", 0.0))


def _sparkline(values: list[float], width: int = 7) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    if not values or all(abs(v) < 1e-12 for v in values):
        return "▁" * width
    minimum, maximum = min(values), max(values)
    scale = maximum - minimum if maximum > minimum else 1.0
    return "".join(
        blocks[min(8, max(1, int((value - minimum) / scale * 7) + 1))]
        for value in values[-width:]
    )


def _ledger_action(current: float, previous: float) -> tuple[str, str]:
    if abs(current) < 0.01 and abs(previous) < 0.01:
        return "Flat", "muted"
    if abs(previous) < 0.01 and abs(current) >= 0.01:
        return "New", "pnl-pos"
    if abs(previous) >= 0.01 and abs(current) < 0.01:
        return "Exit", "pnl-neg"
    if abs(current - previous) > 0.01:
        return ("Raise" if current > previous else "Trim"), "muted"
    return "Hold", "muted"


def build_live_overview(
    strategies: list[dict], strat_cfgs: list[dict], settings: dict
) -> dict[str, object]:
    """Aggregate paper-trading rows into overview-friendly live summaries."""
    tracked: list[dict[str, object]] = []
    since_raw = settings.get("paper_trading_start")
    since_ts = pd.Timestamp(since_raw) if since_raw is not None else None
    since_label = since_ts.date().isoformat() if since_ts is not None else None
    ledger_days = int(settings.get("ledger_days", 30))
    active_date_counts: dict[str, int] = {}
    pnl_by_date: dict[str, float] = {}

    for strategy in strategies:
        cfg = next((item for item in strat_cfgs if item["id"] == strategy["id"]), None)
        live_df = _load_live_rows(cfg.get("paper_log") if cfg else None)
        if live_df is None:
            continue
        if since_ts is not None:
            live_df = live_df[live_df["date"] >= since_ts].copy()
            if len(live_df) == 0:
                continue
        live_df["signal_position"] = live_df.apply(_signal_position, axis=1)
        live_df["live_cum_return"] = (1.0 + live_df["pnl"].astype(float)).cumprod() - 1.0
        tracked.append({"strategy": strategy, "df": live_df})

        for _, row in live_df.iterrows():
            date_str = row["_date_str"]
            pnl_by_date[date_str] = pnl_by_date.get(date_str, 0.0) + float(row["pnl"])
            if abs(float(row["signal_position"])) >= 0.01:
                active_date_counts[date_str] = active_date_counts.get(date_str, 0) + 1

    if not tracked:
        return {
            "has_live_data": False,
            "live_cards": [],
            "recent_days": [],
            "live_perf": [],
            "ledger": [],
            "signals_active": [],
            "n_active": 0,
            "n_flat": 0,
        }

    today_pnl = 0.0
    total_pnl = 0.0
    signals_active: list[dict[str, object]] = []
    n_flat = 0
    latest_dates: list[pd.Timestamp] = []
    earliest_dates: list[pd.Timestamp] = []
    live_perf: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []

    now = pd.Timestamp(datetime.now())
    current_month = now.month
    current_year = now.year

    for item in tracked:
        strategy = item["strategy"]
        live_df = item["df"].copy()
        latest = live_df.iloc[-1]
        previous_signal = (
            float(live_df.iloc[-2]["signal_position"]) if len(live_df) > 1 else 0.0
        )
        current_signal = float(latest["signal_position"])
        today_pnl += float(latest["pnl"])
        total_pnl += float(live_df["pnl"].sum())
        latest_dates.append(pd.Timestamp(latest["date"]))
        earliest_dates.append(pd.Timestamp(live_df.iloc[0]["date"]))

        if abs(current_signal) >= 0.01:
            signals_active.append(
                {
                    "id": strategy["id"],
                    "name": strategy["name"],
                    "color": strategy["color"],
                    "position": current_signal,
                    "today_pnl": float(latest["pnl"]),
                    "changed": abs(current_signal - previous_signal) > 0.01,
                }
            )
        else:
            n_flat += 1

        metrics = compute_metrics(live_df["pnl"].values.astype(float))
        live_perf.append(
            {
                "id": strategy["id"],
                "name": strategy["name"],
                "color": strategy["color"],
                "days": metrics["n_days"],
                "active_days": int((live_df["signal_position"].abs() >= 0.01).sum()),
                "sharpe": metrics["sharpe"],
                "pnl": metrics["cum_return"],
                "max_dd": metrics["max_dd"],
                "win_rate": metrics["win_rate"],
            }
        )

        for idx, row in live_df.iterrows():
            previous = float(live_df.iloc[idx - 1]["signal_position"]) if idx > 0 else 0.0
            action, action_class = _ledger_action(float(row["signal_position"]), previous)
            ledger_rows.append(
                {
                    "date": row["_date_str"],
                    "name": strategy["name"],
                    "color": strategy["color"],
                    "action": action,
                    "action_class": action_class,
                    "position": float(row["signal_position"]),
                    "pnl": float(row["pnl"]),
                    "cum_pnl": float(row["live_cum_return"]),
                }
            )

    signals_active.sort(key=lambda item: abs(float(item["position"])), reverse=True)
    live_perf.sort(key=lambda item: float(item["pnl"]), reverse=True)

    sorted_dates = sorted(pnl_by_date.keys(), reverse=True)
    recent_days = []
    for date_str in sorted_dates[:7]:
        recent_days.append(
            {
                "date": date_str,
                "pnl": pnl_by_date[date_str],
                "n_active": active_date_counts.get(date_str, 0),
            }
        )
    if recent_days:
        spark = _sparkline([day["pnl"] for day in reversed(recent_days)])
        recent_days[0]["spark"] = spark
        for day in recent_days[1:]:
            day["spark"] = ""

    if since_label is not None:
        ledger_dates = [date for date in sorted_dates if date >= since_label]
    else:
        ledger_dates = sorted_dates[:ledger_days]
    ledger = []
    for date_str in ledger_dates:
        entries = [row for row in ledger_rows if row["date"] == date_str]
        entries.sort(key=lambda row: abs(float(row["pnl"])), reverse=True)
        if entries:
            ledger.append(
                {
                    "date": date_str,
                    "total_pnl": sum(float(row["pnl"]) for row in entries),
                    "entries": entries,
                }
            )

    mtd_pnl = 0.0
    for item in tracked:
        live_df = item["df"]
        mask = (live_df["date"].dt.year == current_year) & (
            live_df["date"].dt.month == current_month
        )
        mtd_pnl += float(live_df.loc[mask, "pnl"].sum())

    most_recent = max(latest_dates)
    stale_hours = max((now - most_recent).total_seconds() / 3600, 0.0)
    live_since = since_label or min(earliest_dates).date().isoformat()

    return {
        "has_live_data": True,
        "live_cards": [
            {"label": "Live Return", "value": f"{total_pnl:+.2%}", "subtext": f"Since {live_since}"},
            {"label": "Latest Day", "value": f"{today_pnl:+.2%}", "subtext": most_recent.date().isoformat()},
            {
                "label": "Active Signals",
                "value": f"{len(signals_active)}/{len(tracked)}",
                "subtext": "Using current live signal size",
            },
            {"label": "Updated", "value": most_recent.date().isoformat(), "subtext": f"{stale_hours:.0f}h old"},
        ],
        "recent_days": recent_days,
        "live_perf": live_perf,
        "ledger": ledger,
        "signals_active": signals_active,
        "n_active": len(signals_active),
        "n_flat": n_flat,
        "mtd_pnl": mtd_pnl,
        "today_pnl": today_pnl,
        "total_pnl": total_pnl,
        "live_since": live_since,
        "stale_hours": stale_hours,
    }
