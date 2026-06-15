"""Generic paper row construction and latest-state helpers."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from abel_edge.engine.ledger import (
    append_trade_log_rows,
    compound_cum_return,
    read_trade_log,
)


RESERVED_PAPER_ROW_FIELDS = {
    "date",
    "decision_time",
    "effective_time",
    "close",
    "asset_return",
    "position",
    "pnl",
    "next_position",
    "source",
    "cum_return",
    "gross_pnl",
    "turnover",
    "execution_cost",
}


def _paper_log_path(strategy_cfg: dict) -> str:
    return strategy_cfg.get("paper_log") or strategy_cfg["trade_log"]


def _resolve_repo_path(path: str | Path, repo_root: str | Path | None) -> Path:
    resolved = Path(path)
    if resolved.is_absolute() or repo_root is None:
        return resolved
    return Path(repo_root) / resolved


def _read_last_logged_row(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trade log not found for paper trading: {path}. Run 'abel-edge run' first."
        )

    log_df = read_trade_log(path)
    if len(log_df) == 0:
        raise ValueError(
            f"Trade log is empty for paper trading: {path}. Run 'abel-edge run' first."
        )

    log_df = log_df.sort_values("date").reset_index(drop=True)
    return log_df, log_df.iloc[-1]


def resolve_paper_state(
    strategy_cfg: dict,
    *,
    state_log_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    trade_log_path = _resolve_repo_path(strategy_cfg["trade_log"], repo_root)
    paper_log_path = _resolve_repo_path(_paper_log_path(strategy_cfg), repo_root)

    candidates = []
    if paper_log_path != trade_log_path:
        candidates.append(paper_log_path)
    if state_log_path is not None:
        candidates.append(state_log_path)
    candidates.append(trade_log_path)

    seen = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        try:
            paper_df, paper_last_row = _read_last_logged_row(candidate)
            return paper_df, paper_last_row
        except (FileNotFoundError, ValueError):
            pass

    trade_df, trade_last_row = _read_last_logged_row(trade_log_path)
    return trade_df, trade_last_row


def paper_signal_extra_fields(signal: dict) -> dict:
    extras = {}
    for key, value in signal.items():
        if key in RESERVED_PAPER_ROW_FIELDS:
            continue
        if isinstance(value, pd.Timestamp):
            extras[key] = value.isoformat()
        elif isinstance(value, datetime):
            extras[key] = value.isoformat()
        elif isinstance(value, date):
            extras[key] = value.isoformat()
        elif isinstance(value, np.datetime64):
            extras[key] = pd.Timestamp(value).isoformat()
        elif isinstance(value, np.generic):
            native_value = value.item()
            if native_value is None or isinstance(native_value, (str, int, float, bool)):
                extras[key] = native_value
        elif value is None or isinstance(value, (str, int, float, bool)):
            extras[key] = value
    return extras


def _to_timestamp(value) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value, utc=True)


def _to_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _safe_scalar(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def _format_result_last_date(value) -> str | None:
    ts = _to_timestamp(value)
    if ts is None:
        return None
    return str(ts)


def latest_paper_snapshot(strategy_cfg: dict, row: dict | pd.Series) -> dict:
    row_dict = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    processed_at = _to_timestamp(row_dict.get("date"))
    snapshot = {
        "strategy_id": strategy_cfg["id"],
        "asset": strategy_cfg.get("asset"),
        "last_processed_date": processed_at.isoformat() if processed_at is not None else None,
        "current_position": _to_float(row_dict.get("position")) or 0.0,
        "next_position": _to_float(row_dict.get("next_position", row_dict.get("position"))) or 0.0,
        "latest_close": _to_float(row_dict.get("close")),
        "source": row_dict.get("source"),
    }
    for key, value in row_dict.items():
        if key in RESERVED_PAPER_ROW_FIELDS:
            continue
        scalar = _safe_scalar(value)
        if scalar is not None:
            snapshot[key] = scalar
    return snapshot


def decision_row_from_paper_row(strategy_cfg: dict, row: dict) -> dict:
    processed_at = _to_timestamp(row.get("date"))
    processed_iso = processed_at.isoformat() if processed_at is not None else None
    return {
        "as_of": processed_iso,
        "strategy_id": strategy_cfg["id"],
        "asset": strategy_cfg.get("asset"),
        "data_last_timestamp": row.get("data_latest_timestamp") or processed_iso,
        "asset_close": _to_float(row.get("close")) or 0.0,
        "next_position": _to_float(row.get("next_position", row.get("position"))) or 0.0,
    }


def append_paper_decision_rows(
    strategy_cfg: dict,
    *,
    dates,
    prices,
    positions,
    next_positions,
    as_of=None,
    paper_start=None,
    signal_lookup: Callable[[pd.Timestamp], dict] | None = None,
    state_log_path: str | Path | None = None,
    last_processed_date=None,
    overwrite: bool = False,
    repo_root: str | Path | None = None,
) -> dict:
    sid = strategy_cfg["id"]
    paper_log_path = _paper_log_path(strategy_cfg)
    resolved_paper_log_path = _resolve_repo_path(paper_log_path, repo_root)
    paper_df = pd.DataFrame()
    last_row = None
    if paper_start is None:
        paper_df, last_row = resolve_paper_state(
            strategy_cfg, state_log_path=state_log_path, repo_root=repo_root
        )

    dates = pd.DatetimeIndex(pd.to_datetime(dates, utc=True))
    prices = np.asarray(prices, dtype=float)
    positions = np.asarray(positions, dtype=float)
    next_positions = np.asarray(next_positions, dtype=float)

    if as_of is not None:
        cutoff = pd.to_datetime(as_of, utc=True)
        mask = dates <= cutoff
        dates = dates[mask]
        prices = prices[mask]
        positions = positions[mask]
        next_positions = next_positions[mask]

    if len(dates) == 0:
        raise ValueError(f"No bars available for strategy '{sid}'.")

    returns = np.zeros_like(prices, dtype=float)
    if len(prices) > 1:
        returns[1:] = prices[1:] / prices[:-1] - 1.0

    if paper_start is not None:
        start_ts = pd.to_datetime(paper_start, utc=True)
        start_idx = int(dates.searchsorted(start_ts, side="left"))
        if start_idx >= len(dates):
            raise ValueError(
                f"Paper start {start_ts.isoformat()} is after the latest available bar {dates[-1].isoformat()}."
            )
        selected_indexes = range(start_idx, len(dates))
        carry_position = float(positions[start_idx])
    else:
        assert last_row is not None
        last_logged_date = pd.to_datetime(
            last_processed_date if last_processed_date is not None else last_row["date"],
            utc=True,
        )
        selected_indexes = [idx for idx, ts in enumerate(dates) if ts > last_logged_date]
        if not selected_indexes:
            return {
                "id": sid,
                "n_rows": 0,
                "trade_log": paper_log_path,
                "last_date": _format_result_last_date(last_logged_date),
                "new_dates": [],
                "decision_rows": [],
                "latest_snapshot": latest_paper_snapshot(strategy_cfg, last_row),
            }
        if "next_position" in paper_df.columns and pd.notna(last_row.get("next_position")):
            carry_position = float(last_row["next_position"])
        elif signal_lookup is not None:
            bootstrap = signal_lookup(last_logged_date)
            carry_position = float(bootstrap["next_position"])
        else:
            carry_position = float(last_row.get("position", 0.0))

    rows = []
    decision_rows = []
    for idx in selected_indexes:
        ts = dates[idx]
        signal = signal_lookup(ts) if signal_lookup is not None else {}
        next_position = float(signal.get("next_position", next_positions[idx]))
        row = {
            "date": ts,
            "decision_time": ts,
            "effective_time": ts,
            "close": float(prices[idx]),
            "asset_return": float(returns[idx]),
            "position": carry_position,
            "pnl": float(carry_position * returns[idx]),
            "next_position": next_position,
            "source": "live",
        }
        row.update(paper_signal_extra_fields(signal))
        rows.append(row)
        decision_rows.append(decision_row_from_paper_row(strategy_cfg, row))
        carry_position = next_position

    if overwrite:
        output_path = resolved_paper_log_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.DataFrame(rows)
        combined["date"] = pd.to_datetime(combined["date"], utc=True)
        combined["pnl"] = combined["pnl"].astype(float)
        combined["cum_return"] = compound_cum_return(combined["pnl"].to_numpy(dtype=float))
        combined.to_csv(output_path, index=False)
    else:
        combined = append_trade_log_rows(resolved_paper_log_path, rows)
    latest_row = combined.sort_values("date").iloc[-1]
    return {
        "id": sid,
        "n_rows": len(rows),
        "trade_log": paper_log_path,
        "last_date": _format_result_last_date(latest_row.get("date")),
        "new_dates": [pd.Timestamp(row["date"]) for row in rows],
        "decision_rows": decision_rows,
        "latest_snapshot": latest_paper_snapshot(strategy_cfg, latest_row),
    }
