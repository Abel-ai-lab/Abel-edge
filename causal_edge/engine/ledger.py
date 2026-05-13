"""Trade log read/write. Single source of truth for trade log CSV format."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("date", "pnl", "position", "cum_return", "source")
CURRENT_SCHEMA_VERSION = "1"
PRE_SCHEMA_VERSION = "0"
PROVENANCE_COLUMNS = (
    "schema_version",
    "captured_at_utc",
    "engine_sha",
    "engine_tree_dirty",
    "config_hash",
    "data_version",
    "run_id",
    "live_origin",
    # 2026-05-12 PIT migration A4: pit_snapshot_id is set by paper_run_one
    # when ABEL_PIT_RECORD_SNAPSHOT_ID env var is honored by the data layer.
    # Existing rows keep "" → treated as "PRE_PIT" by engine_purity.
    # When non-empty, points to cache/pit/<value>/ directory where decision-
    # time data was snapshotted. Engine_purity can re-run compute_signals
    # under ABEL_PIT_SNAPSHOT_ID=<this value> to verify bit-exact match,
    # immune to retroactive FMP revisions.
    "pit_snapshot_id",
)
DEFAULT_LIVE_ORIGIN = "cron_live"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id(now: str | None = None) -> str:
    env_run_id = os.environ.get("CAUSAL_EDGE_RUN_ID")
    if env_run_id:
        return env_run_id
    stamp = (now or _utc_now()).replace("+00:00", "Z").replace(":", "").replace("-", "")
    return f"manual-{stamp}-{os.getpid()}"


def _git_output(args: list[str], *, cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(cwd), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def _engine_sha() -> str:
    return os.environ.get("CAUSAL_EDGE_ENGINE_SHA") or _git_output(["rev-parse", "HEAD"], cwd=Path.cwd())


def _engine_tree_dirty() -> str:
    env_dirty = os.environ.get("CAUSAL_EDGE_ENGINE_TREE_DIRTY")
    if env_dirty is not None:
        return env_dirty
    return "true" if _git_output(["status", "--porcelain", "--untracked-files=no"], cwd=Path.cwd()) else "false"


def _config_hash() -> str:
    env_hash = os.environ.get("CAUSAL_EDGE_CONFIG_HASH")
    if env_hash:
        return env_hash
    config_path = Path.cwd() / "strategies.yaml"
    if not config_path.exists():
        return ""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _row_fallback_hash(row: pd.Series) -> str:
    volatile = set(PROVENANCE_COLUMNS) | {"cum_return"}
    payload = {
        str(k): _json_safe(v)
        for k, v in row.items()
        if str(k) not in volatile
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _ensure_schema_columns(df: pd.DataFrame, *, live_incoming: bool) -> pd.DataFrame:
    out = df.copy()
    for col in PROVENANCE_COLUMNS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype("object")
    if len(out) == 0:
        return out

    if live_incoming:
        now = _utc_now()
        defaults = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "captured_at_utc": now,
            "engine_sha": _engine_sha(),
            "engine_tree_dirty": _engine_tree_dirty(),
            "config_hash": _config_hash(),
            "run_id": _run_id(now),
            "live_origin": os.environ.get("CAUSAL_EDGE_LIVE_ORIGIN", DEFAULT_LIVE_ORIGIN),
        }
        for col, default in defaults.items():
            missing = out[col].isna() | (out[col].astype(str) == "")
            out.loc[missing, col] = default
        missing_data_version = out["data_version"].isna() | (out["data_version"].astype(str) == "")
        if missing_data_version.any():
            out.loc[missing_data_version, "data_version"] = out.loc[missing_data_version].apply(
                _row_fallback_hash,
                axis=1,
            )
    else:
        missing_schema = out["schema_version"].isna() | (out["schema_version"].astype(str) == "")
        out.loc[missing_schema, "schema_version"] = PRE_SCHEMA_VERSION
        for col in PROVENANCE_COLUMNS:
            if col == "schema_version":
                continue
            out[col] = out[col].fillna("")
    return out


def read_trade_log(path: str | Path) -> pd.DataFrame:
    """Read a trade log CSV. Returns DataFrame with standard columns.

    Date parsing uses `format="mixed"` because backfill rows write midnight
    timestamps ("2026-04-16 00:00:00+00:00") while live rows carry
    sub-second ISO-8601 ("2026-04-17T05:55:06.150276+00:00"). The default
    strptime fallback on mixed formats raises, which blocked every
    subsequent `causal-edge run` once any live row existed.
    """
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed")
    return df


def write_trade_log(
    dates: pd.DatetimeIndex,
    asset_returns: np.ndarray,
    pnl: np.ndarray,
    positions: np.ndarray,
    path: str | Path,
    source: str = "backfill",
    close_prices: np.ndarray | None = None,
    next_positions: np.ndarray | None = None,
    gross_pnl: np.ndarray | None = None,
    turnover: np.ndarray | None = None,
    execution_cost: np.ndarray | None = None,
) -> None:
    """Write a trade log CSV from strategy output arrays.

    Args:
        dates: Trading dates
        asset_returns: Daily simple returns of the underlying asset
        pnl: Daily net PnL after execution costs
        positions: Daily position sizes
        path: Output CSV path
        source: "backfill" or "live"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "date": dates,
            "asset_return": asset_returns,
            "pnl": pnl,
            "position": positions,
            "source": source,
        }
    )
    df = _ensure_schema_columns(df, live_incoming=source.lower() == "live")
    if close_prices is not None:
        df["close"] = close_prices
    if next_positions is not None:
        df["next_position"] = next_positions
    if gross_pnl is not None:
        df["gross_pnl"] = gross_pnl
    if turnover is not None:
        df["turnover"] = turnover
    if execution_cost is not None:
        df["execution_cost"] = execution_cost
    if path.exists():
        existing = read_trade_log(path)
        existing = _ensure_schema_columns(existing, live_incoming=False)
        if "source" in existing.columns:
            existing["source"] = existing["source"].fillna("backfill").astype(str)
            existing["date"] = pd.to_datetime(existing["date"], utc=True)
            live_rows = existing[existing["source"].str.lower() == "live"].copy()
            if not live_rows.empty:
                live_rows = _dedupe_trade_rows(live_rows)
                df["date"] = pd.to_datetime(df["date"], utc=True)
                df = pd.concat([df, live_rows], ignore_index=True, sort=False)

    df = _dedupe_trade_rows(df)
    df = _ensure_schema_columns(df, live_incoming=False)
    df["cum_return"] = np.cumprod(1.0 + df["pnl"].to_numpy(dtype=float)) - 1.0
    df.to_csv(path, index=False)


def append_trade_log_rows(path: str | Path, rows: list[dict]) -> pd.DataFrame:
    """Append live paper-trading rows and recompute cumulative return."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    incoming = pd.DataFrame(rows)
    if incoming.empty:
        return read_trade_log(path) if path.exists() else incoming

    incoming["date"] = pd.to_datetime(incoming["date"], utc=True)
    incoming = _ensure_schema_columns(incoming, live_incoming=True)

    if path.exists():
        existing = read_trade_log(path)
        existing = _ensure_schema_columns(existing, live_incoming=False)
    else:
        existing = pd.DataFrame(columns=incoming.columns)

    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    combined = _dedupe_trade_rows(combined)
    combined = _ensure_schema_columns(combined, live_incoming=False)
    combined["pnl"] = combined["pnl"].astype(float)
    combined["cum_return"] = np.cumprod(1.0 + combined["pnl"].to_numpy(dtype=float)) - 1.0
    combined.to_csv(path, index=False)
    return combined.reset_index(drop=True)


def _dedupe_trade_rows(df: pd.DataFrame) -> pd.DataFrame:
    combined = df.copy()
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], utc=True)
    if "source" not in combined.columns:
        combined["source"] = "backfill"
    combined["source"] = combined["source"].fillna("backfill").astype(str)
    combined["_src_rank"] = combined["source"].str.lower().map({"live": 1, "backfill": 0}).fillna(0)
    combined = combined.sort_values(["date", "_src_rank"], kind="mergesort")
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    return combined.drop(columns="_src_rank").reset_index(drop=True)
