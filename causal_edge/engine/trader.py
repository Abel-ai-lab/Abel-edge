"""Strategy execution orchestrator. Iterates strategies.yaml, calls engines."""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import pandas as pd


def _make_pit_snapshot_id(sid: str, as_of: str | None = None) -> str:
    """Generate a PIT snapshot id for this paper-run invocation.

    Format: <date>_<sid>_<run_id_short>. Deterministic per (date, sid, run_id).
    Stored in cache/pit/<this_id>/<ticker>.csv by lib/data.py when env var
    ABEL_PIT_RECORD_SNAPSHOT_ID is set.

    The data layer (e.g. trading-internal/lib/data.py) is responsible for
    actually creating the snapshot dir + writing files. causal_edge only
    SIGNALS via env var that snapshotting is desired.
    """
    if as_of:
        date_part = pd.to_datetime(as_of, utc=True).strftime("%Y-%m-%d")
    else:
        date_part = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_id_env = os.environ.get("CAUSAL_EDGE_RUN_ID", "")
    run_id_short = run_id_env.split("-")[-1][:8] if run_id_env else "manual"
    return f"{date_part}_{sid}_{run_id_short}"

from causal_edge.engine.backtest import BacktestSettings, run_backtest
from causal_edge.engine.feed_contract import FeedContractError
from causal_edge.engine.ledger import append_trade_log_rows, read_trade_log, write_trade_log
from causal_edge.engine.loader import load_engine_from_import_path
from causal_edge.engine.signal_contract import SignalContractError
from causal_edge.engine.signal_contract import validate_signal_output


def _load_engine(engine_path: str):
    """Import engine module and find the StrategyEngine subclass."""
    return load_engine_from_import_path(engine_path)


def _compute_validated_signals(engine, strategy_cfg: dict):
    profile = ((strategy_cfg.get("_data_contract") or {}).get("profile", "daily"))
    try:
        raw_output = engine.compute_signals()
        return validate_signal_output(*raw_output, profile=profile)
    except (FeedContractError, SignalContractError) as exc:
        raise click.ClickException(f"{engine.__class__.__name__}: {exc}") from exc


def _canonical_config_hash(strategy_cfg: dict, settings: dict | None) -> str:
    payload = {
        "settings": settings or {},
        "strategy": strategy_cfg,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _data_version(dates: pd.DatetimeIndex, prices: np.ndarray, idx: int) -> str:
    idx = int(idx)
    cutoff_dates = pd.DatetimeIndex(dates[: idx + 1]).astype("int64").to_numpy()
    cutoff_prices = np.asarray(prices[: idx + 1], dtype=np.float64)
    h = hashlib.sha256()
    h.update(str(idx + 1).encode())
    h.update(b"\0dates\0")
    h.update(cutoff_dates.tobytes())
    h.update(b"\0prices\0")
    h.update(cutoff_prices.tobytes())
    return h.hexdigest()


def run_one(strategy_cfg: dict, *, settings: dict | None = None) -> dict:
    """Run a single strategy and write its trade log.

    Args:
        strategy_cfg: Strategy dict from strategies.yaml

    Returns:
        dict with keys: id, n_days, trade_log
    """
    sid = strategy_cfg["id"]
    engine_path = strategy_cfg["engine"]
    trade_log_path = strategy_cfg["trade_log"]

    click.echo(f"  Running {sid}...")
    engine_cls = _load_engine(engine_path)
    engine = engine_cls(context=strategy_cfg)

    positions, dates, prices = _compute_validated_signals(engine, strategy_cfg)
    if len(dates) == 0:
        log_path = Path(trade_log_path)
        if log_path.exists():
            try:
                existing = read_trade_log(log_path)
            except Exception:  # noqa: BLE001
                existing = pd.DataFrame()
            if len(existing) > 0:
                log_path.touch()
                return {
                    "id": sid,
                    "n_days": len(existing),
                    "trade_log": trade_log_path,
                    "skipped": True,
                    "reason": "empty signal output; preserved existing trade log",
                }
        return {
            "id": sid,
            "n_days": 0,
            "trade_log": trade_log_path,
            "skipped": True,
            "reason": "empty signal output",
        }

    execution_cfg = (settings or {}).get("execution") or {}
    result = run_backtest(
        positions,
        prices,
        settings=BacktestSettings(
            cost_bps=float(execution_cfg.get("cost_bps", 0.0) or 0.0),
            max_abs_position=execution_cfg.get("max_abs_position"),
        ),
    )

    write_trade_log(
        pd.DatetimeIndex(dates),
        result["asset_returns"],
        result["pnl"],
        result["positions"],
        trade_log_path,
        close_prices=prices,
        gross_pnl=result["gross_pnl"],
        turnover=result["turnover"],
        execution_cost=result["execution_cost"],
    )

    return {"id": sid, "n_days": len(dates), "trade_log": trade_log_path}


def _ensure_paper_signal(engine, *, as_of):
    try:
        signal = engine.get_paper_signal(as_of=as_of)
    except NotImplementedError as e:
        raise click.ClickException(str(e)) from e
    if "next_position" not in signal:
        raise click.ClickException(
            f"{engine.__class__.__name__}.get_paper_signal(as_of=...) must return 'next_position'."
        )
    return signal


def _resolve_last_logged_row(trade_log_path: str):
    try:
        log_df = read_trade_log(trade_log_path)
    except FileNotFoundError as e:
        raise click.ClickException(
            f"Trade log not found for paper trading: {trade_log_path}. Run 'causal-edge run' first."
        ) from e

    if len(log_df) == 0:
        raise click.ClickException(
            f"Trade log is empty for paper trading: {trade_log_path}. Run 'causal-edge run' first."
        )

    log_df = log_df.sort_values("date").reset_index(drop=True)
    return log_df, log_df.iloc[-1]


def _paper_log_path(strategy_cfg: dict) -> str:
    return strategy_cfg.get("paper_log") or strategy_cfg["trade_log"]


def _resolve_paper_state(strategy_cfg: dict):
    trade_log_path = strategy_cfg["trade_log"]
    paper_log_path = _paper_log_path(strategy_cfg)

    if paper_log_path != trade_log_path:
        try:
            paper_df, paper_last_row = _resolve_last_logged_row(paper_log_path)
            return paper_log_path, paper_df, paper_last_row
        except click.ClickException:
            pass

    trade_df, trade_last_row = _resolve_last_logged_row(trade_log_path)
    return paper_log_path, trade_df, trade_last_row


def paper_run_one(
    strategy_cfg: dict,
    *,
    settings: dict | None = None,
    as_of=None,
) -> dict:
    sid = strategy_cfg["id"]
    engine_path = strategy_cfg["engine"]
    paper_log_path = _paper_log_path(strategy_cfg)

    click.echo(f"  Paper trading {sid}...")
    engine_cls = _load_engine(engine_path)
    engine = engine_cls(context=strategy_cfg)

    # 2026-05-12 PIT A3: wrap compute_signals in RECORD mode. The data
    # layer (lib/data.py) detects ABEL_PIT_RECORD_SNAPSHOT_ID and writes
    # a snapshot of each fetched ticker to cache/pit/<sid>/. Resulting
    # snapshot_id is persisted into each live row's pit_snapshot_id column.
    pit_snapshot_id = _make_pit_snapshot_id(sid, as_of=as_of)
    os.environ["ABEL_PIT_RECORD_SNAPSHOT_ID"] = pit_snapshot_id
    try:
        positions, dates, prices = _compute_validated_signals(engine, strategy_cfg)
    finally:
        # Always clear env var to avoid leaking RECORD mode to subsequent
        # strategy runs OR research scripts.
        os.environ.pop("ABEL_PIT_RECORD_SNAPSHOT_ID", None)
    dates = pd.DatetimeIndex(dates)
    if as_of is not None:
        cutoff = pd.to_datetime(as_of, utc=True)
        mask = dates <= cutoff
        positions = positions[mask]
        prices = prices[mask]
        dates = dates[mask]

    if len(dates) == 0:
        try:
            _, _, last_row = _resolve_paper_state(strategy_cfg)
        except click.ClickException as exc:
            raise click.ClickException(f"No bars available for strategy '{sid}'.") from exc
        return {
            "id": sid,
            "n_rows": 0,
            "trade_log": paper_log_path,
            "last_date": str(pd.to_datetime(last_row["date"], utc=True).date()),
            "skipped": True,
            "reason": "empty signal output; preserved existing paper log",
        }

    returns = np.zeros_like(prices, dtype=float)
    if len(prices) > 1:
        returns[1:] = prices[1:] / prices[:-1] - 1.0

    _, paper_df, last_row = _resolve_paper_state(strategy_cfg)
    last_logged_date = pd.to_datetime(last_row["date"], utc=True)
    new_mask = dates > last_logged_date
    new_dates = dates[new_mask]
    if len(new_dates) == 0:
        return {
            "id": sid,
            "n_rows": 0,
            "trade_log": paper_log_path,
            "last_date": str(last_logged_date),
        }

    if "next_position" in paper_df.columns and pd.notna(last_row.get("next_position")):
        carry_position = float(last_row["next_position"])
    else:
        bootstrap = _ensure_paper_signal(engine, as_of=last_logged_date)
        carry_position = float(bootstrap["next_position"])

    rows = []
    date_to_index = {ts: idx for idx, ts in enumerate(dates)}
    config_hash = _canonical_config_hash(strategy_cfg, settings)
    for ts in new_dates:
        idx = date_to_index[ts]
        signal = _ensure_paper_signal(engine, as_of=ts)
        next_position = float(signal["next_position"])
        rows.append(
            {
                "date": ts,
                "close": float(prices[idx]),
                "asset_return": float(returns[idx]),
                "position": carry_position,
                "pnl": float(carry_position * returns[idx]),
                "next_position": next_position,
                "source": "live",
                "live_origin": "cron_live",
                "config_hash": config_hash,
                "data_version": _data_version(dates, prices, idx),
                # PIT A3: persists snapshot id so engine_purity can later
                # reproduce this row bit-exact via PIT data.
                "pit_snapshot_id": pit_snapshot_id,
            }
        )
        carry_position = next_position

    append_trade_log_rows(paper_log_path, rows)
    return {
        "id": sid,
        "n_rows": len(rows),
        "trade_log": paper_log_path,
        "last_date": str(new_dates[-1].date()),
    }


def run_all(config: dict, strategy_id: str | None = None) -> list[dict]:
    """Run all strategies (or one specific strategy) from config.

    Args:
        config: Loaded config dict from load_config()
        strategy_id: If set, run only this strategy

    Returns:
        List of result dicts from run_one()
    """
    strategies = config["strategies"]
    if strategy_id:
        strategies = [s for s in strategies if s["id"] == strategy_id]
        if not strategies:
            raise ValueError(
                f"Strategy '{strategy_id}' not found in strategies.yaml. "
                f"Available: {[s['id'] for s in config['strategies']]}"
            )

    results = []
    for s_cfg in strategies:
        result = run_one(s_cfg, settings=config.get("settings"))
        results.append(result)
        if result.get("skipped"):
            click.echo(
                f"    → skipped {result['id']}: {result.get('reason')} "
                f"({result['n_days']} existing rows at {result['trade_log']})"
            )
        else:
            click.echo(f"    → {result['n_days']} days written to {result['trade_log']}")

    return results


def paper_run_all(config: dict, strategy_id: str | None = None, as_of=None) -> list[dict]:
    """Run paper-trade emission for all (or one) strategies.

    Batch resilience policy (added 2026-05-19, codex peer-reviewed):
      - Single-strategy mode (`strategy_id` set): fail-hard on first exception.
        Caller (operator / `cli paper --strategy X`) wants the trace.
      - All-mode (`strategy_id` is None): each strategy runs in isolation;
        a single failure does NOT halt subsequent strategies. Failures are
        appended to `data/paper_failures.jsonl` (one JSON-line per failure
        with id, error_type, message, traceback, timestamp). After all
        strategies have been attempted, raises `click.ClickException` if any
        failed, so cron exits non-zero and alerting fires.

    Rationale: prior behavior had one stale-ticker exception (RELI in
    hood_xasset_stab) silently killing all downstream strategies in the same
    cron batch (5/14-5/18 lost rows for abel_portfolio family + sp_recon_v2
    + eth_call_overlay). Batch should be resilient; cron should still be noisy.
    """
    strategies = config["strategies"]
    single_strategy_mode = strategy_id is not None
    if strategy_id:
        strategies = [s for s in strategies if s["id"] == strategy_id]
        if not strategies:
            raise ValueError(
                f"Strategy '{strategy_id}' not found in strategies.yaml. "
                f"Available: {[s['id'] for s in config['strategies']]}"
            )

    results: list[dict] = []
    failures: list[dict] = []
    for s_cfg in strategies:
        sid = s_cfg.get("id", "<unknown>")
        try:
            result = paper_run_one(
                s_cfg,
                settings=config.get("settings"),
                as_of=as_of,
            )
        except Exception as exc:
            if single_strategy_mode:
                raise
            failures.append({
                "strategy": sid,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            click.echo(f"    → FAILED {sid}: {type(exc).__name__}: {exc}")
            continue

        results.append(result)
        if result["n_rows"] == 0:
            click.echo(f"    → no new closed bars for {result['id']}")
        else:
            click.echo(
                f"    → appended {result['n_rows']} live rows to {result['trade_log']} "
                f"through {result['last_date']}"
            )

    if failures:
        _write_paper_failures(failures)
        raise click.ClickException(
            f"{len(failures)} strategies failed during paper batch; "
            f"see data/paper_failures.jsonl. IDs: "
            f"{[f['strategy'] for f in failures]}"
        )

    return results


def _write_paper_failures(failures: list[dict]) -> None:
    """Append failure records to data/paper_failures.jsonl (one JSON per line)."""
    path = Path("data/paper_failures.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for rec in failures:
            f.write(json.dumps(rec) + "\n")
