"""Strategy execution orchestrator. Iterates strategies.yaml, calls engines."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

from causal_edge.engine.ledger import append_trade_log_rows, read_trade_log, write_trade_log
from causal_edge.engine.price_data import resolve_price_config


def _load_engine(engine_path: str):
    """Import engine module and find the StrategyEngine subclass."""
    if engine_path.startswith("strategies.") and (Path.cwd() / "strategies").exists():
        stale = [
            name for name in sys.modules if name == "strategies" or name.startswith("strategies.")
        ]
        for name in stale:
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    mod = importlib.import_module(engine_path)
    from causal_edge.engine.base import StrategyEngine

    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, StrategyEngine)
            and attr is not StrategyEngine
        ):
            return attr
    raise ImportError(
        f"No StrategyEngine subclass found in '{engine_path}'. "
        f"Fix: Ensure your engine.py defines a class inheriting StrategyEngine."
    )


def run_one(strategy_cfg: dict, *, settings: dict | None = None, bars_loader=None) -> dict:
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
    if bars_loader is not None:
        engine.bind_price_loader(
            bars_loader,
            resolve_price_config(settings or {}, strategy_cfg),
        )

    positions, dates, prices = engine.compute_signals()

    returns = np.zeros_like(prices, dtype=float)
    if len(prices) > 1:
        returns[1:] = prices[1:] / prices[:-1] - 1.0

    # PnL: positions[t] * returns[t] is correct because the engine contract
    # requires positions[t] to be decided using data through t-1 only.
    # The engine is responsible for applying shift(1) to indicators.
    pnl = positions * returns
    # First day has no prior position signal
    pnl[0] = 0.0

    write_trade_log(dates, returns, pnl, positions, trade_log_path, close_prices=prices)

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
    bars_loader=None,
    as_of=None,
) -> dict:
    sid = strategy_cfg["id"]
    engine_path = strategy_cfg["engine"]
    paper_log_path = _paper_log_path(strategy_cfg)

    click.echo(f"  Paper trading {sid}...")
    engine_cls = _load_engine(engine_path)
    engine = engine_cls(context=strategy_cfg)
    if bars_loader is not None:
        engine.bind_price_loader(
            bars_loader,
            resolve_price_config(settings or {}, strategy_cfg),
        )

    positions, dates, prices = engine.compute_signals()
    dates = pd.DatetimeIndex(dates)
    if as_of is not None:
        cutoff = pd.to_datetime(as_of, utc=True)
        mask = dates <= cutoff
        positions = positions[mask]
        prices = prices[mask]
        dates = dates[mask]

    if len(dates) == 0:
        raise click.ClickException(f"No bars available for strategy '{sid}'.")

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


def run_all(config: dict, strategy_id: str | None = None, bars_loader=None) -> list[dict]:
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
        result = run_one(s_cfg, settings=config.get("settings"), bars_loader=bars_loader)
        results.append(result)
        click.echo(f"    → {result['n_days']} days written to {result['trade_log']}")

    return results


def paper_run_all(
    config: dict, strategy_id: str | None = None, bars_loader=None, as_of=None
) -> list[dict]:
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
        result = paper_run_one(
            s_cfg,
            settings=config.get("settings"),
            bars_loader=bars_loader,
            as_of=as_of,
        )
        results.append(result)
        if result["n_rows"] == 0:
            click.echo(f"    → no new closed bars for {result['id']}")
        else:
            click.echo(
                f"    → appended {result['n_rows']} live rows to {result['trade_log']} "
                f"through {result['last_date']}"
            )

    return results
