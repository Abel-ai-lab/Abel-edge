"""Strategy execution orchestrator. Iterates strategies.yaml, calls engines."""

from __future__ import annotations

import click
import pandas as pd

from abel_edge.engine.backtest import BacktestSettings, run_backtest
from abel_edge.engine.feed_contract import FeedContractError
from abel_edge.engine.ledger import write_trade_log
from abel_edge.engine.loader import load_engine_from_import_path
from abel_edge.engine.paper_rows import append_paper_decision_rows
from abel_edge.engine.runtime_contract import DecisionContractError
from abel_edge.engine.signal_contract import SignalContractError


def _load_engine(engine_path: str):
    """Import engine module and find the StrategyEngine subclass."""
    return load_engine_from_import_path(engine_path)


def _compute_runtime_output(engine, strategy_cfg: dict):
    try:
        return engine.compute_runtime_output()
    except (DecisionContractError, FeedContractError, SignalContractError, TypeError, ValueError) as exc:
        raise click.ClickException(f"{engine.__class__.__name__}: {exc}") from exc


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

    compiled = _compute_runtime_output(engine, strategy_cfg)
    dates = compiled.decision_index
    prices = compiled.close_prices

    execution_cfg = (settings or {}).get("execution") or {}
    input_semantics = (
        "next_position" if compiled.output_mode == "decision_context" else "effective_position"
    )
    result = run_backtest(
        compiled.next_position if input_semantics == "next_position" else compiled.positions,
        prices,
        dates=dates,
        settings=BacktestSettings(
            cost_bps=float(execution_cfg.get("cost_bps", 0.0) or 0.0),
            max_abs_position=execution_cfg.get("max_abs_position"),
        ),
        input_semantics=input_semantics,
        execution_delay_bars=compiled.runtime_profile.execution_delay_bars,
    )

    write_trade_log(
        pd.DatetimeIndex(dates),
        result["asset_returns"],
        result["pnl"],
        result["positions"],
        trade_log_path,
        decision_times=result.get("decision_time"),
        effective_times=result.get("effective_time"),
        close_prices=prices,
        next_positions=result.get("next_position"),
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


def paper_run_one(
    strategy_cfg: dict,
    *,
    settings: dict | None = None,
    as_of=None,
) -> dict:
    sid = strategy_cfg["id"]
    engine_path = strategy_cfg["engine"]

    click.echo(f"  Paper trading {sid}...")
    engine_cls = _load_engine(engine_path)
    engine = engine_cls(context=strategy_cfg)

    compiled = _compute_runtime_output(engine, strategy_cfg)
    dates = compiled.decision_index
    prices = compiled.close_prices
    try:
        return append_paper_decision_rows(
            strategy_cfg,
            dates=dates,
            prices=prices,
            positions=compiled.positions,
            next_positions=compiled.next_position,
            as_of=as_of,
            signal_lookup=lambda ts: _ensure_paper_signal(engine, as_of=ts),
        )
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e)) from e


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
        click.echo(f"    → {result['n_days']} days written to {result['trade_log']}")

    return results


def paper_run_all(config: dict, strategy_id: str | None = None, as_of=None) -> list[dict]:
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
