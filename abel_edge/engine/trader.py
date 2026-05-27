"""Strategy execution orchestrator. Iterates strategies.yaml, calls engines."""

from __future__ import annotations

import click
import numpy as np
import pandas as pd

from abel_edge.engine.backtest import BacktestSettings, run_backtest
from abel_edge.engine.base import StrategyEngine
from abel_edge.engine.feed_contract import FeedContractError
from abel_edge.engine.ledger import write_trade_log
from abel_edge.engine.loader import load_engine_from_import_path
from abel_edge.engine.paper_rows import append_paper_decision_rows, resolve_paper_state
from abel_edge.engine.runtime_contract import DecisionContractError
from abel_edge.engine.signal_contract import SignalContractError

SYSTEM_LOOKBACK_PADDING_BARS = 20


def _load_engine(engine_path: str):
    """Import engine module and find the StrategyEngine subclass."""
    return load_engine_from_import_path(engine_path)


def _compute_runtime_output(
    engine,
    strategy_cfg: dict,
    *,
    start=None,
    end=None,
    limit: int | None = None,
):
    try:
        return engine.compute_runtime_output(start=start, end=end, limit=limit)
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


def _uses_default_paper_signal(engine) -> bool:
    return type(engine).get_paper_signal is StrategyEngine.get_paper_signal


def _paper_execution_profile(strategy_cfg: dict) -> dict:
    runtime = strategy_cfg.get("runtime")
    profile = runtime.get("paperExecutionProfile") if isinstance(runtime, dict) else None
    if not isinstance(profile, dict) and isinstance(runtime, dict):
        profile = runtime.get("paper_execution_profile")
    if not isinstance(profile, dict):
        profile = strategy_cfg.get("paperExecutionProfile")
    if not isinstance(profile, dict):
        profile = strategy_cfg.get("paper_execution_profile")
    return dict(profile) if isinstance(profile, dict) else {}


def _profile_history(profile: dict) -> dict:
    history = profile.get("history")
    return dict(history) if isinstance(history, dict) else {}


def _resolve_paper_data_window(strategy_cfg: dict) -> dict:
    profile = _paper_execution_profile(strategy_cfg)
    history = _profile_history(profile)
    boundary = str(history.get("boundary") or "").strip()
    if not boundary:
        return {
            "boundary": "origin_anchored",
            "start": None,
            "limit": None,
            "source": "legacy_default",
        }
    if boundary == "fixed_lookback":
        raw_lookback = history.get("lookbackBars", history.get("minBars"))
        try:
            lookback_bars = int(raw_lookback)
        except (TypeError, ValueError) as exc:
            raise click.ClickException(
                "paperExecutionProfile.history.lookbackBars must be a positive integer "
                "for fixed_lookback paper execution."
            ) from exc
        if lookback_bars <= 0:
            raise click.ClickException(
                "paperExecutionProfile.history.lookbackBars must be a positive integer "
                "for fixed_lookback paper execution."
            )
        return {
            "boundary": "fixed_lookback",
            "start": None,
            "limit": lookback_bars + SYSTEM_LOOKBACK_PADDING_BARS,
            "source": "paper_execution_profile",
        }
    if boundary == "origin_anchored":
        return {
            "boundary": "origin_anchored",
            "start": history.get("origin"),
            "limit": None,
            "source": "paper_execution_profile",
        }
    raise click.ClickException(
        "paperExecutionProfile.history.boundary must be 'fixed_lookback' or "
        "'origin_anchored' for paper execution."
    )


def _strategy_cfg_with_paper_window(strategy_cfg: dict, paper_window: dict) -> dict:
    updated = dict(strategy_cfg)
    updated["_paper_data_window"] = dict(paper_window)
    return updated


def _paper_window_audit(paper_window: dict) -> dict:
    audit = {
        "boundary": paper_window.get("boundary"),
        "source": paper_window.get("source"),
    }
    if paper_window.get("start") is not None:
        audit["start"] = paper_window.get("start")
    if paper_window.get("boundary") == "fixed_lookback":
        audit["bounded"] = True
    return audit


def _ensure_paper_window_covers_cursor(strategy_cfg: dict, *, dates) -> None:
    try:
        _, last_row = resolve_paper_state(strategy_cfg)
    except (FileNotFoundError, ValueError):
        return
    date_index = pd.DatetimeIndex(pd.to_datetime(dates, utc=True))
    last_logged_date = pd.to_datetime(last_row["date"], utc=True)
    if len(date_index) and last_logged_date < date_index[0]:
        raise click.ClickException(
            "paperExecutionProfile history boundary does not cover the current paper "
            f"ledger cursor {last_logged_date.isoformat()}; cannot append without "
            "skipping paper dates."
        )


def _target_dates_and_prices(engine, *, as_of=None, limit: int | None = None):
    try:
        ctx = engine.decision_context(end=as_of, limit=limit)
        close = ctx.target.series("close")
    except (DecisionContractError, FeedContractError, SignalContractError, TypeError, ValueError) as exc:
        raise click.ClickException(f"{engine.__class__.__name__}: {exc}") from exc
    close = close.sort_index()
    dates = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    prices = close.to_numpy(dtype=float)
    return dates, prices


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
    paper_window = _resolve_paper_data_window(strategy_cfg)
    runtime_cfg = _strategy_cfg_with_paper_window(strategy_cfg, paper_window)
    engine = engine_cls(context=runtime_cfg)

    mode = "compiled_output"
    signal_lookup = None
    if not _uses_default_paper_signal(engine):
        mode = "direct_paper_signal"
        def signal_lookup(ts):
            return _ensure_paper_signal(engine, as_of=ts)

        dates, prices = _target_dates_and_prices(engine, as_of=as_of)
        positions = np.zeros(len(dates), dtype=float)
        next_positions = np.zeros(len(dates), dtype=float)
    else:
        compiled = _compute_runtime_output(engine, runtime_cfg, end=as_of)
        dates = compiled.decision_index
        prices = compiled.close_prices
        positions = compiled.positions
        next_positions = compiled.next_position
    _ensure_paper_window_covers_cursor(strategy_cfg, dates=dates)

    try:
        result = append_paper_decision_rows(
            strategy_cfg,
            dates=dates,
            prices=prices,
            positions=positions,
            next_positions=next_positions,
            as_of=as_of,
            signal_lookup=signal_lookup,
        )
        result["execution_mode"] = mode
        result["paper_history_boundary"] = _paper_window_audit(paper_window)
        return result
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
