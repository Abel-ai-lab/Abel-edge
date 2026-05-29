from pathlib import Path
import importlib
import sys

import pandas as pd

from abel_edge.engine.ledger import write_trade_log


DECISION_ENGINE_CODE = """
from __future__ import annotations

import pandas as pd

from abel_edge.engine.base import StrategyEngine


class CountingDecisionEngine(StrategyEngine):
    calls = []

    def compute_decisions(self, ctx):
        close = ctx.target.series("close")
        type(self).calls.append((str(close.index[0].date()), str(close.index[-1].date()), len(close)))
        next_position = pd.Series(
            [idx * 0.25 for idx in range(len(close))],
            index=close.index,
        )
        return ctx.decisions(next_position)
""".strip()


DIRECT_SIGNAL_ENGINE_CODE = """
from __future__ import annotations

import pandas as pd

from abel_edge.engine.base import StrategyEngine


class DirectSignalEngine(StrategyEngine):
    calls = []

    def compute_decisions(self, ctx):
        raise AssertionError("paper_run_one direct mode must not compile full output")

    def get_paper_signal(self, *, as_of=None):
        bars = self.load_bars(end=as_of).sort_values("timestamp")
        last = bars.iloc[-1]
        type(self).calls.append((str(pd.to_datetime(last["timestamp"], utc=True).date()), len(bars)))
        return {
            "next_position": 1.0 if float(last["close"]) >= 120.0 else 0.0,
            "data_latest_timestamp": str(pd.to_datetime(last["timestamp"], utc=True)),
        }
""".strip()


BOOTSTRAP_CONTEXT_ENGINE_CODE = """
from __future__ import annotations

import pandas as pd

from abel_edge.engine.base import StrategyEngine


class BootstrapContextEngine(StrategyEngine):
    calls = []

    def compute_decisions(self, ctx):
        raise AssertionError("paper_run_one direct mode must not compile full output")

    def build_paper_initial_state(self, *, cutover_as_of=None):
        ctx = self.paper_bootstrap_context(start="2026-01-01T00:00:00Z", end=cutover_as_of)
        close = ctx.target.series("close")
        type(self).calls.append(
            (
                "bootstrap",
                str(close.index[0].date()),
                str(close.index[-1].date()),
                len(close),
            )
        )
        return {"rows": len(close)}

    def get_paper_signal(self, *, as_of=None):
        ctx = self.decision_context(end=as_of)
        close = ctx.target.series("close")
        type(self).calls.append(
            (
                "daily",
                str(close.index[0].date()),
                str(close.index[-1].date()),
                len(close),
            )
        )
        return {
            "next_position": 1.0 if float(close.iloc[-1]) >= 120.0 else 0.0,
            "data_latest_timestamp": str(close.index[-1]),
        }
""".strip()


def clear_strategy_modules() -> None:
    for name in list(sys.modules):
        if name == "strategies" or name.startswith("strategies."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def daily_price_csv(*, days: int = 25) -> str:
    rows = ["timestamp,close"]
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for idx in range(days):
        ts = start + pd.Timedelta(days=idx)
        rows.append(f"{ts.isoformat()},{100 + idx}")
    return "\n".join(rows) + "\n"


def write_project(
    root: Path,
    *,
    package_name: str,
    engine_code: str,
    profile_yaml: str | None = None,
    days: int = 25,
) -> None:
    clear_strategy_modules()
    strategy_dir = root / "strategies" / package_name
    strategy_dir.mkdir(parents=True)
    (root / "strategies" / "__init__.py").write_text("", encoding="utf-8")
    (strategy_dir / "__init__.py").write_text("", encoding="utf-8")
    (strategy_dir / "engine.py").write_text(engine_code, encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "ethusd.csv").write_text(daily_price_csv(days=days), encoding="utf-8")
    profile_block = f"\n    paper_execution_profile:\n{profile_yaml}" if profile_yaml else ""
    (root / "strategies.yaml").write_text(
        f"""
settings:
  price_data:
    default_source: csv
    default_timeframe: 1d
strategies:
  - id: {package_name}
    name: "{package_name}"
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.{package_name}.engine
    trade_log: data/trade_log_{package_name}.csv
    paper_log: data/paper_log_{package_name}.csv
    price_data:
      source: csv
      path: data/ethusd.csv{profile_block}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def write_bootstrap_log(
    path: str,
    *,
    dates: list[str],
    closes: list[float],
    positions: list[float],
    next_positions: list[float] | None = None,
) -> None:
    parsed = pd.to_datetime(dates, utc=True)
    returns = [0.0]
    returns.extend(closes[idx] / closes[idx - 1] - 1.0 for idx in range(1, len(closes)))
    pnl = [position * value for position, value in zip(positions, returns)]
    write_trade_log(
        parsed,
        returns,
        pnl,
        positions,
        path,
        close_prices=closes,
        next_positions=next_positions,
    )
