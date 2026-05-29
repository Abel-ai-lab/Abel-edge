from pathlib import Path
import importlib
import sys

import pandas as pd
from click.testing import CliRunner

from abel_edge.config import load_config
from abel_edge.engine.ledger import write_trade_log
from abel_edge.engine.trader import SYSTEM_LOOKBACK_PADDING_BARS, paper_run_one


DIRECT_INPUT_SIGNAL_ENGINE_CODE = """
from __future__ import annotations

import pandas as pd

from abel_edge.engine.base import StrategyEngine


class DirectInputSignalEngine(StrategyEngine):
    calls = []

    def compute_decisions(self, ctx):
        raise AssertionError("paper_run_one direct mode must not compile full output")

    def get_paper_signal(self, *, as_of=None):
        ctx = self.decision_context(end=as_of)
        close = ctx.target.series("close")
        driver = ctx.input("driver").native_series("close")
        type(self).calls.append(
            (
                str(pd.to_datetime(as_of, utc=True).date()),
                len(close),
                str(driver.index[-1].date()),
                len(driver),
            )
        )
        return {
            "next_position": 1.0 if float(driver.iloc[-1]) >= 120.0 else 0.0,
            "data_latest_timestamp": str(driver.index[-1]),
        }
""".strip()


RECORDING_ADAPTER_CODE = """
from __future__ import annotations

import pandas as pd

from abel_edge.engine.adapter_registry import register_adapter

REQUEST_ENDS = []
REQUEST_LIMITS = []


class RecordingCSVAdapter:
    assume_utc_for_naive = True

    def load(self, request):
        REQUEST_ENDS.append(
            None
            if request.end is None
            else str(pd.to_datetime(request.end, utc=True).date())
        )
        REQUEST_LIMITS.append(request.limit)
        frame = pd.read_csv(request.options["path"])
        if "symbol" not in frame.columns:
            frame["symbol"] = request.symbol or "DRIVER"
        return frame


register_adapter("recording_csv", RecordingCSVAdapter())
""".strip()


def _clear_modules() -> None:
    for name in list(sys.modules):
        if name in {"strategies", "project_adapters"} or name.startswith("strategies."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _daily_price_csv(*, days: int = 25) -> str:
    rows = ["timestamp,close"]
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for idx in range(days):
        ts = start + pd.Timedelta(days=idx)
        rows.append(f"{ts.isoformat()},{100 + idx}")
    return "\n".join(rows) + "\n"


def _write_bootstrap_log(
    path: str,
    *,
    dates: list[str],
    closes: list[float],
    positions: list[float],
    next_positions: list[float],
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


def _run_direct_input_profile_paper(tmp_path, *, as_of):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _clear_modules()
        (root / "project_adapters.py").write_text(
            RECORDING_ADAPTER_CODE + "\n",
            encoding="utf-8",
        )
        strategy_dir = root / "strategies" / "direct_input_profile"
        strategy_dir.mkdir(parents=True)
        (root / "strategies" / "__init__.py").write_text("", encoding="utf-8")
        (strategy_dir / "__init__.py").write_text("", encoding="utf-8")
        (strategy_dir / "engine.py").write_text(
            DIRECT_INPUT_SIGNAL_ENGINE_CODE,
            encoding="utf-8",
        )
        (root / "data").mkdir()
        (root / "data" / "ethusd.csv").write_text(_daily_price_csv(), encoding="utf-8")
        (root / "data" / "driver.csv").write_text(_daily_price_csv(), encoding="utf-8")
        (root / "strategies.yaml").write_text(
            """
settings:
  price_data:
    default_source: csv
    default_timeframe: 1d
  data_adapters:
    imports:
      - project_adapters
strategies:
  - id: direct_input_profile
    name: "direct_input_profile"
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.direct_input_profile.engine
    trade_log: data/trade_log_direct_input_profile.csv
    paper_log: data/paper_log_direct_input_profile.csv
    price_data:
      source: csv
      path: data/ethusd.csv
    paper_execution_profile:
      history:
        boundary: fixed_lookback
        lookbackBars: 1
        feeds:
          - ETHUSD
          - DRIVER
    feeds:
      driver:
        kind: bars
        adapter: recording_csv
        symbol: DRIVER
        path: data/driver.csv
""".strip()
            + "\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(root))

        try:
            _write_bootstrap_log(
                "data/trade_log_direct_input_profile.csv",
                dates=["2026-01-23T00:00:00Z"],
                closes=[122.0],
                positions=[1.0],
                next_positions=[1.0],
            )
            cfg = load_config()

            result = paper_run_one(
                cfg["strategies"][0],
                settings=cfg.get("settings"),
                as_of=as_of,
            )

            adapter_module = importlib.import_module("project_adapters")
            engine_module = importlib.import_module("strategies.direct_input_profile.engine")
            return {
                "result": result,
                "request_ends": list(adapter_module.REQUEST_ENDS),
                "request_limits": list(adapter_module.REQUEST_LIMITS),
                "engine_calls": list(engine_module.DirectInputSignalEngine.calls),
            }
        finally:
            sys.path.pop(0)
            _clear_modules()


def _assert_final_horizon_warmup(outcome):
    assert outcome["request_ends"] == ["2026-01-25", "2026-01-25"]
    assert outcome["request_limits"] == [
        SYSTEM_LOOKBACK_PADDING_BARS + 3,
        SYSTEM_LOOKBACK_PADDING_BARS + 1,
    ]
    assert outcome["engine_calls"] == [
        ("2026-01-24", SYSTEM_LOOKBACK_PADDING_BARS + 1, "2026-01-24", 21),
        ("2026-01-25", SYSTEM_LOOKBACK_PADDING_BARS + 1, "2026-01-25", 21),
    ]
    assert outcome["result"]["paper_history_boundary"]["boundary"] == "fixed_lookback"
    assert outcome["result"]["n_rows"] == 2


def test_paper_history_fixed_lookback_uses_explicit_as_of_horizon_for_feed_cache(
    tmp_path,
):
    outcome = _run_direct_input_profile_paper(
        tmp_path,
        as_of="2026-01-25T00:00:00Z",
    )

    _assert_final_horizon_warmup(outcome)


def test_paper_history_fixed_lookback_infers_latest_horizon_without_as_of(tmp_path):
    outcome = _run_direct_input_profile_paper(tmp_path, as_of=None)

    _assert_final_horizon_warmup(outcome)
