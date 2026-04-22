"""Tests for engine-backed research evaluation helpers."""

import json
import warnings
from pathlib import Path

from click.testing import CliRunner
import pandas as pd
import pytest

from causal_edge.cli import main
from causal_edge.engine.base import StrategyEngine
import causal_edge.research.evaluate as research_evaluate
from causal_edge.research.evaluate import (
    check_look_ahead,
    compute_k,
    render_validation_markdown,
    run_evaluation,
    run_preflight,
)
from causal_edge.research.data_readiness import run_data_verification
from causal_edge.research.workspace import init_workspace


def _write_engine(path: Path, *, bias: float = 0.02, flat: bool = False) -> None:
    if flat:
        body = (
            "        dates = pd.date_range('2024-01-01', periods=60, freq='D', tz='UTC')\n"
            "        positions = np.ones(60)\n"
            "        returns = np.full(60, 0.01)\n"
            "        prices = 100.0 * np.cumprod(1.0 + returns)\n"
        )
    else:
        body = (
            "        dates = pd.date_range('2024-01-01', periods=120, freq='D', tz='UTC')\n"
            "        phase = np.linspace(0, 8 * np.pi, 120)\n"
            "        positions = np.where(np.sin(phase) > 0, 1.0, -1.0)\n"
            f"        returns = {bias} * positions + 0.002 * np.sin(phase)\n"
            "        prices = 100.0 * np.cumprod(1.0 + returns)\n"
        )

    path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "import pandas as pd",
                "",
                "from causal_edge.engine.base import StrategyEngine",
                "",
                "",
                "class BranchEngine(StrategyEngine):",
                "    def compute_signals(self):",
                *body.rstrip("\n").splitlines(),
                "        return positions, dates, prices",
                "",
                "    def get_latest_signal(self):",
                "        positions, dates, _ = self.compute_signals()",
                "        return {'position': float(positions[-1]), 'date': str(dates[-1].date())}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_start_aware_engine(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "import pandas as pd",
                "",
                "from causal_edge.engine.base import StrategyEngine",
                "",
                "",
                "class BranchEngine(StrategyEngine):",
                "    def compute_signals(self):",
                "        requested = ((self.context or {}).get('_research') or {}).get('requested_window') or {}",
                "        start = requested.get('start') or '2024-01-01'",
                "        dates = pd.date_range(start, periods=120, freq='D', tz='UTC')",
                "        phase = np.linspace(0, 8 * np.pi, 120)",
                "        positions = np.where(np.sin(phase) > 0, 1.0, -1.0)",
                "        returns = 0.02 * positions + 0.002 * np.sin(phase)",
                "        prices = 100.0 * np.cumprod(1.0 + returns)",
                "        return positions, dates, prices",
                "",
                "    def get_latest_signal(self):",
                "        positions, dates, _ = self.compute_signals()",
                "        return {'position': float(positions[-1]), 'date': str(dates[-1].date())}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_alignment_failure_engine(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "import pandas as pd",
                "",
                "from causal_edge.engine.base import StrategyEngine",
                "",
                "",
                "class BranchEngine(StrategyEngine):",
                "    def compute_signals(self):",
                "        dates = pd.date_range('2024-01-01', periods=40, freq='D', tz='UTC')",
                "        aux = pd.Series(",
                "            np.arange(3, dtype=float),",
                "            index=pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-05'], utc=True),",
                "        )",
                "        self.align_series(aux, dates, method=None, allow_gaps=False)",
                "        prices = np.linspace(100.0, 120.0, len(dates))",
                "        positions = np.zeros(len(dates), dtype=float)",
                "        return positions, dates, prices",
                "",
                "    def get_latest_signal(self):",
                "        return {'position': 0.0}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_naive_dates_engine(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "import pandas as pd",
                "",
                "from causal_edge.engine.base import StrategyEngine",
                "",
                "",
                "class BranchEngine(StrategyEngine):",
                "    def compute_signals(self):",
                "        dates = pd.bdate_range('2024-01-01', periods=40)",
                "        prices = np.linspace(100.0, 120.0, len(dates))",
                "        positions = np.zeros(len(dates), dtype=float)",
                "        return positions, dates, prices",
                "",
                "    def get_latest_signal(self):",
                "        return {'position': 0.0}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_feed_loader_engine(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "import pandas as pd",
                "",
                "from causal_edge.engine.base import StrategyEngine",
                "",
                "",
                "class BranchEngine(StrategyEngine):",
                "    def compute_signals(self):",
                "        bars = self.load_research_bars(limit=40)",
                "        target = self.research_target_ticker() or 'SONY'",
                "        target_bars = bars[bars['symbol'] == target].copy().sort_values('timestamp')",
                "        dates = pd.DatetimeIndex(target_bars['timestamp'])",
                "        prices = target_bars['close'].astype(float).to_numpy()",
                "        positions = np.zeros(len(dates), dtype=float)",
                "        return positions, dates, prices",
                "",
                "    def get_latest_signal(self):",
                "        return {'position': 0.0}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_decision_context_engine(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import pandas as pd",
                "",
                "from causal_edge.engine.base import StrategyEngine",
                "",
                "",
                "class BranchEngine(StrategyEngine):",
                "    def compute_decisions(self, ctx):",
                "        close = ctx.target.series('close')",
                "        fast = close.rolling(window=3, min_periods=2).mean()",
                "        slow = close.rolling(window=5, min_periods=3).mean()",
                "        next_position = (fast > slow).astype(float).fillna(0.0)",
                "        if len(next_position) > 0:",
                "            next_position.iloc[0] = 0.0",
                "        return ctx.decisions(next_position)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_decision_context_escape_engine(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from causal_edge.engine.base import StrategyEngine",
                "",
                "",
                "class BranchEngine(StrategyEngine):",
                "    def compute_decisions(self, ctx):",
                "        self.research_close_frame(limit=20)",
                "        return ctx.decisions([0.0])",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv_context(tmp_path: Path) -> Path:
    bars_path = tmp_path / "target.csv"
    bars_path.write_text(
        "timestamp,close\n"
        "2024-01-01T00:00:00Z,100\n"
        "2024-01-02T00:00:00Z,101\n"
        "2024-01-03T00:00:00Z,103\n"
        "2024-01-04T00:00:00Z,104\n"
        "2024-01-05T00:00:00Z,106\n"
        "2024-01-06T00:00:00Z,108\n"
        "2024-01-07T00:00:00Z,109\n"
        "2024-01-08T00:00:00Z,111\n"
        "2024-01-09T00:00:00Z,113\n"
        "2024-01-10T00:00:00Z,114\n"
        "2024-01-11T00:00:00Z,116\n"
        "2024-01-12T00:00:00Z,117\n"
        "2024-01-13T00:00:00Z,119\n"
        "2024-01-14T00:00:00Z,121\n"
        "2024-01-15T00:00:00Z,122\n"
        "2024-01-16T00:00:00Z,124\n"
        "2024-01-17T00:00:00Z,125\n"
        "2024-01-18T00:00:00Z,127\n"
        "2024-01-19T00:00:00Z,128\n"
        "2024-01-20T00:00:00Z,130\n"
        "2024-01-21T00:00:00Z,131\n"
        "2024-01-22T00:00:00Z,133\n"
        "2024-01-23T00:00:00Z,135\n"
        "2024-01-24T00:00:00Z,136\n"
        "2024-01-25T00:00:00Z,138\n"
        "2024-01-26T00:00:00Z,139\n"
        "2024-01-27T00:00:00Z,141\n"
        "2024-01-28T00:00:00Z,143\n"
        "2024-01-29T00:00:00Z,144\n"
        "2024-01-30T00:00:00Z,146\n"
        "2024-01-31T00:00:00Z,147\n"
        "2024-02-01T00:00:00Z,149\n"
        "2024-02-02T00:00:00Z,151\n"
        "2024-02-03T00:00:00Z,152\n"
        "2024-02-04T00:00:00Z,154\n"
        "2024-02-05T00:00:00Z,155\n",
        encoding="utf-8",
    )
    context_path = tmp_path / "context.json"
    context_path.write_text(
        json.dumps(
            {
                "ticker": "SONY",
                "discovery": {"ticker": "SONY"},
                "_feeds": {
                    "primary": {
                        "name": "primary",
                        "kind": "bars",
                        "adapter": "csv",
                        "timeframe": "1d",
                        "symbol": "SONY",
                        "profile": "daily",
                        "path": str(bars_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return context_path


class TestInitWorkspace:
    def test_creates_engine_backed_files(self, tmp_path):
        workspace = init_workspace("SOLUSD", tmp_path / "sol")
        assert (workspace / "engine.py").exists()
        assert (workspace / "results.tsv").exists()
        assert (workspace / "memory.md").exists()
        assert (workspace / "discovery.json").exists()
        template = (workspace / "engine.py").read_text(encoding="utf-8")
        discovery = json.loads((workspace / "discovery.json").read_text(encoding="utf-8"))
        assert "compute_decisions(self, ctx)" in template
        assert "ctx.decisions(" in template
        assert discovery["target_node"] == "SOLUSD.price"

    def test_idempotent_engine_file(self, tmp_path):
        workspace = init_workspace("SOL", tmp_path / "sol")
        (workspace / "engine.py").write_text("# modified", encoding="utf-8")
        init_workspace("SOL", tmp_path / "sol")
        assert (workspace / "engine.py").read_text(encoding="utf-8") == "# modified"


class TestComputeK:
    def test_counts_tickers_and_lags(self, tmp_path):
        engine = tmp_path / "engine.py"
        engine.write_text(
            'PARENTS = [("BTCUSD", 3), ("ETHUSD", 21)]\n'
            "class BranchEngine:\n"
            "    def compute_signals(self):\n"
            "        feature.shift(3)\n"
            "        feature.shift(21)\n",
            encoding="utf-8",
        )
        k_value, tickers, lags = compute_k(engine)
        assert k_value >= 4
        assert "BTCUSD" in tickers
        assert 21 in lags

    def test_filters_market_factors(self, tmp_path):
        engine = tmp_path / "engine.py"
        engine.write_text(
            'tickers = ["BTCUSD", "SPY"]\nclass BranchEngine:\n    def compute_signals(self):\n        feature.shift(5)\n',
            encoding="utf-8",
        )
        _, tickers, _ = compute_k(engine)
        assert "SPY" not in tickers


class TestLookAhead:
    def test_static_check_catches_rolling_without_shift(self, tmp_path):
        engine = tmp_path / "engine.py"
        engine.write_text("x = ret.rolling(20).mean()\npositions = x > 0", encoding="utf-8")
        assert check_look_ahead(engine)


class TestRunEvaluation:
    def test_missing_engine_file(self, tmp_path):
        result = run_evaluation(tmp_path)
        assert result["verdict"] == "ERROR"
        assert "engine.py not found" in result["failures"][0]

    def test_strategy_py_only_is_rejected(self, tmp_path):
        (tmp_path / "strategy.py").write_text("def run_strategy():\n    return None\n", encoding="utf-8")
        result = run_evaluation(tmp_path)
        assert result["verdict"] == "ERROR"
        assert "engine.py not found" in result["failures"][0]

    def test_unimplemented_engine_returns_error(self, tmp_path):
        engine = tmp_path / "engine.py"
        engine.write_text(
            "\n".join(
                [
                    "from causal_edge.engine.base import StrategyEngine",
                    "",
                    "class BranchEngine(StrategyEngine):",
                    "    def compute_signals(self):",
                    "        raise NotImplementedError('todo')",
                    "",
                    "    def get_latest_signal(self):",
                    "        return {'position': 0.0}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = run_evaluation(tmp_path)
        assert result["verdict"] == "ERROR"
        assert any("failed" in failure.lower() for failure in result["failures"])

    def test_viable_engine_passes(self, tmp_path):
        _write_engine(tmp_path / "engine.py")
        result = run_evaluation(tmp_path)
        assert result["verdict"] == "PASS"
        assert result["K"] >= 1
        assert result["profile"] == "equity_daily"
        assert result["implementation_contract"] == "legacy_signal_contract"

    def test_decision_context_engine_passes_and_records_trace(self, tmp_path):
        _write_decision_context_engine(tmp_path / "engine.py")
        context_path = _write_csv_context(tmp_path)

        result = run_evaluation(tmp_path, context_json=context_path)

        assert result["verdict"] != "ERROR"
        assert result["implementation_contract"] == "decision_context"
        assert result["decision_trace"]
        assert result["decision_preview"]
        assert result["semantic"]["verdict"] == "PASS"

    def test_start_aware_engine_records_requested_and_effective_window(self, tmp_path):
        _write_start_aware_engine(tmp_path / "engine.py")
        result = run_evaluation(tmp_path, start="2020-01-01")
        assert result["verdict"] == "PASS"
        assert result["requested_window"] == {"start": "2020-01-01", "end": None}
        assert result["effective_window"]["start"] == "2020-01-01"
        assert result["effective_window"]["end"] == "2020-04-29"

    def test_can_persist_metric_input_csv(self, tmp_path):
        _write_engine(tmp_path / "engine.py")
        output_csv = tmp_path / "artifacts" / "metric-input.csv"

        result = run_evaluation(tmp_path, output_csv=output_csv)

        assert result["verdict"] == "PASS"
        assert output_csv.exists()
        payload = output_csv.read_text(encoding="utf-8")
        assert "date,pnl,position,asset_return" in payload

    def test_includes_runtime_diagnostics_for_constant_position(self, tmp_path):
        _write_engine(tmp_path / "engine.py", flat=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = run_evaluation(tmp_path)

        diagnostics = result["diagnostics"]
        assert diagnostics["failure_signature"] == "constant_position"
        assert diagnostics["signal"]["position_switches"] == 0
        assert diagnostics["signal"]["unique_position_count"] == 1

    def test_validation_exception_surfaces_runtime_stage(self, tmp_path, monkeypatch):
        _write_engine(tmp_path / "engine.py", flat=True)

        def boom(*args, **kwargs):
            raise RuntimeError("validation exploded")

        monkeypatch.setattr(research_evaluate, "validate_strategy", boom)

        result = run_evaluation(tmp_path)

        assert result["verdict"] == "ERROR"
        diagnostics = result["diagnostics"]
        assert diagnostics["runtime_stage"] == "validation"
        assert diagnostics["failure_signature"] == "validation_failed"
        assert "validation exploded" in result["failures"][0]

    def test_reports_alignment_collapse_when_series_cannot_align(self, tmp_path):
        _write_alignment_failure_engine(tmp_path / "engine.py")
        result = run_evaluation(tmp_path)

        diagnostics = result["diagnostics"]
        assert result["verdict"] == "ERROR"
        assert diagnostics["failure_signature"] == "alignment_collapse"
        assert diagnostics["runtime_stage"] == "compute_strategy"

    def test_reports_datetime_contract_violation_for_naive_dates(self, tmp_path):
        _write_naive_dates_engine(tmp_path / "engine.py")
        result = run_evaluation(tmp_path)

        diagnostics = result["diagnostics"]
        assert result["verdict"] == "ERROR"
        assert diagnostics["failure_signature"] == "datetime_contract_violation"
        assert diagnostics["runtime_stage"] == "compute_strategy"
        assert "UTC-aware" in result["failures"][0]

    def test_decision_context_blocks_raw_strategy_escape_hatch(self, tmp_path):
        _write_decision_context_escape_engine(tmp_path / "engine.py")
        context_path = _write_csv_context(tmp_path)

        result = run_evaluation(tmp_path, context_json=context_path)

        assert result["verdict"] == "ERROR"
        assert result["diagnostics"]["failure_signature"] == "decision_context_escape_hatch"

    def test_rejects_engine_that_only_reexports_imported_class(self, tmp_path):
        helper = tmp_path / "helper_engine.py"
        helper.write_text(
            "\n".join(
                [
                    "from causal_edge.engine.base import StrategyEngine",
                    "",
                    "class HelperEngine(StrategyEngine):",
                    "    def compute_signals(self):",
                    "        raise NotImplementedError",
                    "",
                    "    def get_latest_signal(self):",
                    "        return {'position': 0.0}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (tmp_path / "engine.py").write_text(
            "from helper_engine import HelperEngine\n",
            encoding="utf-8",
        )

        result = run_evaluation(tmp_path)

        assert result["verdict"] == "ERROR"
        assert "module-owned StrategyEngine subclass" in result["failures"][0]

    def test_context_json_build_synthesizes_primary_feed(self, tmp_path):
        context_path = tmp_path / "context.json"
        context_path.write_text(
            json.dumps({"ticker": "SONY", "discovery": {"ticker": "SONY"}}),
            encoding="utf-8",
        )
        context = research_evaluate._build_research_context(
            workspace=tmp_path,
            start=None,
            context_json=context_path,
        )

        assert context["_feeds"]["primary"]["symbol"] == "SONY"
        assert context["_feeds"]["primary"]["adapter"] == "abel"
        assert context["_runtime_profile"]["target"] == "SONY"
        assert context["_execution_constraints"]["long_only"] is False

    def test_preflight_keeps_static_checks_as_warnings(self, tmp_path):
        engine = tmp_path / "engine.py"
        engine.write_text(
            "\n".join(
                [
                    "import numpy as np",
                    "import pandas as pd",
                    "",
                    "from causal_edge.engine.base import StrategyEngine",
                    "",
                    "",
                    "class BranchEngine(StrategyEngine):",
                    "    def compute_signals(self):",
                    "        dates = pd.date_range('2024-01-01', periods=40, freq='D', tz='UTC')",
                    "        prices = np.linspace(100.0, 130.0, len(dates))",
                    "        rolling = pd.Series(prices).rolling(5).mean()",
                    "        positions = (rolling > 110).fillna(0.0).to_numpy(dtype=float)",
                    "        return positions, dates, prices",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = run_preflight(tmp_path)

        assert result["verdict"] == "PASS"
        assert any("Static look-ahead heuristics" in warning for warning in result["semantic"]["warnings"])

    @pytest.mark.parametrize(
        "example_dir",
        [
            "sma_crossover",
            "momentum_ml",
            "causal_demo",
        ],
    )
    def test_shipped_examples_do_not_fail_datetime_contract(self, example_dir):
        root = Path(__file__).resolve().parents[1]
        result = run_evaluation(root / "examples" / example_dir)

        assert result["diagnostics"]["failure_signature"] != "datetime_contract_violation"
        assert not any("UTC-aware" in failure for failure in result["failures"])


class TestResearchHelpers:
    def test_strategy_engine_exposes_discovery_and_readiness_helpers(self):
        class DemoEngine(StrategyEngine):
            def compute_signals(self):
                raise NotImplementedError

            def get_latest_signal(self):
                return {"position": 0.0}

        engine = DemoEngine(
            context={
                "ticker": "sony",
                "discovery": {
                    "ticker": "SONY",
                    "parents": [{"ticker": "AAPL", "field": "price"}],
                    "blanket_new": [{"ticker": "MSFT", "field": "price", "roles": ["spouse"]}],
                    "children": [{"ticker": "NVDA", "field": "price"}],
                    "data_readiness": {
                        "results": [
                            {
                                "ticker": "SONY",
                                "status": "start_covered",
                                "usable": True,
                                "covers_requested_start": True,
                            },
                            {
                                "ticker": "AAPL",
                                "status": "start_covered",
                                "usable": True,
                                "covers_requested_start": True,
                                "rows": 252,
                            },
                            {
                                "ticker": "MSFT",
                                "status": "partial_window",
                                "usable": True,
                                "covers_requested_start": False,
                                "rows": 120,
                            },
                            {
                                "ticker": "NVDA",
                                "status": "no_data",
                                "usable": False,
                                "covers_requested_start": False,
                            },
                        ]
                    },
                },
                "_research": {
                    "requested_window": {"start": "2020-01-01", "end": None},
                },
            }
        )

        assert engine.research_target_ticker() == "SONY"
        assert engine.research_requested_start() == "2020-01-01"
        assert engine.research_driver_tickers() == ["AAPL", "MSFT"]
        assert engine.research_driver_tickers(require_full_window=True) == ["AAPL"]
        assert engine.research_driver_tickers(roles=("parent",)) == ["AAPL"]

        candidates = engine.research_driver_candidates(require_usable=False)
        assert [item["ticker"] for item in candidates] == ["AAPL", "MSFT", "NVDA"]
        assert candidates[0]["discovery_roles"] == ["parent"]
        assert candidates[1]["discovery_roles"] == ["blanket", "spouse"]
        assert candidates[2]["readiness_status"] == "no_data"

    def test_strategy_engine_can_load_research_bars_and_close_frame(self):
        class DemoEngine(StrategyEngine):
            def compute_signals(self):
                raise NotImplementedError

            def get_latest_signal(self):
                return {"position": 0.0}

            def _load_research_symbol_bars(self, symbol, **kwargs):
                assert kwargs["start"] == "2020-01-01"
                assert kwargs["limit"] == 600
                rows = {
                    "SONY": [
                        ("2020-01-01", 100.0),
                        ("2020-01-02", 101.0),
                    ],
                    "AAPL": [
                        ("2020-01-01", 50.0),
                        ("2020-01-02", 51.0),
                    ],
                }[symbol]
                return pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime([item[0] for item in rows], utc=True),
                        "symbol": [symbol] * len(rows),
                        "close": [item[1] for item in rows],
                    }
                )

        engine = DemoEngine(
            context={
                "discovery": {
                    "ticker": "SONY",
                    "parents": [{"ticker": "AAPL", "field": "price"}],
                    "data_readiness": {
                        "results": [
                            {"ticker": "AAPL", "status": "start_covered", "usable": True, "covers_requested_start": True},
                        ]
                    },
                },
                "_research": {"requested_window": {"start": "2020-01-01", "end": None}},
            }
        )

        bars = engine.load_research_bars(require_full_window=True)
        frame = engine.research_close_frame(require_full_window=True)

        assert list(bars["symbol"]) == ["SONY", "AAPL", "SONY", "AAPL"]
        assert list(frame.columns) == ["SONY", "AAPL"]
        assert float(frame.iloc[-1]["SONY"]) == 101.0
        assert float(frame.iloc[-1]["AAPL"]) == 51.0

    def test_same_asset_volume_candidate_survives_target_exclusion(self):
        class DemoEngine(StrategyEngine):
            def compute_signals(self):
                raise NotImplementedError

            def get_latest_signal(self):
                return {"position": 0.0}

        engine = DemoEngine(
            context={
                "discovery": {
                    "ticker": "SONY",
                    "target_node": "SONY.price",
                    "parents": [],
                    "blanket_new": [{"ticker": "SONY", "field": "volume", "roles": ["sibling"]}],
                    "children": [],
                    "data_readiness": {
                        "results": [
                            {
                                "ticker": "SONY",
                                "status": "start_covered",
                                "usable": True,
                                "covers_requested_start": True,
                            }
                        ]
                    },
                }
            }
        )

        candidates = engine.research_driver_candidates(require_usable=False)

        assert len(candidates) == 1
        assert candidates[0]["node_id"] == "SONY.volume"
        assert candidates[0]["ticker"] == "SONY"

    def test_load_research_bars_rejects_empty_symbol_selection(self):
        class DemoEngine(StrategyEngine):
            def compute_signals(self):
                raise NotImplementedError

            def get_latest_signal(self):
                return {"position": 0.0}

        engine = DemoEngine(context={"_research": {"requested_window": {"start": "2020-01-01"}}})

        with pytest.raises(ValueError, match="No research symbols were selected"):
            engine.load_research_bars(include_target=False, driver_tickers=[])

    def test_strategy_engine_can_build_target_driver_frame_with_safe_overlap_modes(self):
        class DemoEngine(StrategyEngine):
            def compute_signals(self):
                raise NotImplementedError

            def get_latest_signal(self):
                return {"position": 0.0}

            def _load_research_symbol_bars(self, symbol, **kwargs):
                rows = {
                    "SONY": [
                        ("2020-01-01", 100.0),
                        ("2020-01-02", 101.0),
                        ("2020-01-03", 102.0),
                    ],
                    "AAPL": [
                        ("2020-01-01", 50.0),
                        ("2020-01-03", 52.0),
                    ],
                    "MSFT": [
                        ("2020-01-01", 20.0),
                        ("2020-01-02", 21.0),
                        ("2020-01-03", 22.0),
                    ],
                }[symbol]
                return pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime([item[0] for item in rows], utc=True),
                        "symbol": [symbol] * len(rows),
                        "close": [item[1] for item in rows],
                    }
                )

        engine = DemoEngine(
            context={
                "discovery": {
                    "ticker": "SONY",
                    "parents": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
                    "data_readiness": {
                        "results": [
                            {"ticker": "AAPL", "status": "start_covered", "usable": True, "covers_requested_start": True},
                            {"ticker": "MSFT", "status": "start_covered", "usable": True, "covers_requested_start": True},
                        ]
                    },
                },
                "_research": {"requested_window": {"start": "2020-01-01", "end": None}},
            }
        )

        target_intersection, drivers_intersection = engine.research_target_driver_frame(
            require_full_window=True,
            overlap="intersection",
            require_drivers=True,
        )
        target_target_only, drivers_target_only = engine.research_target_driver_frame(
            require_full_window=True,
            overlap="target_only",
            require_drivers=True,
        )

        assert list(target_intersection.index.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-03"]
        assert list(drivers_intersection.columns) == ["AAPL", "MSFT"]
        assert list(target_target_only.index.strftime("%Y-%m-%d")) == [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
        ]
        assert pd.isna(drivers_target_only.loc[pd.Timestamp("2020-01-02", tz="UTC"), "AAPL"])


class TestValidationMarkdown:
    def test_renders_validation_summary(self, tmp_path):
        _write_engine(tmp_path / "engine.py")
        result = run_evaluation(tmp_path)
        report = render_validation_markdown(result)
        assert "# Evaluation Summary" in report
        assert "## Verdict" in report
        assert result["verdict"] in report
        assert "implementation_contract" in report


class TestEvaluateCli:
    def test_evaluate_cli_writes_raw_outputs(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            workdir = Path("workspace")
            workdir.mkdir()
            _write_engine(workdir / "engine.py")

            result = runner.invoke(
                main,
                [
                    "evaluate",
                    "--workdir",
                    str(workdir),
                    "--output-json",
                    str(workdir / "edge-result.json"),
                    "--output-md",
                    str(workdir / "edge-validation.md"),
                    "--output-csv",
                    str(workdir / "metric-input.csv"),
                ],
            )
            assert result.exit_code == 0, result.output
            assert (workdir / "edge-result.json").exists()
            assert (workdir / "edge-validation.md").exists()
            assert (workdir / "metric-input.csv").exists()
            assert "Verdict: PASS" in result.output
            assert "Input CSV:" in result.output

    def test_evaluate_cli_passes_start_into_research_context(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            workdir = Path("workspace")
            workdir.mkdir()
            _write_start_aware_engine(workdir / "engine.py")

            result = runner.invoke(
                main,
                [
                    "evaluate",
                    "--workdir",
                    str(workdir),
                    "--start",
                    "2020-01-01",
                    "--output-json",
                    str(workdir / "edge-result.json"),
                ],
            )
            assert result.exit_code == 0, result.output
            payload = (workdir / "edge-result.json").read_text(encoding="utf-8")
            assert '"start": "2020-01-01"' in payload

    def test_evaluate_cli_fails_for_bad_engine(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            workdir = Path("workspace")
            workdir.mkdir()
            _write_engine(workdir / "engine.py", flat=True)

            result = runner.invoke(main, ["evaluate", "--workdir", str(workdir)])
            assert result.exit_code != 0
            assert "Verdict: FAIL" in result.output

    def test_debug_evaluate_cli_surfaces_failure_signature(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            workdir = Path("workspace")
            workdir.mkdir()
            _write_engine(workdir / "engine.py", flat=True)

            result = runner.invoke(main, ["debug-evaluate", "--workdir", str(workdir)])
            assert result.exit_code == 0, result.output
            assert "Failure signature: constant_position" in result.output
            assert "Signal activity:" in result.output


class TestVerifyData:
    def test_run_data_verification_reports_full_partial_and_missing(self, monkeypatch, tmp_path):
        from causal_edge.research import data_readiness as data_module

        def _fake_fetch_bars(*, symbols, start=None, end=None, timeframe="1d", limit=None, fields=None, config=None):
            ticker = symbols[0]
            if ticker == "SONY":
                return _bars_frame(["2020-01-01", "2020-01-02", "2020-01-03"])
            if ticker == "CNET":
                return _bars_frame(["2020-08-01", "2020-08-02"])
            if ticker == "EMPTY":
                return _bars_frame([])
            raise RuntimeError("boom")

        monkeypatch.setattr(data_module, "fetch_bars", _fake_fetch_bars)
        discovery = tmp_path / "discovery.json"
        discovery.write_text(
            json.dumps(
                {
                    "ticker": "SONY",
                    "backtest": {"start": "2020-01-01"},
                    "parents": [{"ticker": "CNET"}],
                    "blanket_new": [{"ticker": "EMPTY"}],
                    "children": [{"ticker": "BROKEN"}],
                }
            ),
            encoding="utf-8",
        )

        report = run_data_verification(discovery_json=discovery)

        summary = report["summary"]
        assert summary["ticker_count"] == 4
        assert summary["start_covered_count"] == 1
        assert summary["partial_window_count"] == 1
        assert summary["no_data_count"] == 1
        assert summary["error_count"] == 1
        assert report["probe"]["limit"] == 500
        assert report["probe"]["strategy"] == "target_boundary_confirm"
        assert report["target_boundary"]["classification"] == "confirmed_before_requested_start"
        assert report["target_boundary"]["observed_first_timestamp"] == "2020-01-01"
        assert report["coverage_hints"]["target_safe_start"] == "2020-01-01"
        assert report["coverage_hints"]["dense_overlap_hint_start"] == "2020-08-01"
        cnet = next(item for item in report["results"] if item["ticker"] == "CNET")
        assert cnet["observed_first_timestamp"] == "2020-08-01"
        assert cnet["covers_requested_start"] is False
        assert cnet["left_boundary_confidence"] == "confirmed"

    def test_run_data_verification_marks_target_boundary_unknown_when_probe_truncates(self, monkeypatch, tmp_path):
        from causal_edge.research import data_readiness as data_module

        def _fake_fetch_bars(*, symbols, start=None, end=None, timeframe="1d", limit=None, fields=None, config=None):
            ticker = symbols[0]
            if ticker != "META":
                return _bars_frame(["2024-11-25", "2024-11-26"])
            if limit == 500:
                return _bars_range("2024-04-09", periods=500)
            if limit == 1000:
                return _bars_range("2023-04-10", periods=1000)
            return _bars_range("2022-04-11", periods=2000)

        monkeypatch.setattr(data_module, "fetch_bars", _fake_fetch_bars)
        discovery = tmp_path / "discovery.json"
        discovery.write_text(
            json.dumps(
                {
                    "ticker": "META",
                    "backtest": {"start": "2020-01-01"},
                    "parents": [{"ticker": "PEER"}],
                }
            ),
            encoding="utf-8",
        )

        report = run_data_verification(discovery_json=discovery)

        assert report["target_boundary"]["classification"] == "unknown_probe_truncated"
        assert report["target_boundary"]["observed_first_timestamp"] == "2022-04-11"
        assert report["probe"]["target_confirmation_attempted"] is True
        assert report["probe"]["target_final_limit"] == 2000
        assert report["coverage_hints"]["target_safe_start"] == "2022-04-11"
        meta = next(item for item in report["results"] if item["ticker"] == "META")
        assert meta["left_boundary_confidence"] == "observed"
        assert meta["covers_requested_start"] is False

    def test_verify_data_cli_writes_json(self, monkeypatch, tmp_path):
        from causal_edge.research import data_readiness as data_module

        monkeypatch.setattr(
            data_module,
            "fetch_bars",
            lambda **kwargs: _bars_frame(["2020-01-01", "2020-01-02"]),
        )

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            output_json = Path("report.json")
            result = runner.invoke(
                main,
                [
                    "verify-data",
                    "--ticker",
                    "SONY",
                    "--start",
                    "2020-01-01",
                    "--output-json",
                    str(output_json),
                ],
            )
            assert result.exit_code == 0, result.output
            assert "Research Data Verification" in result.output
            assert output_json.exists()
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            assert payload["summary"]["usable_count"] == 1


def _bars_frame(dates: list[str]):
    import pandas as pd

    if not dates:
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates, utc=True),
            "symbol": ["SONY"] * len(dates),
            "close": [100.0 + idx for idx in range(len(dates))],
        }
    )


def _bars_range(start: str, periods: int):
    import pandas as pd

    dates = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["META"] * len(dates),
            "close": [100.0 + idx for idx in range(len(dates))],
        }
    )
