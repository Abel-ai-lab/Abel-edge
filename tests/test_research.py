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


class TestInitWorkspace:
    def test_creates_engine_backed_files(self, tmp_path):
        workspace = init_workspace("SOLUSD", tmp_path / "sol")
        assert (workspace / "engine.py").exists()
        assert (workspace / "results.tsv").exists()
        assert (workspace / "memory.md").exists()
        assert (workspace / "discovery.json").exists()

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
        assert result["implementation_contract"] == "engine"

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
        assert diagnostics["runtime_stage"] == "compute_signals"

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
                                "status": "full_window",
                                "usable": True,
                                "full_window": True,
                            },
                            {
                                "ticker": "AAPL",
                                "status": "full_window",
                                "usable": True,
                                "full_window": True,
                                "rows": 252,
                            },
                            {
                                "ticker": "MSFT",
                                "status": "partial_window",
                                "usable": True,
                                "full_window": False,
                                "rows": 120,
                            },
                            {
                                "ticker": "NVDA",
                                "status": "no_data",
                                "usable": False,
                                "full_window": False,
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

            def load_bars(self, symbols=None, **kwargs):
                assert symbols == ["SONY", "AAPL"]
                assert kwargs["start"] == "2020-01-01"
                assert kwargs["limit"] == 600
                return pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            [
                                "2020-01-01",
                                "2020-01-01",
                                "2020-01-02",
                                "2020-01-02",
                            ],
                            utc=True,
                        ),
                        "symbol": ["SONY", "AAPL", "SONY", "AAPL"],
                        "close": [100.0, 50.0, 101.0, 51.0],
                    }
                )

        engine = DemoEngine(
            context={
                "discovery": {
                    "ticker": "SONY",
                    "parents": [{"ticker": "AAPL", "field": "price"}],
                    "data_readiness": {
                        "results": [
                            {"ticker": "AAPL", "status": "full_window", "usable": True, "full_window": True},
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

            def load_bars(self, symbols=None, **kwargs):
                assert symbols == ["SONY", "AAPL", "MSFT"]
                return pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            [
                                "2020-01-01",
                                "2020-01-01",
                                "2020-01-01",
                                "2020-01-02",
                                "2020-01-02",
                                "2020-01-03",
                                "2020-01-03",
                                "2020-01-03",
                            ],
                            utc=True,
                        ),
                        "symbol": ["SONY", "AAPL", "MSFT", "SONY", "MSFT", "SONY", "AAPL", "MSFT"],
                        "close": [100.0, 50.0, 20.0, 101.0, 21.0, 102.0, 52.0, 22.0],
                    }
                )

        engine = DemoEngine(
            context={
                "discovery": {
                    "ticker": "SONY",
                    "parents": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
                    "data_readiness": {
                        "results": [
                            {"ticker": "AAPL", "status": "full_window", "usable": True, "full_window": True},
                            {"ticker": "MSFT", "status": "full_window", "usable": True, "full_window": True},
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
        assert summary["full_window_count"] == 1
        assert summary["partial_window_count"] == 1
        assert summary["no_data_count"] == 1
        assert summary["error_count"] == 1
        assert report["probe_limit"] == 500
        assert report["recommended_starts"]["target_recommended_start"] == "2020-01-01"
        assert report["recommended_starts"]["common_recommended_start"] == "2020-08-01"

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
