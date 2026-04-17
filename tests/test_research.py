"""Tests for engine-backed research evaluation helpers."""

from pathlib import Path

from click.testing import CliRunner

from causal_edge.cli import main
from causal_edge.research.evaluate import (
    check_look_ahead,
    compute_k,
    render_validation_markdown,
    run_evaluation,
)
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
