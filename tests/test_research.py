"""Tests for raw evaluation helpers."""

from pathlib import Path

from click.testing import CliRunner

from causal_edge.cli import main
from causal_edge.research.evaluate import (
    check_look_ahead,
    compute_k,
    render_validation_markdown,
    run_evaluation,
)


def _write_strategy(path: Path, *, bias: float = 0.02, flat: bool = False) -> None:
    if flat:
        body = (
            '    dates = pd.date_range("2024-01-01", periods=60, freq="D")\n'
            "    pnl = np.full(60, 0.01)\n"
            "    positions = np.ones(60)\n"
        )
    else:
        body = (
            '    dates = pd.date_range("2024-01-01", periods=120, freq="D")\n'
            "    phase = np.linspace(0, 8 * np.pi, 120)\n"
            f"    pnl = {bias} + 0.012 * np.sin(phase)\n"
            "    positions = np.ones(120)\n"
        )

    path.write_text(
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "def run_strategy():\n" + body + "    return pnl, dates, positions\n",
        encoding="utf-8",
    )


class TestComputeK:
    def test_counts_tickers_and_lags(self, tmp_path):
        strategy = tmp_path / "strategy.py"
        strategy.write_text(
            'PARENTS = [("BTCUSD", 3), ("ETHUSD", 21)]\n'
            "def run_strategy():\n"
            "    feature.shift(3)\n"
            "    feature.shift(21)\n",
            encoding="utf-8",
        )
        k_value, tickers, lags = compute_k(strategy)
        assert k_value >= 4
        assert "BTCUSD" in tickers
        assert 21 in lags

    def test_filters_market_factors(self, tmp_path):
        strategy = tmp_path / "strategy.py"
        strategy.write_text(
            'tickers = ["BTCUSD", "SPY"]\ndef run_strategy():\n    feature.shift(5)\n',
            encoding="utf-8",
        )
        _, tickers, _ = compute_k(strategy)
        assert "SPY" not in tickers


class TestLookAhead:
    def test_static_check_catches_rolling_without_shift(self, tmp_path):
        strategy = tmp_path / "strategy.py"
        strategy.write_text("x = ret.rolling(20).mean()\npositions = x > 0", encoding="utf-8")
        assert check_look_ahead(strategy)


class TestRunEvaluation:
    def test_missing_strategy_file(self, tmp_path):
        result = run_evaluation(tmp_path)
        assert result["verdict"] == "ERROR"

    def test_unimplemented_strategy_returns_error(self, tmp_path):
        strategy = tmp_path / "strategy.py"
        strategy.write_text(
            'def run_strategy():\n    raise NotImplementedError("todo")\n',
            encoding="utf-8",
        )
        result = run_evaluation(tmp_path)
        assert result["verdict"] == "ERROR"
        assert any("failed" in failure.lower() for failure in result["failures"])

    def test_viable_strategy_passes(self, tmp_path):
        _write_strategy(tmp_path / "strategy.py")
        result = run_evaluation(tmp_path)
        assert result["verdict"] == "PASS"
        assert result["K"] >= 1


class TestValidationMarkdown:
    def test_renders_validation_summary(self, tmp_path):
        _write_strategy(tmp_path / "strategy.py")
        result = run_evaluation(tmp_path)
        report = render_validation_markdown(result)
        assert "# Evaluation Summary" in report
        assert "## Verdict" in report
        assert result["verdict"] in report


class TestEvaluateCli:
    def test_evaluate_cli_writes_raw_outputs(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            workdir = Path("workspace")
            workdir.mkdir()
            _write_strategy(workdir / "strategy.py")

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
                ],
            )
            assert result.exit_code == 0, result.output
            assert (workdir / "edge-result.json").exists()
            assert (workdir / "edge-validation.md").exists()
            assert "Verdict: PASS" in result.output

    def test_evaluate_cli_fails_for_bad_strategy(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            workdir = Path("workspace")
            workdir.mkdir()
            _write_strategy(workdir / "strategy.py", flat=True)

            result = runner.invoke(main, ["evaluate", "--workdir", str(workdir)])
            assert result.exit_code != 0
            assert "Verdict: FAIL" in result.output
