"""Tests for the research workflow."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from causal_edge.cli import main
from causal_edge.research.evaluate import append_results_tsv, check_look_ahead, compute_k, run_evaluation
from causal_edge.research.workspace import init_workspace


class TestInitWorkspace:
    def test_creates_files(self, tmp_path):
        workspace = init_workspace("SOLUSD", tmp_path / "sol")
        assert (workspace / "strategy.py").exists()
        assert (workspace / "results.tsv").exists()
        assert (workspace / "memory.md").exists()
        assert (workspace / "discovery.json").exists()

    def test_idempotent_strategy_file(self, tmp_path):
        workspace = init_workspace("SOL", tmp_path / "sol")
        (workspace / "strategy.py").write_text("# modified", encoding="utf-8")
        init_workspace("SOL", tmp_path / "sol")
        assert (workspace / "strategy.py").read_text(encoding="utf-8") == "# modified"


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
            'tickers = ["BTCUSD", "SPY"]\n'
            "def run_strategy():\n"
            "    feature.shift(5)\n",
            encoding="utf-8",
        )
        _, tickers, _ = compute_k(strategy)
        assert "SPY" not in tickers


class TestLookAhead:
    def test_static_check_catches_rolling_without_shift(self, tmp_path):
        strategy = tmp_path / "strategy.py"
        strategy.write_text("x = ret.rolling(20).mean()\npositions = x > 0", encoding="utf-8")
        assert check_look_ahead(strategy)


class TestAppendResults:
    def test_keep_requires_pass(self, tmp_path):
        workspace = init_workspace("TEST", tmp_path / "test")
        with pytest.raises(ValueError, match="Cannot KEEP"):
            append_results_tsv(
                workspace,
                {"verdict": "FAIL", "score": "4/5", "metrics": {}},
                "keep",
                "exploit",
                "test",
            )


class TestRunEvaluation:
    def test_missing_strategy_file(self, tmp_path):
        result = run_evaluation(tmp_path)
        assert result["verdict"] == "ERROR"

    def test_unimplemented_strategy_returns_error(self, tmp_path):
        workspace = init_workspace("TEST", tmp_path / "workspace")
        result = run_evaluation(workspace)
        assert result["verdict"] == "ERROR"
        assert any("failed" in failure.lower() for failure in result["failures"])


class TestResearchCli:
    def test_research_init_and_status(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["research", "init", "ETHUSD"])
            assert result.exit_code == 0, result.output
            assert Path("research/ethusd/strategy.py").exists()

            status = runner.invoke(main, ["research", "status", "--workdir", "research/ethusd"])
            assert status.exit_code == 0, status.output
            assert "Experiments: 0" in status.output
