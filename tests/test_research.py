"""Tests for the research workflow."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from causal_edge.cli import main
from causal_edge.research.evaluate import (
    append_results_tsv,
    check_look_ahead,
    compute_k,
    decide_research_outcome,
    run_evaluation,
)
from causal_edge.research.workspace import init_workspace, read_results_rows


def _write_viable_strategy(path: Path, *, bias: float = 0.018) -> None:
    path.write_text(
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "def run_strategy():\n"
        '    dates = pd.date_range("2024-01-01", periods=120, freq="D")\n'
        "    phase = np.linspace(0, 8 * np.pi, 120)\n"
        f"    pnl = {bias} + 0.012 * np.sin(phase)\n"
        "    positions = np.ones(120)\n"
        "    return pnl, dates, positions\n",
        encoding="utf-8",
    )


class TestInitWorkspace:
    def test_creates_session_and_branch_files(self, tmp_path):
        workspace = init_workspace(
            "SOLUSD",
            tmp_path / "research" / "solusd" / "20260415-120000" / "branches" / "baseline",
        )
        assert (workspace / "strategy.py").exists()
        assert (workspace / "results.tsv").exists()
        assert (workspace / "memory.md").exists()
        assert (workspace / "README.md").exists()
        assert (workspace / "thesis.md").exists()
        assert (workspace / "rounds").exists()
        assert (workspace.parent.parent / "README.md").exists()
        assert (workspace.parent.parent / "discovery.json").exists()
        assert (workspace.parent.parent / "events.tsv").exists()

    def test_init_can_add_second_branch_to_same_session(self, tmp_path):
        first = init_workspace(
            "SOL", tmp_path / "research" / "sol" / "20260415-120000" / "branches" / "baseline"
        )
        second = init_workspace(
            "SOL", tmp_path / "research" / "sol" / "20260415-120000" / "branches" / "alt-v2"
        )
        assert first.parent == second.parent
        assert second.name == "alt-v2"
        events_text = (first.parent.parent / "events.tsv").read_text(encoding="utf-8")
        assert "branch_created" in events_text
        assert "alt-v2" in events_text


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


class TestAppendResults:
    def test_keep_requires_pass(self, tmp_path):
        workspace = init_workspace(
            "TEST", tmp_path / "research" / "test" / "20260415-120000" / "branches" / "baseline"
        )
        with pytest.raises(ValueError, match="Cannot KEEP"):
            append_results_tsv(
                workspace,
                {"verdict": "FAIL", "score": "4/5", "metrics": {}},
                "keep",
                "exploit",
                "test",
                exp_id="20260415-120000",
                ticker="TEST",
                branch_id="baseline",
                round_id="round-001",
                decision="keep",
                validation_path="outputs/round-001-validation.md",
            )


class TestResearchDecision:
    def test_first_pass_is_keep(self, tmp_path):
        workspace = init_workspace(
            "TEST", tmp_path / "research" / "test" / "20260415-120000" / "branches" / "baseline"
        )
        result = {
            "verdict": "PASS",
            "metrics": {
                "lo_adjusted": 1.5,
                "position_ic": 0.05,
                "omega": 1.2,
                "sharpe": 1.8,
                "max_dd": -0.1,
                "total_return": 0.3,
            },
        }
        status, decision = decide_research_outcome(workspace, result)
        assert (status, decision) == ("keep", "keep")

    def test_pass_without_improvement_is_discard(self, tmp_path):
        workspace = init_workspace(
            "TEST", tmp_path / "research" / "test" / "20260415-120000" / "branches" / "baseline"
        )
        append_results_tsv(
            workspace,
            {
                "verdict": "PASS",
                "score": "5/5",
                "metrics": {
                    "lo_adjusted": 2.0,
                    "position_ic": 0.1,
                    "omega": 2.0,
                    "sharpe": 2.0,
                    "max_dd": -0.1,
                    "total_return": 0.4,
                },
            },
            "keep",
            "explore",
            "baseline",
            exp_id="20260415-120000",
            ticker="TEST",
            branch_id="baseline",
            round_id="round-001",
            decision="keep",
            validation_path="outputs/round-001-validation.md",
        )
        result = {
            "verdict": "PASS",
            "metrics": {
                "lo_adjusted": 1.9,
                "position_ic": 0.1,
                "omega": 2.0,
                "sharpe": 2.1,
                "max_dd": -0.1,
                "total_return": 0.4,
            },
        }
        status, decision = decide_research_outcome(workspace, result)
        assert (status, decision) == ("discard", "discard")


class TestRunEvaluation:
    def test_missing_strategy_file(self, tmp_path):
        result = run_evaluation(tmp_path)
        assert result["verdict"] == "ERROR"

    def test_unimplemented_strategy_returns_error(self, tmp_path):
        workspace = init_workspace(
            "TEST", tmp_path / "research" / "test" / "20260415-120000" / "branches" / "baseline"
        )
        result = run_evaluation(workspace)
        assert result["verdict"] == "ERROR"
        assert any("failed" in failure.lower() for failure in result["failures"])


class TestResearchCli:
    def test_research_init_run_status_and_check(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                main,
                [
                    "research",
                    "init",
                    "ETHUSD",
                    "--exp-id",
                    "20260415-120000",
                    "--branch-id",
                    "graph-v1",
                ],
            )
            assert result.exit_code == 0, result.output
            branch_dir = Path("research/ethusd/20260415-120000/branches/graph-v1")
            assert (branch_dir / "strategy.py").exists()
            assert Path("research/ethusd/20260415-120000/events.tsv").exists()
            branch_dir.joinpath("memory.md").write_text(
                "# ETHUSD Research Memory\n\n"
                "## K Budget\n"
                "- Discovery: K=? (fill after discovery)\n\n"
                "## Baseline\n"
                "- (none yet)\n",
                encoding="utf-8",
            )

            _write_viable_strategy(branch_dir / "strategy.py")

            first_run = runner.invoke(
                main,
                [
                    "research",
                    "run",
                    "--workdir",
                    str(branch_dir),
                    "--mode",
                    "explore",
                    "-d",
                    "baseline branch",
                    "--hypothesis",
                    "Abel parents can improve trend timing",
                    "--action",
                    "Ran validation",
                ],
            )
            assert first_run.exit_code == 0, first_run.output
            assert (branch_dir / "rounds" / "round-001.md").exists()
            assert (branch_dir / "outputs" / "round-001-validation.md").exists()
            rows = read_results_rows(branch_dir)
            assert rows[-1]["round_id"] == "round-001"
            assert rows[-1]["branch_id"] == "graph-v1"

            _write_viable_strategy(branch_dir / "strategy.py", bias=0.020)
            second_run = runner.invoke(
                main,
                [
                    "research",
                    "run",
                    "--workdir",
                    str(branch_dir),
                    "--mode",
                    "exploit",
                    "-d",
                    "baseline refinement",
                    "--action",
                    "Adjusted pnl path",
                ],
            )
            assert second_run.exit_code == 0, second_run.output
            assert "Recent activity" in second_run.output
            assert "graph-v1" in second_run.output

            status = runner.invoke(
                main, ["research", "status", "--workdir", "research/ethusd/20260415-120000"]
            )
            assert status.exit_code == 0, status.output
            assert "Branches: 1" in status.output
            assert "graph-v1" in status.output

            session_readme = Path("research/ethusd/20260415-120000/README.md").read_text(
                encoding="utf-8"
            )
            assert "Document why this exploration started" not in session_readme
            assert (
                "Record how many candidate branches this exploration produced"
                not in session_readme
            )
            assert "State the next exploration move for this session." not in session_readme
            assert "## Executive Summary" in session_readme
            assert "## Branch Outcome Snapshot" in session_readme
            assert "Trend: Lo" in session_readme
            assert "Current lead is `graph-v1`" in session_readme

            memory_text = (branch_dir / "memory.md").read_text(encoding="utf-8")
            assert "fill after discovery" not in memory_text

            branch_readme = (branch_dir / "README.md").read_text(encoding="utf-8")
            assert "## Decision Rationale" in branch_readme
            assert "## Metric Progression" in branch_readme
            assert "latest_hypothesis" in branch_readme
            assert "dSharpe" in branch_readme

            thesis_text = (branch_dir / "thesis.md").read_text(encoding="utf-8")
            assert "Describe the causal or market thesis." not in thesis_text
            assert (
                "List the target asset, related assets, and any special data inputs."
                not in thesis_text
            )
            assert "List the largest assumptions and failure modes." not in thesis_text
            assert "Abel CAP (live)" in thesis_text or "template" in thesis_text
            assert "Abel parents can improve trend timing" in thesis_text

            check = runner.invoke(
                main, ["research", "check", "--workdir", "research/ethusd/20260415-120000"]
            )
            assert check.exit_code == 0, check.output
            assert "Research check passed" in check.output

            strict_check = runner.invoke(
                main,
                [
                    "research",
                    "check",
                    "--strict",
                    "--workdir",
                    "research/ethusd/20260415-120000",
                ],
            )
            assert strict_check.exit_code == 0, strict_check.output

            events_text = Path("research/ethusd/20260415-120000/events.tsv").read_text(
                encoding="utf-8"
            )
            assert "branch_created" in events_text
            assert "round_recorded" in events_text
