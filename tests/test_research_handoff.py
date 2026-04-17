"""Tests for the strategy handoff contract."""

import json
from pathlib import Path

from click.testing import CliRunner

from causal_edge.cli import main
from causal_edge.research.handoff import HANDOFF_CONTRACT


def _write_strategy(path: Path, *, bias: float = 0.02) -> None:
    path.write_text(
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "def run_strategy(*, start=None):\n"
        "    start = start or '2024-01-01'\n"
        "    dates = pd.date_range(start, periods=120, freq='D')\n"
        f"    pnl = {bias} + 0.012 * np.sin(np.linspace(0, 8 * np.pi, 120))\n"
        "    positions = np.ones(120)\n"
        "    return pnl, dates, positions\n",
        encoding="utf-8",
    )


def test_evaluate_cli_writes_edge_handoff_and_validate_handoff_accepts(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        workdir = Path("workspace")
        workdir.mkdir()
        _write_strategy(workdir / "strategy.py", bias=0.02)

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
                "--output-handoff",
                str(workdir / "edge-handoff.json"),
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads((workdir / "edge-handoff.json").read_text(encoding="utf-8"))
        assert payload["contract"] == HANDOFF_CONTRACT
        assert payload["verdict"] == "PASS"
        assert payload["profile"] == "equity_daily"
        assert payload["blocking_failures"] == []

        accepted = runner.invoke(main, ["validate-handoff", str(workdir / "edge-handoff.json")])
        assert accepted.exit_code == 0, accepted.output
        assert "Handoff accepted." in accepted.output


def test_validate_handoff_rejects_missing_required_fields(tmp_path):
    handoff = tmp_path / "edge-handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "contract": HANDOFF_CONTRACT,
                "strategy_path": "strategy.py",
                "verdict": "PASS",
                "blocking_failures": [],
                "edge_result_path": "edge-result.json",
                "edge_report_path": "edge-validation.md",
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["validate-handoff", str(handoff)])
    assert result.exit_code != 0
    assert "Missing required fields: profile" in result.output


def test_validate_handoff_rejects_result_mismatch(tmp_path):
    strategy = tmp_path / "strategy.py"
    strategy.write_text("def run_strategy():\n    raise NotImplementedError\n", encoding="utf-8")
    report = tmp_path / "edge-validation.md"
    report.write_text("# report\n", encoding="utf-8")
    edge_result = tmp_path / "edge-result.json"
    edge_result.write_text(
        json.dumps({"verdict": "FAIL", "profile": "equity_daily", "failures": ["bad"]}),
        encoding="utf-8",
    )
    handoff = tmp_path / "edge-handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "contract": HANDOFF_CONTRACT,
                "strategy_path": "strategy.py",
                "verdict": "PASS",
                "profile": "equity_daily",
                "blocking_failures": [],
                "edge_result_path": "edge-result.json",
                "edge_report_path": "edge-validation.md",
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["validate-handoff", str(handoff)])
    assert result.exit_code != 0
    assert "handoff verdict does not match edge_result_path verdict." in result.output
    assert "blocking_failures does not match edge_result_path failures." in result.output
