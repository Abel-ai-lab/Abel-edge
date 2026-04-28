"""Tests for the strategy handoff contract."""

import json
from pathlib import Path

from click.testing import CliRunner

from abel_edge.cli import main
from abel_edge.research.handoff import HANDOFF_CONTRACT


def _write_engine(path: Path, *, bias: float = 0.02) -> None:
    path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "import pandas as pd",
                "",
                "from abel_edge.engine.base import StrategyEngine",
                "",
                "",
                "class BranchEngine(StrategyEngine):",
                "    def compute_signals(self):",
                "        requested = ((self.context or {}).get('_research') or {}).get('requested_window') or {}",
                "        start = requested.get('start') or '2024-01-01'",
                "        dates = pd.date_range(start, periods=120, freq='D', tz='UTC')",
                "        phase = np.linspace(0, 8 * np.pi, 120)",
                "        positions = np.where(np.sin(phase) > 0, 1.0, -1.0)",
                f"        returns = {bias} * positions + 0.002 * np.sin(phase)",
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


def test_evaluate_cli_writes_edge_handoff_and_validate_handoff_accepts(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        workdir = Path("workspace")
        workdir.mkdir()
        _write_engine(workdir / "engine.py", bias=0.02)

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
        assert payload["strategy_path"] == "engine.py"

        accepted = runner.invoke(main, ["validate-handoff", str(workdir / "edge-handoff.json")])
        assert accepted.exit_code == 0, accepted.output
        assert "Handoff accepted." in accepted.output


def test_validate_handoff_rejects_missing_required_fields(tmp_path):
    handoff = tmp_path / "edge-handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "contract": HANDOFF_CONTRACT,
                "strategy_path": "engine.py",
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
    engine = tmp_path / "engine.py"
    engine.write_text("# engine\n", encoding="utf-8")
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
                "strategy_path": "engine.py",
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
