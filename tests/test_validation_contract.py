from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
import pandas as pd

from abel_edge.cli import main
from abel_edge.validation.gate import validate_strategy


FIXTURES = Path(__file__).parent / "fixtures" / "validation"


def _write_grandma_csv(path: Path, *, position: float = 1.0) -> None:
    dates = pd.bdate_range("2024-01-02", periods=60)
    pnl = [0.005] * 60
    pnl[10] = -0.02
    frame = pd.DataFrame(
        {
            "date": dates,
            "pnl": pnl,
            "position": [position] * 60,
            "asset_return": [value / position for value in pnl],
        }
    )
    frame.to_csv(path, index=False)


def test_csv_insufficient_rows_contract() -> None:
    result = validate_strategy(FIXTURES / "insufficient_rows.csv")
    assert result["verdict"] == "FAIL"
    assert result["score"] == "0/0"
    assert result["profile"] == "unknown"
    assert result["triangle"] == {"ratio": 0, "rank": 0, "shape": 0}
    assert result["failures"] == ["Insufficient data: 20 rows (need 30+)"]


def test_grandma_daily_profile_passes_simple_return_with_no_leverage(tmp_path: Path) -> None:
    csv_path = tmp_path / "grandma-pass.csv"
    _write_grandma_csv(csv_path, position=1.0)

    result = validate_strategy(csv_path, profile="grandma_daily")

    assert result["verdict"] == "PASS"
    assert result["score"] == "3/3"
    assert result["profile"] == "grandma_daily"
    assert result["metrics"]["pnl_to_maxdd"] >= 1.5
    assert result["metrics"]["max_abs_position"] <= 1.0


def test_grandma_daily_profile_rejects_levered_position(tmp_path: Path) -> None:
    csv_path = tmp_path / "grandma-levered.csv"
    _write_grandma_csv(csv_path, position=1.25)

    result = validate_strategy(csv_path, profile="grandma_daily")

    assert result["verdict"] == "FAIL"
    assert any("Grandma leverage" in item for item in result["failures"])


def test_csv_without_position_has_current_conditional_denominator() -> None:
    result = validate_strategy(FIXTURES / "ic_unsupported_no_position.csv", profile="equity_daily")
    assert result["verdict"] == "FAIL"
    assert result["score"] == "4/5"
    assert result["metrics"]["position_ic_applicable"] is False
    assert result["metrics"]["loss_years_applicable"] is False
    assert result["metrics"]["omega_applicable"] is False
    assert result["triangle"]["rank"] == 0.0
    assert result["profile"] == "equity_daily"


def test_csv_with_position_marks_ic_family_applicable() -> None:
    result = validate_strategy(FIXTURES / "ic_supported.csv", profile="equity_daily")
    assert result["metrics"]["position_ic_applicable"] is True
    assert result["metrics"]["position_ic_stability_applicable"] is False
    assert result["score"] == "7/7"


def test_position_aware_csv_without_ic_failures_uses_15_test_contract() -> None:
    result = validate_strategy(FIXTURES / "positive_daily.csv", profile="equity_daily")
    assert result["metrics"]["position_ic_applicable"] is True
    assert result["metrics"]["position_ic_stability_applicable"] is False
    assert result["metrics"]["loss_years_applicable"] is False
    assert result["metrics"]["omega_applicable"] is False
    assert result["score"] == "5/6"


def test_csv_without_position_omits_ic_gate_labels_from_failures() -> None:
    result = validate_strategy(FIXTURES / "ic_unsupported_no_position.csv", profile="equity_daily")
    joined = " | ".join(result["failures"])
    assert "IC " not in joined
    assert "IC stab" not in joined
    assert "T7 PBO" not in joined
    assert "T13 NegRoll" not in joined
    assert "T13 DrawdownTime" not in joined
    assert "T13 MaxDDDuration" not in joined
    assert "T14 LossYrs" not in joined
    assert "T15 Omega" not in joined
    assert "T12 OOS/IS" not in joined


def test_removed_oos_split_sharpe_metrics_are_absent() -> None:
    result = validate_strategy(FIXTURES / "positive_daily.csv", profile="equity_daily")
    assert "oos_is" not in result["metrics"]
    assert "is_sharpe" not in result["metrics"]
    assert "oos_sharpe" not in result["metrics"]


def test_verbose_output_includes_current_metric_section() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "validate",
            "--csv",
            str(FIXTURES / "ic_unsupported_no_position.csv"),
            "--verbose",
        ],
    )
    assert result.exit_code == 1
    assert "ic_unsupported_no_position     4/5  FAIL" in result.output
    assert "ic_unsupported_no_position metrics:" in result.output
    assert "sharpe" in result.output
    assert "dsr_trials_used" in result.output
    assert "position_hit_rate" not in result.output
    assert "pbo" not in result.output
    assert "oos_is" not in result.output


def test_validate_cli_accepts_dsr_trials_override() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "validate",
            "--csv",
            str(FIXTURES / "ic_unsupported_no_position.csv"),
            "--verbose",
            "--dsr-trials",
            "17",
        ],
    )
    assert result.exit_code == 1
    assert "dsr_trials_used      17.0000" in result.output


def test_validate_cli_rejects_non_positive_dsr_trials() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "validate",
            "--csv",
            str(FIXTURES / "ic_unsupported_no_position.csv"),
            "--dsr-trials",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "0 is not in the range" in result.output


def test_export_output_matches_current_report_contract(tmp_path) -> None:
    export_path = tmp_path / "report.txt"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "validate",
            "--csv",
            str(FIXTURES / "ic_unsupported_no_position.csv"),
            "--export",
            str(export_path),
        ],
    )
    assert result.exit_code == 1
    exported = export_path.read_text(encoding="utf-8")
    assert "ABEL PROOF VALIDATION REPORT" in exported
    assert "ic_unsupported_no_position     4/5  FAIL" in exported
    assert "Annualized return floor" in exported
    assert "< +5.00%" in exported
    assert "Report exported to" in result.output


def test_contract_drift_public_claim_for_15_test_validation() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "validate",
            "--csv",
            str(FIXTURES / "ic_unsupported_no_position.csv"),
        ],
    )
    assert result.exit_code == 1
    import re

    match = re.search(r"(\d+)/(\d+)", result.output)
    assert match, f"No score found in output: {result.output!r}"
    denominator = int(match.group(2))
    assert denominator == 5
