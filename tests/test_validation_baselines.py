from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from causal_edge.validation.gate import validate_strategy
from causal_edge.validation.metrics import compute_all_metrics, detect_profile, load_profile


FIXTURES = Path(__file__).parent / "fixtures" / "validation"


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / name, parse_dates=["date"])


def _compute(name: str) -> dict:
    df = _load_csv(name)
    profile = load_profile("equity_daily")
    asset_returns = df["asset_return"].to_numpy() if "asset_return" in df.columns else None
    if "position" in df.columns:
        return compute_all_metrics(
            df["pnl"].to_numpy(),
            pd.DatetimeIndex(df["date"]),
            df["position"].to_numpy(),
            profile,
            asset_returns=asset_returns,
        )
    return compute_all_metrics(
        df["pnl"].to_numpy(),
        pd.DatetimeIndex(df["date"]),
        asset_returns=asset_returns,
        profile=profile,
    )


def test_positive_daily_baselines() -> None:
    metrics = _compute("positive_daily.csv")
    assert metrics["sharpe"] == pytest.approx(262.59971101022364, rel=1e-9)
    assert metrics["sortino"] == 0.0
    assert metrics["max_dd"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["calmar"] == 0.0
    assert metrics["omega"] == 0.0
    assert metrics["omega_applicable"] is False


def test_clipped_fixture_changes_shape_contract() -> None:
    raw = _compute("positive_daily.csv")
    clipped = _compute("positive_clipped.csv")
    assert clipped["sharpe"] == 0.0
    assert clipped["omega"] == pytest.approx(raw["omega"], rel=1e-9)
    assert clipped["omega_applicable"] is False
    assert clipped["skew"] == 0.0


def test_autocorrelated_fixture_has_elevated_sharpe_lo_ratio() -> None:
    metrics = _compute("autocorrelated.csv")
    assert metrics["sharpe_lo_ratio"] > 1.0


def test_sharpe_annualization_respects_profile_periods_per_year() -> None:
    df = _load_csv("positive_daily.csv")
    pnl = df["pnl"].to_numpy()
    dates = pd.DatetimeIndex(df["date"])
    positions = df["position"].to_numpy()
    equity = compute_all_metrics(pnl, dates, positions, load_profile("equity_daily"))
    crypto = compute_all_metrics(pnl, dates, positions, load_profile("crypto_daily"))
    expected = (365 / 252) ** 0.5
    assert crypto["sharpe"] / equity["sharpe"] == pytest.approx(expected, rel=1e-9)


def test_dsr_respects_profile_specific_k() -> None:
    df = _load_csv("positive_daily.csv")
    pnl = df["pnl"].to_numpy()
    dates = pd.DatetimeIndex(df["date"])
    positions = df["position"].to_numpy()
    equity = compute_all_metrics(pnl, dates, positions, load_profile("equity_daily"))
    crypto = compute_all_metrics(pnl, dates, positions, load_profile("crypto_daily"))
    hft = compute_all_metrics(pnl, dates, positions, load_profile("hft"))
    assert 0.0 <= equity["dsr"] <= 1.0
    assert 0.0 <= crypto["dsr"] <= 1.0
    assert 0.0 <= hft["dsr"] <= 1.0
    assert equity["dsr_trials_used"] == load_profile("equity_daily")["validation"]["dsr_K"]
    assert crypto["dsr_trials_used"] == load_profile("crypto_daily")["validation"]["dsr_K"]
    assert hft["dsr_trials_used"] == load_profile("hft")["validation"]["dsr_K"]


def test_explicit_dsr_trials_override_profile_default() -> None:
    df = _load_csv("positive_daily.csv")
    pnl = df["pnl"].to_numpy()
    dates = pd.DatetimeIndex(df["date"])
    positions = df["position"].to_numpy()
    metrics = compute_all_metrics(
        pnl,
        dates,
        positions,
        load_profile("equity_daily"),
        dsr_trials=17,
    )
    assert metrics["dsr_trials_used"] == 17


def test_bootstrap_uses_profile_trial_count() -> None:
    df = _load_csv("positive_daily.csv")
    pnl = df["pnl"].to_numpy()
    dates = pd.DatetimeIndex(df["date"])
    positions = df["position"].to_numpy()
    equity = compute_all_metrics(pnl, dates, positions, load_profile("equity_daily"))
    hft = compute_all_metrics(pnl, dates, positions, load_profile("hft"))
    assert 0.0 <= equity["bootstrap_p"] <= 1.0
    assert 0.0 <= hft["bootstrap_p"] <= 1.0


def test_detect_profile_equity_for_positive_daily_fixture() -> None:
    df = _load_csv("positive_daily.csv")
    detected = detect_profile(
        df["pnl"].to_numpy(),
        pd.DatetimeIndex(df["date"]),
        asset_returns=df["asset_return"].to_numpy(),
    )
    assert detected == "equity_daily"


def test_position_ic_supported_fixture_computes_positive_ic() -> None:
    metrics = _compute("ic_supported.csv")
    assert metrics["position_ic_applicable"] is True
    assert metrics["position_ic"] > 0.95
    assert metrics["position_hit_rate"] == pytest.approx(1.0, rel=1e-9)
    assert metrics["position_ic_stability_applicable"] is False


def test_position_ic_unsupported_fixture_keeps_family_zero_without_position() -> None:
    metrics = _compute("ic_unsupported_no_position.csv")
    assert metrics["position_ic_applicable"] is False
    assert metrics["position_ic"] == 0.0
    assert metrics["position_hit_rate"] == 0.0
    assert metrics["position_ic_stability"] == 0.0
    assert metrics["position_ic_monthly_mean"] == 0.0


def test_loss_years_requires_full_calendar_year() -> None:
    metrics = _compute("positive_daily.csv")
    assert metrics["loss_years_applicable"] is False
    assert metrics["full_years_count"] == 0
    assert metrics["loss_years"] == 0


def test_yearly_pnl_is_exposed_for_audit() -> None:
    metrics = _compute("positive_daily.csv")
    assert "yearly_pnl" in metrics
    assert 2020 in metrics["yearly_pnl"]


def test_insufficient_rows_csv_fails_validation_contract() -> None:
    result = validate_strategy(FIXTURES / "insufficient_rows.csv")
    assert result["verdict"] == "FAIL"
    assert result["score"] == "0/0"
    assert "need 30+" in result["failures"][0]


def test_defer_candidate_metrics_are_not_gate_failures() -> None:
    result = validate_strategy(FIXTURES / "ic_unsupported_no_position.csv", profile="equity_daily")
    joined = " | ".join(result["failures"])
    assert "hill_alpha" not in result["metrics"]
    assert "cvar_var_ratio" not in result["metrics"]
    assert "hill_alpha" not in joined
    assert "cvar_var_ratio" not in joined
    assert "T7 PBO" not in joined
    assert "T13 NegRoll" not in joined
    assert "T13 DrawdownTime" not in joined
    assert "T13 MaxDDDuration" not in joined


def test_public_claim_denominator_drift_is_visible() -> None:
    result = validate_strategy(FIXTURES / "positive_daily.csv", profile="equity_daily")
    denominator = int(result["score"].split("/")[1])
    assert denominator == 6


def test_removed_oos_family_metrics_are_not_in_payload() -> None:
    metrics = _compute("positive_daily.csv")
    assert "oos_is" not in metrics
    assert "is_sharpe" not in metrics
    assert "oos_sharpe" not in metrics


def test_removed_pbo_metric_is_not_in_payload() -> None:
    metrics = _compute("positive_daily.csv")
    assert "pbo" not in metrics


def test_drawdown_time_metrics_are_in_payload() -> None:
    metrics = _compute("positive_daily.csv")
    assert "drawdown_time_frac" in metrics
    assert "max_drawdown_duration_bars" in metrics
    assert "neg_roll_frac" not in metrics


def test_validate_strategy_reports_explicit_dsr_trials_used() -> None:
    result = validate_strategy(
        FIXTURES / "positive_daily.csv", profile="equity_daily", dsr_trials=23
    )
    assert result["metrics"]["dsr_trials_used"] == 23


def test_fixture_files_are_deterministic() -> None:
    hashes = {}
    for path in sorted(FIXTURES.glob("*.csv")):
        raw = path.read_bytes()
        df1 = pd.read_csv(path)
        df2 = pd.read_csv(path)
        assert list(df1.columns) == list(df2.columns)
        assert df1.equals(df2)
        hashes[path.name] = sha256(raw).hexdigest()
    assert len(hashes) == 7
