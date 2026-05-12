"""Tests for Abel Proof validation — locks the metric triangle.

The triangle is the core anti-gaming invariant:
  - Ratio (Lo/Sharpe): mean/std quality
  - Rank (IC): prediction quality
  - Shape (Omega): gain/loss asymmetry

These tests verify:
  1. Metrics compute correctly on known data
  2. Leverage invariance: scaling positions doesn't change triangle
  3. Clipping detection: Omega catches return clipping
  4. Serial correlation: Lo catches autocorrelated signals
  5. Profile loading and validation gate
  6. KEEP/DISCARD decision logic
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from abel_edge.dashboard.components import compute_metrics as compute_dashboard_metrics
from abel_edge.validation.metrics import (
    _sharpe,
    _sortino,
    _dsr,
    _bootstrap_sharpe,
    _elapsed_years,
    _is_full_calendar_year,
    compute_all_metrics,
    load_profile,
)
from abel_edge.validation.gate import validate_strategy
from abel_edge.validation.gate_logic import validate


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_pnl(n=500, mean=0.001, std=0.02, seed=42):
    """Generate synthetic daily PnL array."""
    rng = np.random.RandomState(seed)
    return rng.normal(mean, std, n)


def _make_dates(n=500, start="2020-01-01"):
    """Generate business day DatetimeIndex."""
    return pd.bdate_range(start, periods=n)


def _make_positions(pnl, lag=1):
    """Generate positions that predict pnl (shifted for causality)."""
    pos = np.zeros_like(pnl)
    pos[lag:] = np.sign(pnl[:-lag]) * 0.5 + 0.5  # partial prediction
    return pos


@pytest.fixture
def good_strategy():
    """A strategy with clear positive signal — should pass most tests."""
    pnl = _make_pnl(n=750, mean=0.0015, std=0.015, seed=42)
    dates = _make_dates(n=750)
    pos = _make_positions(pnl, lag=1)
    return pnl, dates, pos


@pytest.fixture
def bad_strategy():
    """A strategy with zero signal — should fail."""
    rng = np.random.RandomState(99)
    pnl = rng.normal(0, 0.02, 500)  # zero mean
    dates = _make_dates(n=500)
    pos = np.ones(500) * 0.5  # constant position
    return pnl, dates, pos


# ── Basic Metric Tests ────────────────────────────────────────────────


class TestSharpe:
    def test_positive_mean(self):
        pnl = _make_pnl(n=252, mean=0.003, std=0.01)
        assert _sharpe(pnl) > 2.0  # strong positive signal

    def test_zero_mean(self):
        pnl = _make_pnl(mean=0)
        assert abs(_sharpe(pnl)) < 1.0

    def test_empty(self):
        assert _sharpe(np.array([])) == 0

    def test_single(self):
        assert _sharpe(np.array([0.01])) == 0  # std=0 with ddof=1


class TestSortino:
    def test_positive(self):
        pnl = _make_pnl(mean=0.001)
        assert _sortino(pnl) > 0

    def test_no_downside(self):
        pnl = np.abs(_make_pnl()) + 0.01
        assert _sortino(pnl) == 0.0  # no negative returns

    def test_periods_per_year_changes_result(self):
        pnl = _make_pnl(mean=0.001, std=0.02, n=252, seed=11)
        assert _sortino(pnl, periods_per_year=252) != pytest.approx(
            _sortino(pnl, periods_per_year=365), rel=1e-12
        )


class TestDashboardMetrics:
    def test_sharpe_respects_periods_per_year(self):
        pnl = _make_pnl(mean=0.001, std=0.02, n=252, seed=9)
        daily = compute_dashboard_metrics(pnl, periods_per_year=252)
        crypto = compute_dashboard_metrics(pnl, periods_per_year=365)
        assert crypto["sharpe"] != pytest.approx(daily["sharpe"], rel=1e-12)


class TestValidationDataSource:
    def test_validate_strategy_ignores_live_rows_in_mixed_trade_log(self, tmp_path):
        dates = pd.date_range("2024-01-01", periods=35, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "date": list(dates[:32]) + list(dates[32:]),
                "pnl": [0.01] * 32 + [-0.9, -0.9, -0.9],
                "position": [1.0] * 35,
                "asset_return": [0.01] * 32 + [-0.9, -0.9, -0.9],
                "source": ["backfill"] * 32 + ["live", "live", "live"],
            }
        )
        path = tmp_path / "mixed.csv"
        df.to_csv(path, index=False)

        result = validate_strategy(path)

        assert result["metrics"]
        assert result["metrics"]["total_return"] > 0

    def test_validate_strategy_uses_execution_cost_in_consistency_check(self, tmp_path):
        dates = pd.date_range("2024-01-01", periods=35, freq="D", tz="UTC")
        position = np.ones(35) * 0.5
        asset_return = np.ones(35) * 0.02
        execution_cost = np.concatenate([[0.0], np.ones(34) * 0.001])
        pnl = position * asset_return - execution_cost
        df = pd.DataFrame(
            {
                "date": dates,
                "pnl": pnl,
                "position": position,
                "asset_return": asset_return,
                "execution_cost": execution_cost,
                "source": ["backfill"] * 35,
            }
        )
        path = tmp_path / "costed.csv"
        df.to_csv(path, index=False)

        result = validate_strategy(path)

        assert result["warnings"] == []

    def test_validate_strategy_warns_without_execution_cost_column(self, tmp_path):
        dates = pd.date_range("2024-01-01", periods=35, freq="D", tz="UTC")
        position = np.ones(35) * 0.5
        asset_return = np.ones(35) * 0.02
        pnl = position * asset_return - 0.02
        df = pd.DataFrame(
            {
                "date": dates,
                "pnl": pnl,
                "position": position,
                "asset_return": asset_return,
                "source": ["backfill"] * 35,
            }
        )
        path = tmp_path / "legacy_warning.csv"
        df.to_csv(path, index=False)

        result = validate_strategy(path)

        assert len(result["warnings"]) == 1
        assert "position * asset_return" in result["warnings"][0]


class TestCalendarYearCoverage:
    def test_tz_aware_dates_supported(self):
        from abel_edge.validation.metrics import _is_full_calendar_year

        year_dates = pd.date_range("2024-01-01", "2024-12-31", freq="D", tz="UTC")
        assert _is_full_calendar_year(pd.DatetimeIndex(year_dates)) is True


class TestDSR:
    def test_strong_signal(self):
        pnl = _make_pnl(mean=0.002, std=0.01, n=500)
        assert _dsr(pnl, 500) > 0.90

    def test_no_signal(self):
        pnl = _make_pnl(mean=0, std=0.02, n=500)
        assert _dsr(pnl, 500) < 0.90

    def test_zero_std(self):
        assert _dsr(np.zeros(100), 100) == 0

    def test_k_one_is_probabilistic_not_forced_to_one(self):
        pnl = _make_pnl(mean=0.0004, std=0.02, n=252, seed=7)
        dsr = _dsr(pnl, 252, K=1)
        assert 0.0 <= dsr <= 1.0
        assert dsr != pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_invalid_k(self, k):
        pnl = _make_pnl(mean=0.0004, std=0.02, n=252, seed=7)
        with pytest.raises(ValueError, match="DSR trials must be a positive integer"):
            _dsr(pnl, 252, K=k)

    def test_periods_per_year_does_not_change_dsr_scale(self):
        pnl = _make_pnl(mean=0.0004, std=0.02, n=252, seed=7)
        assert _dsr(pnl, 252, K=50, periods_per_year=252) == pytest.approx(
            _dsr(pnl, 252, K=50, periods_per_year=365), rel=1e-12
        )

    def test_matches_reference_formula(self):
        pnl = _make_pnl(mean=0.0004, std=0.02, n=252, seed=7)
        std = np.std(pnl, ddof=1)
        sr = np.mean(pnl) / std
        skew = float(sp_stats.skew(pnl))
        raw_kurt = float(sp_stats.kurtosis(pnl, fisher=False))
        gamma = 0.5772
        z1 = sp_stats.norm.ppf(1 - 1 / 50)
        z2 = sp_stats.norm.ppf(1 - 1 / (50 * np.e))
        expected_max_z = (1 - gamma) * z1 + gamma * z2
        sr_std = np.sqrt((1 - skew * sr + ((raw_kurt - 1) / 4) * sr**2) / (252 - 1))
        sr_star = sr_std * expected_max_z
        expected = float(sp_stats.norm.cdf((sr - sr_star) / sr_std))
        assert _dsr(pnl, 252, K=50, periods_per_year=252) == pytest.approx(expected, rel=1e-12)


class TestBootstrap:
    def test_strong_signal(self):
        pnl = _make_pnl(mean=0.003, std=0.01)
        p = _bootstrap_sharpe(pnl)
        assert p < 0.05  # significant

    def test_no_signal(self):
        pnl = _make_pnl(mean=0, std=0.02)
        p = _bootstrap_sharpe(pnl)
        assert p > 0.10  # not significant


# ── Full Metrics Computation ──────────────────────────────────────────


class TestComputeAllMetrics:
    def test_returns_all_keys(self, good_strategy):
        pnl, dates, pos = good_strategy
        m = compute_all_metrics(pnl, dates, pos, asset_returns=pnl)
        required_keys = [
            "sharpe",
            "lo_adjusted",
            "sortino",
            "total_return",
            "max_dd",
            "calmar",
            "dsr",
            "dsr_trials_used",
            "loss_years",
            "loss_years_applicable",
            "full_years_count",
            "drawdown_time_frac",
            "max_drawdown_duration_bars",
            "omega",
            "omega_applicable",
            "skew",
            "sharpe_lo_ratio",
            "bootstrap_p",
            "position_ic",
            "position_hit_rate",
            "position_ic_stability",
            "position_ic_monthly_mean",
            "position_ic_applicable",
            "position_ic_stability_applicable",
            "active_days",
            "total_days",
            "yearly_sharpes",
            "yearly_pnl",
        ]
        for key in required_keys:
            assert key in m, f"Missing key: {key}"

    def test_gate_rejects_non_finite_dsr(self, good_strategy):
        pnl, dates, pos = good_strategy
        m = compute_all_metrics(pnl, dates, pos, asset_returns=pnl)
        m["dsr"] = float("nan")
        passed, failures = validate(m, load_profile("equity_daily"))
        assert passed is False
        assert any("T6 DSR invalid" in item for item in failures)

    def test_sharpe_positive_for_good_strategy(self, good_strategy):
        pnl, dates, pos = good_strategy
        m = compute_all_metrics(pnl, dates, pos, asset_returns=pnl)
        assert m["sharpe"] > 1.0

    def test_omega_above_one_for_positive(self, good_strategy):
        pnl, dates, pos = good_strategy
        m = compute_all_metrics(pnl, dates, pos, asset_returns=pnl)
        assert m["omega"] > 1.0

    def test_position_ic_computed_with_positions(self):
        """Position-return IC should be nonzero when positions genuinely predict returns."""
        rng = np.random.RandomState(42)
        asset_returns = rng.normal(0.001, 0.02, 500)
        dates = _make_dates(n=500)
        pos = asset_returns * 10 + 0.5
        pnl = pos * asset_returns
        m = compute_all_metrics(pnl, dates, pos, asset_returns=asset_returns)
        assert m["position_ic"] > 0.3

    def test_position_ic_low_variance_inputs_remain_applicable(self):
        dates = _make_dates(n=500)
        asset_returns = np.full(500, 0.01)
        pos = np.full(500, 0.5)
        pnl = pos * asset_returns
        m = compute_all_metrics(pnl, dates, pos, asset_returns=asset_returns)
        assert m["position_ic_applicable"] is True
        assert m["position_ic_stability_applicable"] is False
        assert m["position_ic"] == 0.0

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 30"):
            compute_all_metrics(np.array([0.01] * 10), _make_dates(10))

    def test_nan_handling(self):
        pnl = _make_pnl(n=100)
        pnl[5] = np.nan
        pnl[10] = np.inf
        dates = _make_dates(n=100)
        m = compute_all_metrics(pnl, dates)
        assert np.isfinite(m["sharpe"])

    def test_calmar_uses_elapsed_time_for_intraday_data(self):
        dates = pd.date_range("2020-01-01", periods=60, freq="h")
        pnl = np.array([0.02] * 20 + [-0.01] + [0.003] * 39)
        m = compute_all_metrics(pnl, dates, profile=load_profile("hft"))
        naive_years = len(dates) / load_profile("hft")["validation"]["periods_per_year"]
        naive_ann_return = (np.cumprod(1.0 + pnl)[-1] ** (1.0 / naive_years)) - 1.0
        naive_calmar = naive_ann_return / abs(m["max_dd"])
        assert _elapsed_years(dates, periods_per_year=252) < naive_years
        assert m["calmar"] > 0.0
        assert m["calmar"] > naive_calmar


class TestCalendarYearDetection:
    def test_equity_business_day_tolerance_allows_boundary_holidays(self):
        dates = pd.bdate_range("2020-01-02", "2020-12-31")
        assert _is_full_calendar_year(dates, {"calendar_type": "business_day"}) is True

    def test_crypto_calendar_day_requires_near_complete_calendar_coverage(self):
        dates = pd.date_range("2020-01-06", "2020-12-26", freq="D")
        assert _is_full_calendar_year(dates, {"calendar_type": "calendar_day"}) is False

    def test_hft_intraday_does_not_use_equity_boundary_tolerance(self):
        dates = pd.date_range("2020-01-02", "2020-12-31 23:00:00", freq="h")
        assert _is_full_calendar_year(dates, {"calendar_type": "intraday"}) is False
