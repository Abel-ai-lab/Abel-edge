"""Tests for metric triangle invariants, anti-gaming, and validation gate.

These tests LOCK the core anti-gaming property of the metric triangle:
  - Leverage invariance (Lo, IC, Omega don't change with scale)
  - Clipping detection (Omega catches return clipping)
  - Serial correlation detection (Lo catches autocorrelated signals)
  - KEEP/DISCARD decision logic
  - Profile loading and validation gate
  - Integration: validate_strategy on CSV trade logs
"""

import numpy as np
import pandas as pd
import pytest

from causal_edge.validation.metrics import (
    _sharpe,
    compute_all_metrics,
    detect_profile,
    load_profile,
    validate,
    decide_keep_discard,
)
from causal_edge.validation.gate import validate_strategy


def _make_pnl(n=500, mean=0.001, std=0.02, seed=42):
    return np.random.RandomState(seed).normal(mean, std, n)


def _make_dates(n=500, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def _make_positions(pnl, lag=1):
    pos = np.zeros_like(pnl)
    pos[lag:] = np.sign(pnl[:-lag]) * 0.5 + 0.5
    return pos


@pytest.fixture
def good_strategy():
    pnl = _make_pnl(n=750, mean=0.0015, std=0.015, seed=42)
    dates = _make_dates(n=750)
    pos = _make_positions(pnl, lag=1)
    return pnl, dates, pos


@pytest.fixture
def bad_strategy():
    pnl = np.random.RandomState(99).normal(0, 0.02, 500)
    dates = _make_dates(n=500)
    pos = np.ones(500) * 0.5
    return pnl, dates, pos


# ═══════════════════════════════════════════════════════════════════
# TRIANGLE INVARIANT TESTS
# ═══════════════════════════════════════════════════════════════════


class TestLeverageInvariance:
    """Metric triangle must be leverage-invariant.
    Scaling PnL by 2x must NOT change Lo, IC, or Omega.
    MaxDD DOES scale (that's why it's not in the triangle).
    """

    def test_sharpe_invariant(self):
        pnl = _make_pnl(n=500, mean=0.001)
        assert abs(_sharpe(pnl) - _sharpe(pnl * 2)) < 0.01

    def test_lo_invariant(self):
        pnl = _make_pnl(n=500, mean=0.001)
        dates = _make_dates(n=500)
        m1 = compute_all_metrics(pnl, dates)
        m2 = compute_all_metrics(pnl * 2, dates)
        assert abs(m1["lo_adjusted"] - m2["lo_adjusted"]) < 0.05

    def test_ic_invariant(self):
        pnl = _make_pnl(n=500, mean=0.001)
        dates = _make_dates(n=500)
        pos = _make_positions(pnl)
        m1 = compute_all_metrics(pnl, dates, pos, asset_returns=pnl)
        m2 = compute_all_metrics(pnl * 2, dates, pos * 2, asset_returns=pnl * 2)
        assert abs(m1["position_ic"] - m2["position_ic"]) < 0.01

    def test_omega_invariant(self):
        pnl = _make_pnl(n=500, mean=0.001)
        dates = _make_dates(n=500)
        m1 = compute_all_metrics(pnl, dates)
        m2 = compute_all_metrics(pnl * 2, dates)
        assert abs(m1["omega"] - m2["omega"]) < 0.05

    def test_maxdd_scales_with_leverage(self):
        pnl = _make_pnl(n=500, mean=0.001)
        dates = _make_dates(n=500)
        m1 = compute_all_metrics(pnl, dates)
        m2 = compute_all_metrics(pnl * 2, dates)
        ratio = m2["max_dd"] / m1["max_dd"] if m1["max_dd"] != 0 else 1
        assert ratio > 1.5  # MaxDD roughly doubles (NOT invariant)


class TestClippingDetection:
    """Omega catches return clipping — the shape dimension of the triangle."""

    def test_clipping_degrades_omega(self):
        rng = np.random.RandomState(42)
        pnl = rng.normal(0.001, 0.03, 2000)
        dates = _make_dates(n=2000)
        m_raw = compute_all_metrics(pnl, dates)
        m_clip = compute_all_metrics(np.clip(pnl, -0.015, 0.015), dates)
        assert m_clip["omega"] < m_raw["omega"]

    def test_triangle_catches_clipping_via_keep_discard(self):
        rng = np.random.RandomState(42)
        pnl_raw = rng.normal(0.001, 0.03, 1000)
        pnl_clip = np.clip(pnl_raw, -0.015, 0.015)
        dates = _make_dates(n=1000)
        m_raw = compute_all_metrics(pnl_raw, dates)
        m_clip = compute_all_metrics(pnl_clip, dates)
        p = load_profile("equity_daily")
        if m_clip["omega"] < m_raw["omega"] - 0.10:
            assert decide_keep_discard(m_clip, m_raw, p) == "DISCARD"


class TestSerialCorrelationDetection:
    """Lo catches autocorrelated signals — the ratio dimension."""

    def test_autocorrelated_has_high_ratio(self):
        rng = np.random.RandomState(42)
        raw = rng.normal(0.001, 0.02, 100)
        pnl = np.repeat(raw, 5)
        dates = _make_dates(n=len(pnl))
        m = compute_all_metrics(pnl, dates)
        assert m["sharpe_lo_ratio"] > 1.2


# ═══════════════════════════════════════════════════════════════════
# Profile & Gate Tests
# ═══════════════════════════════════════════════════════════════════


class TestProfileLoading:
    def test_crypto_daily(self):
        p = load_profile("crypto_daily")
        assert p["name"] == "crypto_daily"
        assert "metric_triangle" in p

    def test_equity_daily(self):
        assert load_profile("equity_daily")["name"] == "equity_daily"

    def test_hft(self):
        assert load_profile("hft")["name"] == "hft"

    def test_missing(self):
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent")


class TestDetectProfile:
    def test_high_vol_crypto(self):
        assert detect_profile(_make_pnl(std=0.05), _make_dates()) == "crypto_daily"

    def test_low_vol_equity(self):
        assert detect_profile(_make_pnl(std=0.01), _make_dates()) == "equity_daily"


class TestValidationGate:
    def test_good_passes_most(self, good_strategy):
        m = compute_all_metrics(*good_strategy)
        _, failures = validate(m, load_profile("equity_daily"))
        assert len(failures) <= 5

    def test_bad_fails(self, bad_strategy):
        m = compute_all_metrics(*bad_strategy)
        passed, failures = validate(m, load_profile("equity_daily"))
        assert not passed and len(failures) >= 1


class TestKeepDiscard:
    def test_improvement_keeps(self):
        b = {"lo_adjusted": 2.0, "position_ic": 0.10, "omega": 2.0, "max_dd": -0.10}
        c = {"lo_adjusted": 2.5, "position_ic": 0.12, "omega": 2.1, "max_dd": -0.08}
        assert decide_keep_discard(c, b, load_profile("crypto_daily")) == "KEEP"

    def test_regression_discards(self):
        b = {"lo_adjusted": 2.0, "position_ic": 0.10, "omega": 2.0, "max_dd": -0.10}
        c = {"lo_adjusted": 1.8, "position_ic": 0.12, "omega": 2.1, "max_dd": -0.08}
        assert decide_keep_discard(c, b, load_profile("crypto_daily")) == "DISCARD"

    def test_omega_guardrail(self):
        b = {"lo_adjusted": 2.0, "position_ic": 0.10, "omega": 2.0, "max_dd": -0.10}
        c = {"lo_adjusted": 2.5, "position_ic": 0.10, "omega": 1.8, "max_dd": -0.08}
        assert decide_keep_discard(c, b, load_profile("crypto_daily")) == "DISCARD"

    def test_ic_guardrail(self):
        b = {"lo_adjusted": 2.0, "position_ic": 0.10, "omega": 2.0, "max_dd": -0.10}
        c = {"lo_adjusted": 2.5, "position_ic": 0.08, "omega": 2.1, "max_dd": -0.08}
        assert decide_keep_discard(c, b, load_profile("crypto_daily")) == "DISCARD"

    def test_maxdd_gate_absolute(self):
        b = {"lo_adjusted": 2.0, "position_ic": 0.10, "omega": 2.0, "max_dd": -0.10}
        c = {"lo_adjusted": 3.0, "position_ic": 0.15, "omega": 2.5, "max_dd": -0.30}
        assert decide_keep_discard(c, b, load_profile("crypto_daily")) == "DISCARD"

    def test_maxdd_uses_validation_policy_key(self):
        profile = {
            "metric_triangle": {"optimize": "sharpe", "guardrails": []},
            "anti_gaming": {},
            "validation": {"max_dd": -0.20},
        }
        b = {"sharpe": 1.0, "max_dd": -0.10}
        c = {"sharpe": 2.0, "max_dd": -0.25}
        assert decide_keep_discard(c, b, profile) == "DISCARD"


# ═══════════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════════


class TestValidateStrategyIntegration:
    def test_with_csv(self, tmp_path):
        pnl = _make_pnl(n=300, mean=0.001, std=0.015)
        df = pd.DataFrame(
            {
                "date": _make_dates(n=300),
                "asset_return": pnl,
                "pnl": pnl,
                "position": _make_positions(pnl),
                "cum_return": np.cumprod(1.0 + pnl) - 1.0,
                "source": "backfill",
            }
        )
        csv_path = tmp_path / "trade_log_test.csv"
        df.to_csv(csv_path, index=False)
        result = validate_strategy(csv_path, profile="equity_daily")
        assert result["verdict"] in ("PASS", "FAIL")
        assert result["triangle"]["ratio"] != 0

    def test_too_short_fails(self, tmp_path):
        df = pd.DataFrame({"date": _make_dates(10), "pnl": [0.01] * 10})
        csv_path = tmp_path / "short.csv"
        df.to_csv(csv_path, index=False)
        result = validate_strategy(csv_path)
        assert result["verdict"] == "FAIL"
