from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel_edge.dashboard.components import compute_metrics as compute_dashboard_metrics
from abel_edge.validation.gate_logic import validate
from abel_edge.validation.metrics import compute_all_metrics, load_profile


def _passing_metrics(**overrides):
    metrics = {
        "dsr": 0.99,
        "loss_years_applicable": False,
        "loss_years": 0,
        "lo_adjusted": 1.0,
        "omega_applicable": False,
        "omega": 0.0,
        "max_dd": -0.01,
        "total_return": 0.10,
        "annual_return": 0.051,
        "sharpe": 1.0,
        "sharpe_lo_ratio": 1.0,
        "position_ic_applicable": False,
        "position_ic_stability_applicable": False,
    }
    metrics.update(overrides)
    return metrics


def test_dashboard_cumulative_return_uses_simple_interest():
    metrics = compute_dashboard_metrics(np.array([0.10, 0.10]))
    assert metrics["cum_return"] == pytest.approx(0.20)


def test_validation_total_return_uses_simple_interest():
    dates = pd.bdate_range("2020-01-01", periods=30)
    pnl = np.array([0.10, 0.10] + [0.0] * 28)
    metrics = compute_all_metrics(pnl, dates, profile=load_profile("equity_daily"))
    assert metrics["total_return"] == pytest.approx(0.20)


def test_daily_return_floor_uses_annualized_simple_return():
    passed, failures = validate(_passing_metrics(), load_profile("equity_daily"))
    assert passed is True
    assert failures == []

    passed, failures = validate(
        _passing_metrics(annual_return=0.049), load_profile("equity_daily")
    )
    assert passed is False
    assert "Annualized return floor +4.90% < +5.00%" in failures


def test_hft_return_floor_stays_total_return_based():
    passed, failures = validate(
        _passing_metrics(total_return=0.04, annual_return=0.50), load_profile("hft")
    )
    assert passed is False
    assert "Return floor +4.00% < +5.00%" in failures
