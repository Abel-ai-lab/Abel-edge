from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from abel_edge.dashboard.components import compute_metrics as compute_dashboard_metrics
from abel_edge.dashboard.components import drawdown_chart
from abel_edge.validation import explain_metric_gates
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


def test_dashboard_max_drawdown_counts_first_row_loss():
    metrics = compute_dashboard_metrics(np.array([-0.20, 0.0]))
    assert metrics["max_dd"] == pytest.approx(0.20)


def test_dashboard_drawdown_chart_counts_first_row_loss():
    dates = pd.bdate_range("2020-01-01", periods=2)
    payload = json.loads(drawdown_chart(dates, np.array([-0.20, -0.20]), "first loss"))
    assert payload["data"][0]["y"][0] == pytest.approx(-20.0)


def test_validation_total_return_uses_simple_interest():
    dates = pd.bdate_range("2020-01-01", periods=30)
    pnl = np.array([0.10, 0.10] + [0.0] * 28)
    metrics = compute_all_metrics(pnl, dates, profile=load_profile("equity_daily"))
    assert metrics["total_return"] == pytest.approx(0.20)


def test_validation_max_drawdown_counts_first_row_loss():
    dates = pd.bdate_range("2020-01-01", periods=30)
    pnl = np.array([-0.20] + [0.0] * 29)
    metrics = compute_all_metrics(pnl, dates, profile=load_profile("equity_daily"))
    assert metrics["max_dd"] == pytest.approx(-0.20)


def test_daily_return_floor_uses_annualized_simple_return():
    passed, failures = validate(_passing_metrics(), load_profile("equity_daily"))
    assert passed is True
    assert failures == []

    passed, failures = validate(
        _passing_metrics(annual_return=0.049), load_profile("equity_daily")
    )
    assert passed is False
    assert "Annualized return floor +4.90% < +5.00%" in failures


def test_explain_metric_gates_marks_old_annual_return_fallback():
    metrics = _passing_metrics(total_return=0.04)
    metrics.pop("annual_return")

    explanation = explain_metric_gates(metrics, load_profile("equity_daily"))
    return_floor = next(
        check for check in explanation["checks"] if check["id"] == "return_floor"
    )

    assert explanation["passed"] is False
    assert explanation["score"] == "4/5"
    assert return_floor["metric"] == "annual_return"
    assert return_floor["observed_metric"] == "total_return"
    assert return_floor["observed"] == pytest.approx(0.04)
    assert return_floor["threshold"] == pytest.approx(0.05)
    assert return_floor["compatibility_fallback"] == "annual_return_missing_total_return"
    assert return_floor["passed"] is False
    assert return_floor["message"] == "Annualized return floor +4.00% < +5.00%"
    assert explanation["failures"] == ["Annualized return floor +4.00% < +5.00%"]


def test_explain_metric_gates_allows_old_annual_return_fallback_to_pass():
    metrics = _passing_metrics(total_return=0.08)
    metrics.pop("annual_return")

    explanation = explain_metric_gates(metrics, load_profile("equity_daily"))
    return_floor = next(
        check for check in explanation["checks"] if check["id"] == "return_floor"
    )

    assert explanation["passed"] is True
    assert return_floor["passed"] is True
    assert return_floor["observed_metric"] == "total_return"
    assert return_floor["compatibility_fallback"] == "annual_return_missing_total_return"
    assert return_floor["message"] == "Annualized return floor +8.00% >= +5.00%"


def test_explain_metric_gates_uses_new_annual_return_without_fallback():
    explanation = explain_metric_gates(
        _passing_metrics(total_return=0.40, annual_return=0.049),
        load_profile("equity_daily"),
    )
    return_floor = next(
        check for check in explanation["checks"] if check["id"] == "return_floor"
    )

    assert explanation["passed"] is False
    assert return_floor["observed_metric"] == "annual_return"
    assert "compatibility_fallback" not in return_floor
    assert return_floor["observed"] == pytest.approx(0.049)
    assert return_floor["message"] == "Annualized return floor +4.90% < +5.00%"


def test_hft_return_floor_stays_total_return_based():
    passed, failures = validate(
        _passing_metrics(total_return=0.04, annual_return=0.50), load_profile("hft")
    )
    assert passed is False
    assert "Return floor +4.00% < +5.00%" in failures
