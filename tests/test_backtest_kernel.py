"""Tests for the minimal backtest kernel."""

import numpy as np
import pytest

from abel_edge.engine.backtest import BacktestSettings, run_backtest


def test_run_backtest_matches_legacy_behavior_by_default():
    positions = np.array([0.0, 1.0, 1.0])
    prices = np.array([100.0, 110.0, 121.0])

    result = run_backtest(positions, prices)

    assert result["positions"] == pytest.approx([0.0, 1.0, 1.0])
    assert result["asset_returns"] == pytest.approx([0.0, 0.1, 0.1])
    assert result["gross_pnl"] == pytest.approx([0.0, 0.1, 0.1])
    assert result["execution_cost"] == pytest.approx([0.0, 0.0, 0.0])
    assert result["pnl"] == pytest.approx([0.0, 0.1, 0.1])


def test_run_backtest_deducts_turnover_cost():
    positions = np.array([0.0, 1.0, 0.25])
    prices = np.array([100.0, 110.0, 121.0])

    result = run_backtest(positions, prices, settings=BacktestSettings(cost_bps=50))

    assert result["turnover"] == pytest.approx([0.0, 1.0, 0.75])
    assert result["execution_cost"] == pytest.approx([0.0, 0.005, 0.00375])
    assert result["gross_pnl"] == pytest.approx([0.0, 0.1, 0.025])
    assert result["pnl"] == pytest.approx([0.0, 0.095, 0.02125])


def test_run_backtest_clips_positions_before_turnover_and_pnl():
    positions = np.array([0.0, 2.0, -2.0])
    prices = np.array([100.0, 110.0, 99.0])

    result = run_backtest(
        positions,
        prices,
        settings=BacktestSettings(cost_bps=100, max_abs_position=0.5),
    )

    assert result["positions"] == pytest.approx([0.0, 0.5, -0.5])
    assert result["turnover"] == pytest.approx([0.0, 0.5, 1.0])
    assert result["execution_cost"] == pytest.approx([0.0, 0.005, 0.01])
    assert result["gross_pnl"] == pytest.approx([0.0, 0.05, 0.05])
    assert result["pnl"] == pytest.approx([0.0, 0.045, 0.04])


def test_run_backtest_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        run_backtest(np.array([0.0, 1.0]), np.array([100.0]))


def test_run_backtest_compiles_next_position_intent_when_requested():
    next_position = np.array([1.0, 0.5, -0.5])
    prices = np.array([100.0, 110.0, 99.0])

    result = run_backtest(
        next_position,
        prices,
        input_semantics="next_position",
        execution_delay_bars=1,
    )

    assert result["next_position"] == pytest.approx([1.0, 0.5, -0.5])
    assert result["positions"] == pytest.approx([0.0, 1.0, 0.5])
    assert result["gross_pnl"] == pytest.approx([0.0, 0.1, -0.05])
