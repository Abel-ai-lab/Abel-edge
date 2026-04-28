"""Tests for static and runtime look-ahead detection."""

import numpy as np

from abel_edge.validation.look_ahead import check_runtime, check_static


def test_t2_flags_rolling_without_shift() -> None:
    code = "x = ret.rolling(20).mean()\npositions = x > 0"
    assert any("T2" in violation for violation in check_static(code))


def test_t2_allows_rolling_with_shift() -> None:
    code = "x = ret.rolling(20).mean().shift(1)\npositions = x > 0"
    assert not any("T2" in violation for violation in check_static(code))


def test_t3_flags_global_numpy_reduction() -> None:
    code = "def f(pnl):\n    return np.mean(pnl) / np.std(pnl)"
    assert any("T3" in violation for violation in check_static(code))


def test_t3_respects_slice_and_noqa() -> None:
    code = "def f(pnl, i):\n    return np.std(pnl[:i])  # noqa: T3"
    assert check_static(code) == []


def test_t4_flags_walk_forward_slice() -> None:
    code = "X_train = X[:i+1]\nmodel.fit(X_train, y_train)"
    assert any("T4" in violation for violation in check_static(code))


def test_t5_flags_current_day_trend_filter() -> None:
    code = "if close[i] < sma[i]: positions[i] = 0"
    assert any("T5" in violation for violation in check_static(code))


def test_runtime_r1_flags_leaky_positions() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.02, 200)
    positions = np.abs(returns) * 50
    pnl = positions * returns
    messages = check_runtime(pnl, positions, returns)
    assert any(message.startswith("R1") for message in messages)


def test_runtime_r2_flags_suspicious_hit_rate() -> None:
    pnl = np.abs(np.random.default_rng(42).normal(0, 0.02, 200))
    positions = np.ones(200) * 0.5
    messages = check_runtime(pnl, positions)
    assert any(message.startswith("R2") for message in messages)


def test_runtime_r2_skips_constant_return_series() -> None:
    positions = np.ones(64) * 0.5
    returns = np.ones(64) * 0.02
    pnl = positions * returns
    assert not any(message.startswith("R2") for message in check_runtime(pnl, positions, returns))
