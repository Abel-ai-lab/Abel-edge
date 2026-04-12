"""Generate deterministic CSV fixtures for validation tests.

Run once:  .venv/bin/python tests/generate_fixtures.py
All CSVs land in tests/fixtures/validation/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures" / "validation"
FIXTURES.mkdir(parents=True, exist_ok=True)


def _bdays(start: str, periods: int) -> pd.DatetimeIndex:
    """Generate business-day DatetimeIndex."""
    return pd.bdate_range(start=start, periods=periods)


def generate_positive_daily() -> None:
    """~210 business days starting 2020-01-02, spanning ~10 months (no full calendar year).

    All-positive pnl, with position and asset_return columns.
    Detected as equity_daily (ann_vol <= 0.60).
    """
    rng = np.random.RandomState(42)
    # ~110 bdays ≈ 5 months → position_ic_stability_applicable=False (<6 months)
    # Starts 2020-01-02, ends ~June → no full calendar year → loss_years_applicable=False
    n = 110
    dates = _bdays("2020-01-02", n)

    # Small positive pnl each day: mean ~0.003, std ~0.0002 → always positive
    # With n=110: total_return ≈ prod(1+0.003)^110 - 1 ≈ 0.39 → passes return_floor 0.30
    pnl = 0.003 + rng.randn(n) * 0.0002
    pnl = np.abs(pnl)  # ensure strictly positive

    positions = 0.5 + rng.randn(n) * 0.05
    # Realistic equity asset returns (mean ~0.001, std ~0.01) → ann_vol ~15.9%
    asset_return = rng.randn(n) * 0.01 + 0.001

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "pnl": np.round(pnl, 8),
        "position": np.round(positions, 6),
        "asset_return": np.round(asset_return, 8),
    })
    df.to_csv(FIXTURES / "positive_daily.csv", index=False)
    print(f"positive_daily.csv: {len(df)} rows, last date {dates[-1].date()}")


def generate_positive_clipped() -> None:
    """Same shape as positive_daily but pnl = 0 everywhere → sharpe=0, skew=0."""
    rng = np.random.RandomState(42)
    n = 110
    dates = _bdays("2020-01-02", n)

    pnl = np.zeros(n)
    positions = 0.5 + rng.randn(n) * 0.05
    asset_return = rng.randn(n) * 0.01 + 0.001

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "pnl": pnl,
        "position": np.round(positions, 6),
        "asset_return": np.round(asset_return, 8),
    })
    df.to_csv(FIXTURES / "positive_clipped.csv", index=False)
    print(f"positive_clipped.csv: {len(df)} rows")


def generate_autocorrelated() -> None:
    """Daily data with high serial correlation → sharpe_lo_ratio > 1.0.

    An AR(1) process with high phi creates positive autocorrelation which
    inflates raw Sharpe relative to Lo-adjusted Sharpe → ratio > 1.
    """
    rng = np.random.RandomState(42)
    n = 252
    dates = _bdays("2020-01-02", n)

    # AR(1) with phi=0.7 → strong positive autocorrelation
    phi = 0.7
    noise = rng.randn(n) * 0.005
    pnl = np.zeros(n)
    pnl[0] = 0.003 + noise[0]
    for i in range(1, n):
        pnl[i] = 0.001 + phi * pnl[i - 1] + noise[i]

    positions = 0.5 + rng.randn(n) * 0.05
    asset_return = rng.randn(n) * 0.01 + 0.001

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "pnl": np.round(pnl, 8),
        "position": np.round(positions, 6),
        "asset_return": np.round(asset_return, 8),
    })
    df.to_csv(FIXTURES / "autocorrelated.csv", index=False)
    print(f"autocorrelated.csv: {len(df)} rows")


def generate_ic_supported() -> None:
    """Positions nearly perfectly predict returns → IC > 0.95, hit_rate ~1.0.

    < 6 months of data → position_ic_stability_applicable is False.
    Must have some losses → omega_applicable=True → total tests = 9.
    """
    rng = np.random.RandomState(42)
    n = 100  # ~5 months → < 6 months
    dates = _bdays("2020-01-02", n)

    # Positions with strong signal
    positions = rng.randn(n) * 0.3
    # asset_return = position * scale + small noise (still very correlated)
    asset_return = positions * 0.05 + rng.randn(n) * 0.001

    # pnl: mostly position * asset_return but add independent noise so some are negative
    # This creates a mix of gains and losses while maintaining high IC
    base_pnl = positions * asset_return
    pnl = base_pnl + rng.randn(n) * 0.0005  # small additive noise

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "pnl": np.round(pnl, 8),
        "position": np.round(positions, 6),
        "asset_return": np.round(asset_return, 8),
    })
    df.to_csv(FIXTURES / "ic_supported.csv", index=False)
    print(f"ic_supported.csv: {len(df)} rows")


def generate_ic_unsupported_no_position() -> None:
    """No 'position' column. total_return ~0.041 so export shows '+4.1%'.

    Tests expect score "6/7" with equity_daily profile.
    """
    rng = np.random.RandomState(42)
    n = 210
    dates = _bdays("2020-01-02", n)

    # We need total_return ~0.041.
    # total_return = prod(1 + pnl_i) - 1
    # For small daily returns: total_return ≈ sum(pnl_i)
    # Target sum ≈ 0.041, so mean pnl ≈ 0.041/210 ≈ 0.000195
    # Need mix of positive and negative (omega must not be applicable → wait, test says omega_applicable is False)
    # Actually test says omega_applicable is False. Let's check: omega_applicable = len(losses) > 0 and loss_mass > 1e-12
    # If omega_applicable is False, it means either no losses or loss_mass is negligible.
    # But test also says verdict=FAIL, score="6/7". Let's check what fails:
    # With equity_daily: return_floor = 0.30, total_return = 0.041 < 0.30 → 1 failure
    # score = (total - failures) / total, "6/7" means total=7, failures=1
    # _count_total: base 6 + sharpe_lo_ratio = 7 (no loss_years, no omega, no IC, no IC_stability)
    # So we need: loss_years_applicable=False, omega_applicable=False, position_ic_applicable=False
    # omega_applicable=False means no negative pnl (all positive or zero)
    # But then only 1 failure (return_floor), giving 6/7. Perfect.
    # All-positive pnl with total_return ~0.041

    # target: prod(1+pnl_i) = 1.041
    # Use small positive daily returns
    target_total = 1.041
    # daily_return such that target_total^(1/n) - 1
    mean_daily = target_total ** (1.0 / n) - 1  # ~0.000191
    pnl = mean_daily + rng.randn(n) * 0.00005
    pnl = np.abs(pnl)  # ensure all positive → omega_applicable=False

    # Adjust to hit exact total_return
    # current total_return = prod(1+pnl) - 1
    current = np.prod(1.0 + pnl) - 1.0
    # Scale pnl to hit target. For small values, scaling linearly is close enough.
    # More precisely: we want prod(1 + pnl_i * s) = 1.041
    # For small pnl, log(1+x) ≈ x, so sum(pnl*s) ≈ 0.041, s = 0.041/sum(pnl)
    scale = np.log(1.041) / np.sum(np.log(1.0 + pnl))
    # Apply scaling: pnl_new = (1+pnl)^scale - 1
    pnl_scaled = (1.0 + pnl) ** scale - 1.0

    # Asset returns for detect_profile (equity_daily requires ann_vol <= 0.60)
    asset_return = rng.randn(n) * 0.01 + 0.001

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "pnl": np.round(pnl_scaled, 8),
        "asset_return": np.round(asset_return, 8),
    })
    df.to_csv(FIXTURES / "ic_unsupported_no_position.csv", index=False)

    actual_tr = np.prod(1.0 + pnl_scaled) - 1.0
    print(f"ic_unsupported_no_position.csv: {len(df)} rows, total_return={actual_tr:.6f}")


def generate_insufficient_rows() -> None:
    """Only 20 rows → triggers 'need 30+' failure."""
    rng = np.random.RandomState(42)
    n = 20
    dates = _bdays("2020-01-02", n)
    pnl = rng.randn(n) * 0.01

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "pnl": np.round(pnl, 8),
    })
    df.to_csv(FIXTURES / "insufficient_rows.csv", index=False)
    print(f"insufficient_rows.csv: {len(df)} rows")


if __name__ == "__main__":
    generate_positive_daily()
    generate_positive_clipped()
    generate_autocorrelated()
    generate_ic_supported()
    generate_ic_unsupported_no_position()
    generate_insufficient_rows()
    print("\nAll fixtures generated.")
