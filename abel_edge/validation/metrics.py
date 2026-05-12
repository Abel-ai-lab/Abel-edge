"""Abel Proof — Core metrics engine.

Computes all strategy metrics from a PnL array. Used by gate.py for
strategy admission validation. Zero external dependencies beyond
pandas, numpy, scipy.

Metric triangle (leverage-invariant, orthogonal):
  - Ratio: Lo-adjusted Sharpe (crypto) or raw Sharpe (equity)
  - Rank:  Position-Return IC (Spearman rank correlation of position vs asset return)
  - Shape: Omega (sum of gains / sum of |losses|)

Anti-gaming:
  - Clipping inflates Sharpe but tanks Omega (catches return clipping)
  - Serial correlation inflates Sharpe but not Lo (catches autocorr gaming)
  - MaxDD is absolute gate only (scales with leverage, not in triangle)
"""

import os
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from abel_edge.validation.gate_logic import (
    decide_keep_discard as decide_keep_discard,
    validate as validate,
)
from abel_edge.validation.position_ic import compute_position_ic

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "profiles")


def _lag1_autocorr(series: np.ndarray) -> float:
    """Return lag-1 autocorrelation without emitting warnings on degenerate inputs."""
    values = np.asarray(series, dtype=float)
    if values.size < 2:
        return 0.0
    current = values[1:]
    previous = values[:-1]
    if np.std(current, ddof=1) <= 1e-12 or np.std(previous, ddof=1) <= 1e-12:
        return 0.0
    rho1 = pd.Series(values).autocorr(lag=1)
    if np.isnan(rho1):
        return 0.0
    return float(rho1)


# ═══════════════════════════════════════════════════════════════════
# Profile Loading
# ═══════════════════════════════════════════════════════════════════
def load_profile(name_or_path: str) -> dict:
    """Load a metric profile by name ('crypto_daily') or file path."""
    if os.path.exists(name_or_path):
        path = name_or_path
    else:
        path = os.path.join(PROFILES_DIR, f"{name_or_path}.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Profile not found: {name_or_path} (searched {path})")
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_profile(
    pnl: np.ndarray, dates: pd.DatetimeIndex, asset_returns: np.ndarray = None
) -> str:
    """Auto-detect profile from data characteristics."""
    if len(dates) > 1:
        gaps = pd.Series(dates).diff().dropna()
        median_gap = gaps.median()
        if median_gap < pd.Timedelta(hours=1):
            return "hft"
    series = asset_returns if asset_returns is not None and len(asset_returns) == len(pnl) else pnl
    ann_vol = np.std(series, ddof=1) * np.sqrt(252)
    if ann_vol > 0.60:
        return "crypto_daily"
    return "equity_daily"


# ═══════════════════════════════════════════════════════════════════
# Metrics Computation
# ═══════════════════════════════════════════════════════════════════


def compute_all_metrics(
    pnl: np.ndarray,
    dates: pd.DatetimeIndex,
    positions: np.ndarray = None,
    profile: dict | None = None,
    dsr_trials: int | None = None,
    asset_returns: np.ndarray = None,
) -> dict:
    """Compute all strategy metrics from a PnL array.

    Args:
        pnl: daily simple-return strategy PnL array
        dates: DatetimeIndex aligned with pnl
        positions: optional position array for position-return IC computation
        asset_returns: optional underlying asset simple-return array aligned with pnl

    Returns dict with all metrics needed for validation gate.
    """
    T = len(pnl)
    if T < 30:
        raise ValueError(f"Need at least 30 days, got {T}")

    pnl = np.nan_to_num(pnl, nan=0.0, posinf=0.0, neginf=0.0)
    if positions is not None:
        positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)

    equity = np.cumprod(1.0 + pnl)
    cum_return = equity - 1.0
    peak_equity = np.maximum.accumulate(equity)
    dd = (equity / peak_equity) - 1.0
    std = np.std(pnl, ddof=1)
    validation_cfg = (profile or {}).get("validation", {})
    periods_per_year = validation_cfg.get("periods_per_year", 252)

    sharpe = float(np.mean(pnl) / std * np.sqrt(periods_per_year)) if std > 1e-10 else 0
    sortino = _sortino(pnl, periods_per_year=periods_per_year)
    max_dd = float(np.min(dd))
    total_return = float(cum_return[-1])
    elapsed_years = _elapsed_years(dates, periods_per_year=periods_per_year)
    ann_return = (
        (equity[-1] ** (1.0 / elapsed_years) - 1.0)
        if elapsed_years > 0 and equity[-1] > 0
        else 0.0
    )
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    # Simplified serial-correlation penalty: lag-1 autocorrelation only.
    rho1 = _lag1_autocorr(pnl)
    cf = 1 + 2 * rho1 * (1 - 1 / periods_per_year)
    lo_adjusted = sharpe * np.sqrt(1 / cf) if cf > 0 else sharpe

    dsr_trials_used = _positive_dsr_trials(
        dsr_trials if dsr_trials is not None else validation_cfg.get("dsr_K", 300)
    )
    dsr = _dsr(pnl, T, K=dsr_trials_used, periods_per_year=periods_per_year)

    # Year-by-year stability: count only full calendar years with negative total PnL.
    loss_years = 0
    yearly_sharpes = {}
    yearly_pnl = {}
    full_years_count = 0
    years = sorted(dates.year.unique())
    for yr in years:
        mask = dates.year == yr
        year_dates = dates[mask]
        year_pnl = pnl[mask]
        yearly_sharpes[yr] = _sharpe(year_pnl, periods_per_year=periods_per_year)
        total_year_pnl = float(np.cumprod(1.0 + year_pnl)[-1] - 1.0)
        yearly_pnl[yr] = total_year_pnl
        if _is_full_calendar_year(year_dates, validation_cfg):
            full_years_count += 1
            if total_year_pnl < -1e-12:
                loss_years += 1
    loss_years_applicable = full_years_count > 0

    # Drawdown-time stability: fraction of bars underwater and longest underwater spell.
    underwater = equity < (peak_equity - 1e-12)
    drawdown_time_frac = float(np.mean(underwater)) if T > 0 else 0.0
    max_drawdown_duration_bars = _max_true_run(underwater)

    # Omega (gain/loss asymmetry — catches clipping)
    active = pnl[np.abs(pnl) > 1e-10]
    gains = active[active > 0]
    losses = active[active < 0]
    loss_mass = float(abs(np.sum(losses)))
    omega_applicable = len(losses) > 0 and loss_mass > 1e-12
    omega = float(np.sum(gains) / loss_mass) if omega_applicable else 0.0

    # Tail risk
    skew = float(sp_stats.skew(pnl)) if np.std(pnl) > 1e-10 else 0.0

    sharpe_lo_ratio = sharpe / lo_adjusted if lo_adjusted > 0 else 999
    bootstrap_p = _bootstrap_sharpe(pnl, n_boot=validation_cfg.get("permutation_trials", 1000))

    # Position-Return IC (time-series Spearman rank correlation).
    position_ic, position_hit_rate = 0.0, 0.0
    position_ic_stability, position_ic_monthly_mean = 0.0, 0.0
    position_ic_applicable = False
    position_ic_stability_applicable = False
    if (
        positions is not None
        and asset_returns is not None
        and len(positions) == T
        and len(asset_returns) == T
    ):
        (
            position_ic,
            position_hit_rate,
            position_ic_stability,
            position_ic_monthly_mean,
            position_ic_applicable,
            position_ic_stability_applicable,
        ) = compute_position_ic(asset_returns, positions, dates)

    active_days = (
        int(np.sum(np.abs(positions) > 0.01))
        if positions is not None
        else int(np.sum(np.abs(pnl) > 1e-10))
    )

    return {
        "sharpe": sharpe,
        "lo_adjusted": lo_adjusted,
        "sortino": sortino,
        "total_return": total_return,
        "max_dd": max_dd,
        "calmar": calmar,
        "dsr": dsr,
        "dsr_trials_used": int(dsr_trials_used),
        "loss_years": loss_years,
        "loss_years_applicable": loss_years_applicable,
        "full_years_count": full_years_count,
        "drawdown_time_frac": drawdown_time_frac,
        "max_drawdown_duration_bars": max_drawdown_duration_bars,
        "omega": omega,
        "omega_applicable": omega_applicable,
        "skew": skew,
        "sharpe_lo_ratio": sharpe_lo_ratio,
        "bootstrap_p": bootstrap_p,
        "position_ic": position_ic,
        "position_hit_rate": position_hit_rate,
        "position_ic_stability": position_ic_stability,
        "position_ic_monthly_mean": position_ic_monthly_mean,
        "position_ic_applicable": position_ic_applicable,
        "position_ic_stability_applicable": position_ic_stability_applicable,
        "active_days": active_days,
        "total_days": T,
        "yearly_sharpes": yearly_sharpes,
        "yearly_pnl": yearly_pnl,
    }


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════


def _sharpe(pnl, periods_per_year=252):
    s = np.std(pnl, ddof=1)
    return float(np.mean(pnl) / s * np.sqrt(periods_per_year)) if s > 1e-10 else 0


def _sortino(pnl, periods_per_year=252):
    """Sortino ratio using downside deviation (all observations, MAR=0)."""
    downside = np.minimum(pnl, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    return float(np.mean(pnl) / dd * np.sqrt(periods_per_year)) if dd > 1e-10 else 0.0


def _max_true_run(mask) -> int:
    max_run = 0
    run = 0
    for flag in mask:
        if flag:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return int(max_run)


def _elapsed_years(dates: pd.DatetimeIndex, periods_per_year: int = 252) -> float:
    if len(dates) == 0:
        return 0.0
    if len(dates) == 1:
        return 1.0 / periods_per_year
    span = dates.max() - dates.min()
    median_gap = pd.Series(dates).diff().dropna().median()
    total_span = span + median_gap
    seconds_per_year = 365.25 * 24 * 60 * 60
    return max(total_span.total_seconds() / seconds_per_year, 1.0 / periods_per_year)


def _is_full_calendar_year(
    year_dates: pd.DatetimeIndex, validation_cfg: dict | None = None
) -> bool:
    """Check if year_dates span a full calendar year.

    Tolerance is profile-driven by explicit calendar class rather than by annualization.
    """
    if len(year_dates) == 0:
        return False
    year_dates = pd.DatetimeIndex(year_dates)
    validation_cfg = validation_cfg or {}
    year = int(year_dates[0].year)
    if year_dates.tz is not None:
        start = pd.Timestamp(year=year, month=1, day=1, tz=year_dates.tz)
        end = pd.Timestamp(year=year, month=12, day=31, tz=year_dates.tz)
    else:
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year, month=12, day=31)
    calendar_type = validation_cfg.get("calendar_type", "business_day")
    tolerance_days = {
        "business_day": 5,
        "calendar_day": 1,
        "intraday": 0,
    }.get(calendar_type, 0)
    tolerance = pd.Timedelta(days=tolerance_days)
    min_date = year_dates.min().normalize()
    max_date = year_dates.max().normalize()
    return min_date <= start + tolerance and max_date >= end - tolerance


def _dsr(pnl, T, K=300, periods_per_year=252):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)."""
    K = _positive_dsr_trials(K)
    sample_size = int(T)
    if sample_size < 2:
        return 0.0

    pnl = np.asarray(pnl, dtype=float)
    std = np.std(pnl, ddof=1)
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0

    sr = float(np.mean(pnl) / std)
    if not np.isfinite(sr):
        return 0.0
    skew = float(sp_stats.skew(pnl))
    raw_kurt = float(sp_stats.kurtosis(pnl, fisher=False))
    if not np.isfinite(skew) or not np.isfinite(raw_kurt):
        return 0.0

    sr_variance_term = 1 - skew * sr + ((raw_kurt - 1) / 4) * sr**2
    sr_std = np.sqrt(max(sr_variance_term, 1e-20) / (sample_size - 1))
    if not np.isfinite(sr_std) or sr_std <= 0:
        return 0.0

    gamma = 0.5772
    expected_max_z = 0.0
    if K > 1:
        z1 = sp_stats.norm.ppf(1 - 1 / K)
        z2 = sp_stats.norm.ppf(1 - 1 / (K * np.e))
        expected_max_z = (1 - gamma) * z1 + gamma * z2
    sr_star = sr_std * expected_max_z
    dsr = float(sp_stats.norm.cdf((sr - sr_star) / sr_std))
    if not np.isfinite(dsr):
        return 0.0
    return float(np.clip(dsr, 0.0, 1.0))


def _positive_dsr_trials(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("DSR trials must be a positive integer")
    parsed = int(value)
    if parsed < 1:
        raise ValueError("DSR trials must be a positive integer")
    return parsed


def _bootstrap_sharpe(pnl, n_boot=1000):
    """Bootstrap p-value for Sharpe > 0."""
    rng = np.random.RandomState(42)
    T = len(pnl)
    boot = [_sharpe(rng.choice(pnl, size=T, replace=True)) for _ in range(n_boot)]
    return float(np.mean(np.array(boot) <= 0))
