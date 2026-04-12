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

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from causal_edge.validation.position_ic import compute_position_ic

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "profiles")


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
    years = T / periods_per_year
    ann_return = (equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else 0.0
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    # Simplified serial-correlation penalty: lag-1 autocorrelation only.
    rho1 = pd.Series(pnl).autocorr(lag=1)
    rho1 = 0.0 if np.isnan(rho1) else float(rho1)
    cf = 1 + 2 * rho1 * (1 - 1 / periods_per_year)
    lo_adjusted = sharpe * np.sqrt(1 / cf) if cf > 0 else sharpe

    dsr_trials_used = dsr_trials if dsr_trials is not None else validation_cfg.get("dsr_K", 300)
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
        if _is_full_calendar_year(year_dates, periods_per_year=periods_per_year):
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
    var_5 = float(np.percentile(pnl, 5))
    cvar_5 = float(np.mean(pnl[pnl <= var_5])) if np.any(pnl <= var_5) else var_5

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
# Validation
# ═══════════════════════════════════════════════════════════════════


def validate(metrics: dict, profile: dict) -> tuple[bool, list[str]]:
    """Run validation gate. Returns (passed, list_of_failures)."""
    v = profile.get("validation", {})
    ag = profile.get("anti_gaming", {})
    failures = []

    if metrics["dsr"] < v.get("dsr_min", 0.90):
        failures.append(f"T6 DSR {metrics['dsr']:.1%} < {v['dsr_min']:.0%}")
    if metrics["drawdown_time_frac"] > v.get("drawdown_time_frac_max", 0.35):
        failures.append(
            f"T13 DrawdownTime {metrics['drawdown_time_frac']:.0%} > {v['drawdown_time_frac_max']:.0%}"
        )
    max_dd_bars_limit = v.get("max_drawdown_duration_bars_max")
    if max_dd_bars_limit is not None and metrics["max_drawdown_duration_bars"] > max_dd_bars_limit:
        failures.append(
            f"T13 MaxDDDuration {metrics['max_drawdown_duration_bars']} > {max_dd_bars_limit} bars"
        )
    if metrics.get("loss_years_applicable", False) and metrics["loss_years"] > v.get(
        "max_loss_years", 2
    ):
        failures.append(f"T14 LossYrs {metrics['loss_years']} > {v['max_loss_years']}")
    if metrics["lo_adjusted"] < v.get("lo_adjusted_min", 1.0):
        failures.append(f"T15 Lo {metrics['lo_adjusted']:.2f} < {v['lo_adjusted_min']}")
    if metrics.get("omega_applicable", False) and metrics["omega"] + 1e-12 < v.get(
        "omega_min", 1.0
    ):
        failures.append(f"T15 Omega {metrics['omega']:.2f} < {v['omega_min']}")
    if metrics["max_dd"] < v.get("max_dd", -0.20):
        failures.append(
            f"T15 MaxDD {abs(metrics['max_dd']) * 100:.1f}% > {abs(v['max_dd']) * 100:.0f}%"
        )
    if metrics["total_return"] < ag.get("return_floor", 1.0):
        failures.append(
            f"Return floor {metrics['total_return'] * 100:+.1f}% < +{ag['return_floor'] * 100:.0f}%"
        )
    if (
        metrics["sharpe"] > 0
        and metrics["lo_adjusted"] > 0
        and metrics["sharpe_lo_ratio"] > ag.get("sharpe_lo_ratio_max", 2.5)
    ):
        failures.append(
            f"Sharpe/Lo {metrics['sharpe_lo_ratio']:.1f} > {ag['sharpe_lo_ratio_max']}"
        )
    if metrics.get("position_ic_applicable", False) and metrics["position_ic"] < ag.get(
        "position_ic_min", 0.02
    ):
        failures.append(f"PositionIC {metrics['position_ic']:.3f} < {ag['position_ic_min']}")
    if metrics.get("position_ic_stability_applicable", False) and metrics[
        "position_ic_stability"
    ] < ag.get("position_ic_stability_min", 0.55):
        failures.append(
            f"PositionIC stab {metrics['position_ic_stability']:.0%} < {ag['position_ic_stability_min']:.0%}"
        )
    return len(failures) == 0, failures


def decide_keep_discard(current: dict, baseline: dict, profile: dict) -> str:
    """Metric triangle KEEP/DISCARD decision.

    Three leverage-invariant, orthogonal dimensions:
      Ratio (Lo-adj or Sharpe) — optimized, must improve
      Rank  (Position IC)      — guardrail, must not degrade
      Shape (Omega)            — guardrail, catches clipping

    MaxDD is an absolute gate (not relative — scales with leverage).
    """
    mt = profile.get("metric_triangle", {})
    ag = profile.get("anti_gaming", {})
    v = profile.get("validation", {})

    opt_key = {"lo_adjusted_sharpe": "lo_adjusted", "sharpe": "sharpe"}.get(
        mt.get("optimize", "lo_adjusted_sharpe"), "lo_adjusted"
    )
    if current.get(opt_key, 0) <= baseline.get(opt_key, 0):
        return "DISCARD"

    for guard in mt.get("guardrails", []):
        key = {
            "raw_sharpe": "sharpe",
            "ic": "position_ic",
            "position_ic": "position_ic",
            "omega": "omega",
            "total_pnl": "total_return",
            "total_return": "total_return",
        }.get(guard["metric"], guard["metric"])
        tol = guard.get("tolerance", 0)
        if key == "total_return" and baseline.get(key, 0) > 0:
            if current.get(key, 0) < baseline[key] * (1 - tol):
                return "DISCARD"
        else:
            if current.get(key, 0) < baseline.get(key, 0) - tol:
                return "DISCARD"

    if current.get("max_dd", 0) < v.get("max_dd", -0.25):
        return "DISCARD"

    return "KEEP"


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


def _is_full_calendar_year(year_dates: pd.DatetimeIndex, periods_per_year: int = 252) -> bool:
    """Check if year_dates span a full calendar year.

    Tolerance is profile-driven:
      - Equity (252 periods/yr): ±5 days to handle non-trading Jan 1/Dec 31.
      - Crypto/24-7 (365 periods/yr): ±1 day (trades every calendar day).
    """
    if len(year_dates) == 0:
        return False
    year_dates = pd.DatetimeIndex(year_dates)
    year = int(year_dates[0].year)
    if year_dates.tz is not None:
        start = pd.Timestamp(year=year, month=1, day=1, tz=year_dates.tz)
        end = pd.Timestamp(year=year, month=12, day=31, tz=year_dates.tz)
    else:
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year, month=12, day=31)
    tolerance_days = 5 if periods_per_year <= 252 else 1
    tolerance = pd.Timedelta(days=tolerance_days)
    return year_dates.min() <= start + tolerance and year_dates.max() >= end - tolerance


def _dsr(pnl, T, K=300, periods_per_year=252):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)."""
    std = np.std(pnl, ddof=1)
    if std == 0:
        return 0
    sr_d = (np.mean(pnl) / std) * np.sqrt(periods_per_year)
    skew = float(sp_stats.skew(pnl))
    raw_kurt = float(sp_stats.kurtosis(pnl, fisher=False))
    gamma = 0.5772
    z1 = sp_stats.norm.ppf(1 - 1 / K)
    z2 = sp_stats.norm.ppf(1 - 1 / (K * np.e))
    emax = ((1 - gamma) * z1 + gamma * z2) / np.sqrt(T)
    var_sr = (1 / T) * (1 - skew * sr_d + (raw_kurt / 4) * sr_d**2)
    return float(sp_stats.norm.cdf((sr_d - emax) / np.sqrt(max(var_sr, 1e-20))))


def _bootstrap_sharpe(pnl, n_boot=1000):
    """Bootstrap p-value for Sharpe > 0."""
    rng = np.random.RandomState(42)
    T = len(pnl)
    boot = [_sharpe(rng.choice(pnl, size=T, replace=True)) for _ in range(n_boot)]
    return float(np.mean(np.array(boot) <= 0))
