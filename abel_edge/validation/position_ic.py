from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def compute_position_ic(asset_returns, positions, dates):
    """Compute position-return IC, active win rate, and monthly stability."""
    active_mask = np.abs(positions) > 0.01
    if active_mask.sum() < 30:
        return 0.0, 0.0, 0.0, 0.0, False, False

    ap, ar = positions[active_mask], asset_returns[active_mask]
    nonzero_return_mask = np.abs(ar) > 1e-12
    hit_rate = (
        float(np.mean(np.sign(ap[nonzero_return_mask]) == np.sign(ar[nonzero_return_mask])))
        if nonzero_return_mask.sum() > 0
        else 0.0
    )
    if np.std(ap) < 1e-10 or np.std(ar) < 1e-10:
        return 0.0, hit_rate, 0.0, 0.0, True, False

    position_ic = float(sp_stats.spearmanr(ap, ar)[0])
    if np.isnan(position_ic):
        position_ic = 0.0

    ad = dates[active_mask]
    monthly_ics = []
    for ym in sorted(set(zip(ad.year, ad.month))):
        m = (ad.year == ym[0]) & (ad.month == ym[1])
        if m.sum() >= 10:
            mp, mr = ap[m], ar[m]
            if np.std(mp) > 1e-10 and np.std(mr) > 1e-10:
                mic = float(sp_stats.spearmanr(mp, mr)[0])
                if not np.isnan(mic):
                    monthly_ics.append(mic)

    stability_applicable = len(monthly_ics) >= 6
    stability = float(np.mean(np.array(monthly_ics) > 0)) if stability_applicable else 0.0
    monthly_mean = float(np.mean(monthly_ics)) if monthly_ics else 0.0
    return position_ic, hit_rate, stability, monthly_mean, True, stability_applicable
