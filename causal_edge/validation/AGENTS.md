# Validation Subsystem — Abel Proof Gate

Three leverage-invariant dimensions:
- **Ratio**: Lo-adjusted Sharpe
- **Rank**: Position-Return IC
- **Shape**: Omega

## I want to...
### Validate a strategy
    causal-edge validate --strategy <ID> --verbose

### Understand the live contract
- Denominators are applicability-based, not legacy `20/21` style scores.
- Typical base denominator is `5`, then add slots only when `Omega`, `LossYrs`, `position_ic`, or `position_ic_stability` are applicable.
- DSR accepts optional declared exploration counts through `dsr_trials`.
- Deferred gates and removed profile keys live in `deferred_registry.yaml`.
- Audit timing and comparability notes live in `docs/validation-audit-matrix.md`.

### Check for look-ahead
- Static source checks: `T2-T5`
- Runtime leak diagnostics: `R1-R2`
- Semantic repair checklist: `look_ahead_rules.md`

### Understand why it failed
| Code | Fix |
|------|-----|
| T6 DSR | Reduce search breadth, declare realistic `dsr_trials` |
| T14 LossYrs | Split regimes or reduce exposure in unstable years |
| T15-Lo | Add persistence penalty to repeated holds |
| T15-Omega | Stop clipping PnL; clip features only |
| T15-MaxDD | Reduce sizing or cap position |

### Common snippets
```python
# Trend filter
sma = prices.rolling(50).mean().shift(1)
positions[prices.shift(1) < sma] = 0.0

# Persistence penalty
hold = (positions > 0).astype(int)
hold_days = hold.groupby((hold != hold.shift()).cumsum()).cumcount()
positions *= np.maximum(0.3, 1.0 - 0.1 * hold_days)
```

## Key Files
- `gate.py` — CSV/path in, PASS/FAIL payload out
- `metrics.py` — metric computation + KEEP/DISCARD logic
- `look_ahead.py` — static/runtime leakage checks
- `look_ahead_rules.md` — semantic review checklist
- `profiles/` — profile thresholds
