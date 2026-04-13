# Validation Subsystem — Abel Proof Gate

Three leverage-invariant dimensions:
- **Ratio** (Lo-adjusted Sharpe) — optimized
- **Rank** (Position-Return IC) — guardrail on whether larger positions align with better underlying returns
- **Shape** (Omega) — guardrail, catches clipping

## I want to...

### Validate a strategy
    causal-edge validate --strategy <ID> --verbose

### Contract notes
- The audited live validation contract uses applicable-gate denominators rather than legacy `20/21` score narratives.
- Typical denominators start at `5`, add `+1` for `Omega` applicability, `+1` for full-year `LossYrs` applicability, `+1` for `position_ic_applicable`, and `+1` for `position_ic_stability_applicable`.
- DSR accepts optional externally declared exploration counts via `dsr_trials`; otherwise it falls back to the profile default `validation.dsr_K`.
- Deferred/removed gates and profile keys are tracked in `causal_edge/validation/deferred_registry.yaml`.
- The long-lived timing and audit contract is summarized in `docs/validation-audit-matrix.md`.

### Understand why it failed

| Code | Fix | How |
|------|-----|-----|
| T6 DSR | Reduce trials | Fewer param combos in grid search. Declare realistic `dsr_trials`; K<50 ideal |
| T14 LossYrs | Reduce full-year losses | Split regimes, de-risk bad periods, or narrow the strategy to the years it truly supports |
| T15-Lo | Fix serial corr | Persistence penalty: `pos[t] *= max(0.3, 1-0.1*hold_days)` |
| T15-Omega | Stop clipping | Use raw returns for PnL: `pnl = pos * returns` not `clip()` |
| T15-MaxDD | Reduce sizing | Cap position: `pos = min(pos, 0.5)` |

### Common fix patterns

**Trend filter (fixes T13):**
```python
sma = prices.rolling(50).mean().shift(1)
positions[prices.shift(1) < sma] = 0.0
```

**Persistence penalty (fixes T15-Lo):**
```python
hold = (positions > 0).astype(int)
hold_days = hold.groupby((hold != hold.shift()).cumsum()).cumcount()
positions *= np.maximum(0.3, 1.0 - 0.1 * hold_days)
```

**Unclipped PnL (fixes T15-Omega):**
```python
# WRONG: pnl = pos * np.clip(returns, -0.02, 0.02)
# RIGHT: pnl = pos * returns  (clip features only, never PnL)
```

### Understand the metric triangle
Read docstring at top of `metrics.py`. No known transformation improves all three simultaneously except genuine signal improvement.

### Diagnostic-only metrics
- `drawdown_time_frac` remains in the payload for audit and diagnostics, but it is no longer a live PASS/FAIL gate.
- `max_drawdown_duration_bars` remains in the payload for audit and diagnostics, but it is no longer a live PASS/FAIL gate.

## Key Files
- `metrics.py` — `compute_all_metrics()`, `validate()`, `decide_keep_discard()`
- `gate.py` — `validate_strategy()` (CSV in → PASS/FAIL out)
- `profiles/` — YAML threshold configs (crypto_daily, equity_daily, hft)
- `deferred_registry.yaml` — removed/deferred gates, metrics, and profile keys
