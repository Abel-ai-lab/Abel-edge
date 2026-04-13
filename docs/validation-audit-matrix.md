# Validation Audit Matrix

## Scope

This is the long-lived validation contract reference for the `validate` subsystem.
It records the current runtime contract, score semantics, and migration notes for the
public validation surface.

## Runtime Surfaces

| Surface | File | Current contract |
|---|---|---|
| Metric computation | `causal_edge/validation/metrics.py`, `causal_edge/validation/position_ic.py` | Live payload computes ratio, shape, and Position-Return IC metrics from `pnl`, optional `position`, and optional `asset_return` |
| Gate evaluation | `causal_edge/validation/metrics.py` | Live failures are conditional on applicability rather than fixed legacy gate counts |
| Result contract | `causal_edge/validation/gate.py` | `validate_strategy()` returns `verdict`, `score`, `failures`, `metrics`, `triangle`, `profile` |
| Trade-log contract | `causal_edge/engine/ledger.py`, `causal_edge/validation/gate.py` | Validation consumes trade logs with `date`, `pnl`, optional `position`, optional `asset_return`, and derived `cum_return` |
| Public wording | `README.md`, `CAPABILITY.md`, `causal_edge/validation/AGENTS.md` | Public docs should describe the applicable-gate live contract rather than legacy fixed-denominator narratives |

## Timing Contract

The engine and validation pipeline relies on this bar-by-bar timing relationship:

```text
price[t-1], price[t] -> asset_return[t]
information through t-1 -> position[t]
position[t] * asset_return[t] -> pnl[t]
cumprod(1 + pnl[:t]) - 1 -> cum_return[t]
```

This implies:

- `asset_return[t]` is the return realized over the interval from `t-1` to `t`
- `position[t]` is the exposure chosen before `asset_return[t]` is realized
- `pnl[t]` is the realized payoff of that pre-chosen exposure over that interval

The live contract therefore forbids any decision path that uses `price[t]` or
`asset_return[t]` when constructing `position[t]`.

## Audit Checklist

Apply these checks when auditing strategy math:

1. Every feature used to determine `position[t]` must be lagged by at least one bar.
2. No decision path may use `price[t]` or `asset_return[t]` when setting `position[t]`.
3. No alignment step may propagate future observations backward into earlier timestamps.
4. The emitted trade log must preserve the interpretation `pnl[t] = position[t] * asset_return[t]`.

## Score Semantics

- The validation score uses an applicable-gate denominator, not a fixed legacy count.
- The base denominator starts from the always-considered checks, then adds optional
  checks only when their supporting inputs and applicability conditions are present.
- `max_drawdown_duration_bars` remains a diagnostic metric in the payload, but no longer contributes a live gate failure or denominator slot.
- `Omega`, `LossYrs`, and Position-Return IC family checks are conditional.
- `validation.max_dd` is the single MaxDD policy key used both for PASS/FAIL and for
  the absolute KEEP/DISCARD veto.

## Migration Notes

- Historical scorecards with older denominators are not directly comparable to the
  current applicable-gate contract.
- Trade logs without `asset_return` remain supported, but they cannot activate the
  full Position-Return IC family.
- Deferred and removed validation items belong in
  `causal_edge/validation/deferred_registry.yaml` rather than in this contract file.
