# Validation Subsystem — Abel Proof Gate

Three leverage-invariant dimensions:

- **Ratio**: Lo-adjusted Sharpe
- **Rank**: Position-Return IC
- **Shape**: Omega

## I want to...

### Validate a strategy
    abel-edge validate --strategy <ID> --verbose

### Understand the live contract

- denominators are applicability-based, not legacy `20/21` style scores
- DSR accepts optional declared exploration counts through `dsr_trials`
- deferred gates and removed profile keys live in `deferred_registry.yaml`
- audit timing and comparability notes live in `docs/validation-audit-matrix.md`

### Understand timing or legality first

- semantic preflight lives in `abel-edge debug-evaluate --workdir ...`
- validation assumes the runtime contract has already produced a legal executed series
- static look-ahead checks remain optional diagnostics, not the main safety story

### Understand why it failed

| Code | Fix |
|------|-----|
| T6 DSR | Reduce search breadth or declare realistic `dsr_trials` |
| T14 LossYrs | Split regimes or reduce exposure in unstable years |
| T15-Lo | Reduce persistence or serially correlated exposure patterns |
| T15-Omega | Stop clipping PnL; clip features or position intent instead |
| T15-MaxDD | Reduce sizing or cap position |

## Mental model

Validation does not decide what the strategy was allowed to see. The runtime
contract and semantic preflight do that first.

Validation answers a different question:

- given the executed output, does it pass the audited quality gate?

## Key Files

- `gate.py` — CSV/path in, PASS/FAIL payload out
- `metrics.py` — metric computation plus KEEP/DISCARD logic
- `look_ahead.py` — static/runtime leakage diagnostics
- `look_ahead_rules.md` — lower-level diagnostic checklist
- `profiles/` — profile thresholds
