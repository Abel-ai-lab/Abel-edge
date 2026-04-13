# Changelog

## [Unreleased]

### Changed
- **Validation contract migration**: `validate` no longer uses the legacy `20/21`-style denominator narrative. The audited live contract now reports denominators based on applicable gates: typically `5` or `7`, rising further when `Omega` and full-year loss accounting are applicable.
- **Stability gate redesign**: `T13 NegRoll` was replaced by diagnostic drawdown-time payloads on underwater bar fraction and longest underwater duration rather than live PASS/FAIL gates.
- **Longest drawdown duration downgraded**: `max_drawdown_duration_bars` remains in the validation payload for audit, but `T13 MaxDDDuration` no longer acts as a live PASS/FAIL gate or score denominator slot.
- **Drawdown-time fraction downgraded**: `drawdown_time_frac` remains in the validation payload for audit, but `T13 DrawdownTime` no longer acts as a live PASS/FAIL gate or score denominator slot.
- **Loss-year contract redesign**: `T14 LossYrs` now counts only full calendar years with negative total PnL, and partial-year backtests no longer activate the gate.
- **Mathematical corrections**: no-loss `omega` now becomes an applicability case instead of a live gate failure, zero-drawdown `calmar` now normalizes to `0.0` instead of sentinel `999`, and constant-series `skew` now normalizes to `0.0` instead of `NaN`.
- **Applicability semantics**: Position-Return IC behavior is now explicit via `position_ic_*_applicable` flags rather than inferred from zero-valued IC metrics.

### Removed / Deferred
- **Unsupported live gate removed**: `T12 OOS/IS` and its split-Sharpe payload family (`oos_is`, `is_sharpe`, `oos_sharpe`) were removed because a final PnL path does not carry defensible IS/OOS provenance.
- **Unsupported live gate removed**: `T7 PBO` and its payload/config family (`pbo`, `_cpcv()`, `validation.pbo_max`) were removed because a single strategy trade log cannot supply the candidate-by-fold structure required for true PBO.
- **Orphaned profile key deferred**: `validation.oos_is_min` was removed from live profiles and recorded in `causal_edge/validation/deferred_registry.yaml`.
- **Unsupported live gate removed**: `Bootstrap p` is no longer part of live validation because it lacked a profile-configurable threshold and public/operator contract.
- **Unused profile keys deferred**: `validation.permutation_trials`, `validation.permutation_p_max`, `validation.look_ahead_mag_corr_max`, `validation.look_ahead_hit_rate_max`, and `anti_gaming.relative_pnl_drop_max` were removed from live profiles and recorded in `causal_edge/validation/deferred_registry.yaml`.

### Comparability
- Historical validation scores are **not directly comparable** across this migration when they relied on the old denominator narrative (`15`, `20`, `21`) or on sentinel metric values (`omega=999`, `calmar=999`). Compare runs only within the same audited contract version.

## [0.1.0] - 2026-04-02

### Added
- **Framework core**: StrategyEngine ABC, config loader, CLI (init/run/dashboard/validate/discover/status)
- **Abel Proof validation**: initial validation gate with anti-gaming metric triangle (Lo-adjusted Sharpe, Position-Return IC, Omega)
- **Dashboard**: Dark-theme static HTML with Plotly equity curves and position charts
- **3 demo strategies**: SMA crossover (simple), Momentum ML (walk-forward GBDT), Causal Voting (Abel graph)
- **Causal demo**: Bundled TON causal graph (5 parents + 3 children from Abel), vote² sizing, conviction threshold
- **Agent-native architecture**: CAPABILITY.md for capability acquisition, AGENTS.md decision trees, 15 structural tests
- **Autonomous workflow**: validate → diagnose → fix loop with copy-paste code snippets
- **Self-internalization**: 4 levels (skill, memory, CLAUDE.md, inline knowledge)
- **Project scaffolding**: `causal-edge init` creates harness-contagious project with 3 demos
- **Quick validation**: `causal-edge validate --csv` for instant backtest validation, `--export` for sharing
