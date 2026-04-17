# Data Feed Contract Status

This document records the current implementation status of the adapter-backed
data-feed contract described in
[`data-feed-contract-plan.md`](data-feed-contract-plan.md).

## Current State

The framework now owns:

- primary-feed synthesis from `price_data`
- auxiliary-feed declaration and loading from `feeds`
- adapter registration for built-in and project-local data backends
- timestamp normalization into the runtime UTC-aware daily contract
- alignment gates between auxiliary feeds and strategy dates
- signal-output validation before backtest and paper execution

Third-party projects now own only adapter-specific external-data access. They do
not bypass framework normalization, alignment, or output validation.

## Milestone Status

- Milestone A: done
- Milestone B: done
- Milestone C: done
- Milestone D: done
- Milestone E: done
- Milestone F: effective and evidenced by downstream migration validation

## Milestone F Evidence

`trading-internal` completed a migrated rerun of all registered strategies on
2026-04-16. The reviewed backup comparisons confirm that the new adapter-backed
path preserved the established signal behavior on the representative strategies
used to validate the migration:

| Strategy | Overlap Rows | Overlap PnL Delta | Position Corr | PnL Corr |
| --- | ---: | ---: | ---: | ---: |
| `seven_comp` | 1530 | `+0.011844` | `0.985706` | `0.997453` |
| `dr_v2` | 1530 | `+0.169192` | `1.000000` | `0.999711` |
| `dual_resonance` | 1530 | `+0.187342` | `1.000000` | `0.999717` |

This is sufficient to treat Milestone F as effective for the framework
refactor. Broader downstream reruns remain operational follow-through, not an
architecture blocker.

## Near-Term Cleanup Policy

- keep docs and examples aligned with `adapter` terminology
- prefer project-local adapters over expanding framework core with project-owned
  backend logic
- keep runtime validation strict and framework-owned
- avoid adding static direct-I/O guards unless runtime gates prove insufficient
