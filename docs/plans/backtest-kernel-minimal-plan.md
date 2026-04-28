# Superseded

This plan describes the legacy `compute_signals() -> (positions, dates, prices)`
contract and is kept only as historical context.

For the active authoring/runtime model, see the agent-first rollout documents
in the mono workspace and the current `DecisionContext` runtime in `abel-edge`.

# Backtest Kernel Minimal Plan

## Goal

Create a small deterministic backtest kernel under `abel_edge/engine/` so execution semantics stop living inside CLI-orchestrator code.

This iteration is intentionally narrow:

- keep strategy-facing changes minimal
- keep execution speed as a first-order constraint
- add only two execution controls to the runtime contract
- do not expand scope into paper trading or data-source redesign

## In Scope

This task should deliver:

1. a small backtest kernel module that owns the core execution math
2. a runtime config surface for execution cost in percentage form
3. a runtime config surface for leverage cap / absolute position cap
4. a single canonical path for `positions + prices -> asset_return + pnl + trade_log rows`
5. tests that lock the new math and keep old behavior when new settings are unset

## Out Of Scope

This task must not include:

- paper trading redesign
- `paper_log` contract changes
- data source abstraction redesign
- market-specific rules like T+1, lot size, short borrow, funding, liquidation
- strategy generation workflow changes
- new artifact families such as `metrics.csv`, `equity.csv`, `trades.csv`

Those can be addressed in later iterations after this kernel boundary is stable.

## Guiding Principles

1. Smallest correct change.
2. Preserve current user-visible workflow: `strategies.yaml -> abel-edge run -> trade_log.csv -> validate/dashboard`.
3. Do not widen strategy obligations. Existing `compute_signals()` engines should keep working.
4. Move execution semantics out of `trader.py`, but do not build a big framework.
5. Default behavior must remain backward-compatible when cost and leverage settings are absent.
6. Keep the kernel vectorized and array-first. No row-by-row simulation unless strictly required.
7. Make the execution contract more explicit before making it more sophisticated.

## Target End State

After this iteration, the execution flow should be:

```text
strategies.yaml
  -> config.py
  -> trader.py
  -> engine.compute_signals()
  -> backtest kernel
  -> ledger.write_trade_log()
```

The kernel should be the only place that defines:

- `asset_return[t]`
- leverage clipping / position cap
- turnover estimate used for cost deduction
- `pnl[t]`

`trader.py` should become orchestration only:

- load engine
- load bars
- call strategy
- call kernel
- write results
- print status

## Minimal Runtime Contract

The kernel should accept:

- `positions`: strategy-intended exposure series
- `prices`: aligned close-price series
- optional execution settings

The kernel should return arrays ready for ledger output:

- `positions`: effective positions after leverage cap
- `asset_returns`
- `pnl`

### Execution Cost

Use a simple proportional turnover model.

Proposed config key:

```yaml
settings:
  execution:
    cost_bps: 0
    max_abs_position: null
```
```

Interpretation:

- `cost_bps` is the one-way execution cost in basis points per unit turnover
- turnover at bar `t` is `abs(position[t] - position[t-1])`
- cost deduction at bar `t` is `turnover[t] * cost_bps / 10000`

Resulting math:

```text
raw_position[t] = strategy output
position[t] = clip(raw_position[t], -max_abs_position, +max_abs_position)  # when configured
asset_return[t] = close[t] / close[t-1] - 1
gross_pnl[t] = position[t] * asset_return[t]
turnover[t] = abs(position[t] - position[t-1])
cost[t] = turnover[t] * cost_bps / 10000
net_pnl[t] = gross_pnl[t] - cost[t]
```

This is intentionally simple and fast.

### Leverage Constraint

Use a single absolute position cap.

Interpretation:

- if `max_abs_position` is unset, preserve current behavior
- if set, clip every position into `[-cap, +cap]`

This is enough for the current engine contract because engines already output direct position size, not notional orders.

## Recommended File Shape

Keep this small. A good first cut is:

- `abel_edge/engine/backtest.py`
  - `BacktestSettings` dataclass or small helper
  - `run_backtest(positions, prices, settings=None)`
- `abel_edge/engine/trader.py`
  - call the kernel instead of computing returns/PnL inline
- `abel_edge/config.py`
  - validate the new `settings.execution` block

Avoid introducing more layers than this in the first pass.

## Implementation Sequence

1. Add config parsing and validation for `settings.execution.cost_bps` and `settings.execution.max_abs_position`.
2. Add a tiny kernel module that computes effective positions, returns, turnover, and net PnL.
3. Replace the inline math in `trader.run_one()` with a call into the kernel.
4. Keep `ledger.py` unchanged unless a small docstring or column comment is needed.
5. Add tests for:
   - default behavior unchanged when execution settings are absent
   - positive cost reduces PnL by turnover-scaled amount
   - max position cap clips oversized exposures
   - first bar handling remains explicit and deterministic
6. Run targeted CLI and validation tests to confirm no regression in existing workflow.

## Test Expectations

At minimum, add or update tests covering:

1. kernel math on a tiny synthetic series
2. `run` path with execution settings enabled
3. config validation failure cases for negative cost or non-positive cap
4. backward-compatible behavior when `settings.execution` is missing

The most important invariant to lock is:

```text
net_pnl[t] = clipped_position[t] * asset_return[t] - turnover_cost[t]
```

Validation should continue to operate on emitted `pnl`, with the known trade-off that the existing PnL consistency warning may fire when costs are enabled unless that warning is intentionally refined in the same pass.

## Known Trade-Offs Accepted In This Iteration

These are acceptable for now:

1. cost model is linear and close-to-close only
2. no distinction between entry and exit venues
3. no spread model
4. no borrow or funding model
5. no market-specific fill rules
6. no separate gross-vs-net artifact outputs

The point of this iteration is not realism parity with a full simulator. The point is to establish a clean execution boundary with the smallest useful amount of real execution semantics.

## Non-Goals For The Code Review

During implementation review, avoid expanding into:

- order objects
- fill events
- broker simulation
- intraday execution hooks
- provider fallback chains
- paper/live state unification

If a change starts pulling in any of those, it is out of scope for this pass.

## Definition Of Done

This plan is complete when:

1. `trader.py` no longer computes returns and PnL directly
2. a dedicated kernel module owns execution-cost and leverage-cap math
3. existing strategies still run without modification
4. execution settings are optional and validated
5. tests prove old behavior is preserved by default and new behavior activates only when configured

## Next Step After This Plan

Once this minimal kernel refactor lands, the next planning pass can decide whether to:

1. add canonical execution artifacts beyond `trade_log.csv`
2. narrow the strategy contract further
3. unify backtest and paper execution semantics
4. redesign data/market adapters
