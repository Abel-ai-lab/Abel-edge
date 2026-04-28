# Adding a Strategy

## Fastest Path: Validate an Existing Backtest

Already have a CSV with `date` and simple-return `pnl` columns? Skip engine
authoring and validate directly:

```bash
abel-edge validate --csv my_backtest.csv
```

Add `position` and `asset_return` columns when you want Position-Return IC
analysis too.

## Choose A Starting Point

| Path | Copy from | What it teaches |
|------|-----------|-----------------|
| Simple | `examples/sma_crossover/` | `compute_decisions(self, ctx)` on the primary target feed |
| ML | `examples/momentum_ml/` | vectorized `DecisionContext` features plus walk-forward training |
| Feed Path | `examples/feed_overlay_demo/` | declared auxiliary feeds through `ctx.feed(name)` |
| Causal | `examples/causal_demo/` | graph-shaped voting over named driver feeds |

`abel-edge init` scaffolds the first three paths by default with local sample
CSV data. `causal_demo` remains an optional repo example when you want a
graph-shaped pattern.

## Quick Path

1. Copy an example into `strategies/my_strategy/`.
2. Declare primary `price_data` and any auxiliary `feeds` in `strategies.yaml`.
3. Implement `compute_decisions(self, ctx)`.
4. Run `abel-edge run --strategy my_strategy`.
5. Run `abel-edge validate --strategy my_strategy`.

If you are iterating on one strategy workspace directly, use semantic
preflight:

```bash
abel-edge debug-evaluate --workdir strategies/my_strategy
```

## Engine Interface

New strategies should implement the branch-default decision contract:

```python
from abel_edge.engine.base import StrategyEngine


class MyEngine(StrategyEngine):
    def compute_decisions(self, ctx):
        close = ctx.target.series("close")
        driver = ctx.feed("btc_ref").asof_series("close")
        next_position = (close > driver).astype(float).fillna(0.0)
        return ctx.decisions(next_position)
```

That contract means:

- the runtime owns the decision index
- the strategy reads market data through `DecisionContext`
- the strategy emits `next_position` intent
- the backtest kernel compiles intent into effective exposure

Legacy `compute_signals()` engines still exist internally during the rollout,
but they are not the authoring surface new strategies should learn from.

## DecisionContext Surface

Use these runtime-owned reads inside `compute_decisions(self, ctx)`:

- `ctx.target.series("close")`
- `ctx.feed(name).native_series("close")`
- `ctx.feed(name).asof_series("close")`
- `ctx.points()`
- `ctx.trace_point(date)`
- `ctx.decisions(next_position)`

Choose the surface that matches the mechanism:

- batch/vectorized reads for feature engineering and ML
- point reads for explainability, interval reasoning, and debugging

Do not call raw frame loaders or ad hoc alignment helpers from inside
`compute_decisions()`. If the data you need is not representable through
`DecisionContext`, that is a framework or contract issue to fix explicitly.

## External Data Contract

Primary price data is declared through `price_data`. Every non-primary external
input is declared through `feeds`.

Example:

```yaml
strategies:
  - id: my_strategy
    name: "My Strategy"
    asset: ETHUSD
    color: "#FF2D55"
    engine: strategies.my_strategy.engine
    trade_log: data/trade_log_my_strategy.csv
    price_data:
      adapter: csv
      path: data/ethusd.csv
      symbol: ETHUSD
    feeds:
      btc_ref:
        kind: bars
        adapter: csv
        path: data/btcusd.csv
        symbol: BTCUSD
      risk_scale:
        kind: series
        adapter: csv
        path: data/risk_scale.csv
        field: value
```

File-backed CSV feeds on the supported daily path may use either
`2026-01-01T00:00:00Z` or a naive date like `2026-01-01`. The framework
normalizes file-backed timestamps into the runtime daily contract during load.

## Timing And Semantics

The strategy decides the next exposure from the current decision-time world:

```text
decision-time data at t -> next_position[t]
kernel applies execution delay -> effective position
effective position * asset_return -> pnl
```

That is why the strategy should return `ctx.decisions(next_position)` instead
of hand-compiling an already-effective position series.

## Rules

- implement `compute_decisions(self, ctx)` for new strategies
- read market data only through `DecisionContext`
- return `ctx.decisions(next_position)`
- declare primary `price_data` and all auxiliary `feeds` explicitly
- let runtime profile and execution constraints stay system-owned
- use `debug-evaluate` when you need semantic feedback about visibility or timing
- treat static look-ahead heuristics as optional diagnostics, not the primary safety story

## Standalone Vs Abel-alpha Branches

`abel-edge init` gives you a standalone project with local sample data and a
few runnable examples.

Inside an Abel-alpha branch workspace, the upstream branch flow should prepare
runtime inputs first, then the engine should be authored against the injected
`DecisionContext` world. The strategy contract stays the same; the orchestration
layer changes.
