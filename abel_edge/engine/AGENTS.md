# Engine Subsystem

Strategy execution framework. New engines should implement `StrategyEngine`
through the decision contract:

- `compute_decisions(self, ctx)`
- `ctx.target.series(...)`
- `ctx.feed(name)...`
- `ctx.decisions(next_position)`

Legacy `compute_signals()` engines still exist for internal rollout support,
but they are not the teaching surface for new work.

## I want to...

### Create a new engine
1. Copy `examples/sma_crossover/engine.py`, `examples/momentum_ml/engine.py`, or `examples/feed_overlay_demo/engine.py`.
2. Implement `compute_decisions(self, ctx)`.
3. Declare `price_data` and any auxiliary `feeds` in `strategies.yaml`.
4. Register the strategy engine module path.

### Understand the execution flow
```text
strategies.yaml
  -> config.py
  -> engine.compute_runtime_output()
  -> backtest kernel compiles next_position into effective exposure
  -> ledger.py writes trade log
```

### Debug a semantic or visibility problem
1. Check the engine only reads market data through `DecisionContext`.
2. Check the required feeds are declared in `strategies.yaml`.
3. If this is a research workspace, run `abel-edge debug-evaluate --workdir ...`.
4. Inspect the semantic verdict and decision trace before looking at metrics.

### Debug "engine not importable"
1. Check the engine path in `strategies.yaml`.
2. Check `__init__.py` exists in the strategy directory.
3. Run `python -c "import strategies.my_strategy.engine"`.
4. Run the engine loader tests if needed.

## Key Files

- `base.py` — `StrategyEngine`, `DecisionContext`, and runtime helpers
- `decision_context.py` — strategy-visible read surface
- `runtime_contract.py` — `DecisionDraft`, runtime profile, execution constraints
- `trader.py` — orchestrates execution and writes logs
- `ledger.py` — trade-log IO
