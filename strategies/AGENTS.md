# Strategies

Strategy engine implementations. Each is a directory with an engine module such
as `strategies/ethusd_causal/engine.py` implementing `StrategyEngine` from
`causal_edge/engine/base.py`.

## I want to...

### Add a strategy
1. Create strategies/my_strategy/ directory
2. Create strategies/my_strategy/__init__.py (empty)
3. Create the strategy engine module under `strategies/my_strategy/`
4. Add entry to root `strategies.yaml` with `engine: strategies.my_strategy.engine`
5. `make test` — `TestEngineModuleImportable` verifies import works
6. `causal-edge validate` — runs the Abel Proof audited live gate contract

### Use the SMA example as template
    cp -r examples/sma_crossover/ strategies/my_strategy/

Bundled examples are synthetic demos that show framework wiring and contract
usage. Treat them as starting patterns, not as research-ready real-data
baselines.

### Rules
- Engine wrappers should be < 100 lines
- strategies/ must NOT import from `causal_edge/` (except `causal_edge/engine/base.py`)
- `TestStrategiesStandalone` enforces this mechanically
- All features must use `shift(1)` — zero look-ahead tolerance
- the strategy engine module must define its own `StrategyEngine` subclass; do
  not only import or re-export one
- `self.load_bars()` returns UTC-aware timestamps, so normalize auxiliary series before `reindex(...)`
