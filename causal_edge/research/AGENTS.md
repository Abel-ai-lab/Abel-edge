# Research Module — Agent Guide

## I want to...

### Evaluate a research branch
```bash
cd <branch-dir>
causal-edge evaluate --workdir .
```

### Emit raw artifacts for upstream orchestration
```bash
causal-edge evaluate \
  --workdir <branch-dir> \
  --output-json edge-result.json \
  --output-md edge-validation.md \
  --output-handoff edge-handoff.json
```

### Check whether discovered drivers have usable data
```bash
causal-edge verify-data \
  --discovery-json <session-or-branch>/discovery.json \
  --start 2020-01-01
```

### Debug why a branch is dead or failing
```bash
causal-edge debug-evaluate --workdir <branch-dir>
```

### Validate a handoff
```bash
causal-edge validate-handoff path/to/edge-handoff.json
```

## What the research layer enforces

- `engine.py` is required
- the branch must define a module-owned `StrategyEngine` subclass
- K is auto-computed from `engine.py` AST
- static look-ahead checks run before evaluation
- engine outputs are validated through the shared signal contract
- validation artifacts come from the same `validate_strategy()` gate used elsewhere
- `verify-data` reports which discovered tickers have full, partial, missing, or broken history
- `debug-evaluate` surfaces runtime diagnostics such as flat signals, constant positions, and alignment collapse

## What strategy authors decide

- what to write in `engine.py`
- which causal drivers and lags to test
- how to classify explore vs exploit at the orchestration layer
- how to interpret failures and iterate
