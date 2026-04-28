# Research Module — Agent Guide

## I want to...

### Evaluate a research branch
```bash
cd <branch-dir>
abel-edge evaluate --workdir .
```

### Run semantic preflight before metrics
```bash
abel-edge debug-evaluate --workdir <branch-dir>
```

### Emit raw artifacts for upstream orchestration
```bash
abel-edge evaluate \
  --workdir <branch-dir> \
  --output-json edge-result.json \
  --output-md edge-validation.md \
  --output-handoff edge-handoff.json
```

### Check whether discovered drivers have usable data
```bash
abel-edge verify-data \
  --discovery-json <session-or-branch>/discovery.json \
  --start 2020-01-01
```

### Validate a handoff
```bash
abel-edge validate-handoff path/to/edge-handoff.json
```

## What the research layer enforces

- `<branch-dir>/engine.py` is required
- the branch must define a module-owned `StrategyEngine` subclass
- branch-default strategies should author against `DecisionContext`
- semantic preflight runs before metric validation
- the backtest kernel interprets `next_position` intent centrally
- static look-ahead checks are warning diagnostics, not the primary safety contract
- `debug-evaluate` surfaces sampled reads, output-shape issues, and semantic blockers
- validation artifacts come from the same `validate_strategy()` gate used elsewhere

## What strategy authors decide

- what mechanism to encode in `compute_decisions(self, ctx)`
- which feeds and driver relationships to test
- how to interpret semantic warnings versus metric failures
- how to classify explore vs exploit at the orchestration layer
