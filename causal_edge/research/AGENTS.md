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

## What strategy authors decide

- what to write in `engine.py`
- which causal drivers and lags to test
- how to classify explore vs exploit at the orchestration layer
- how to interpret failures and iterate
