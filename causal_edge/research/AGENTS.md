# Research Module - Agent Guide

## I want to...

### Start researching a new asset
```bash
causal-edge research init SOLUSD --branch-id graph-v1
causal-edge research run --workdir research/solusd/<exp_id>/branches/graph-v1 -d "baseline"
```

### Run an experiment
```bash
causal-edge research run --mode exploit -d "added xcorr overlay"
```

### Check progress
```bash
causal-edge research status --workdir research/solusd/<exp_id>
causal-edge research check --workdir research/solusd/<exp_id>
causal-edge research check --strict --workdir research/solusd/<exp_id>
```

## What the harness enforces

- K is auto-computed from strategy.py AST
- validate_strategy() runs on every experiment
- KEEP requires PASS and triangle improvement vs the latest KEEP baseline
- Look-ahead checks run before execution
- Every round writes a markdown note and validation summary alongside results.tsv
- The session appends `events.tsv` so parallel branch activity stays traceable
