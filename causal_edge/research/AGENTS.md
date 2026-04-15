# Research Module - Agent Guide

## I want to...

### Start researching a new asset
```bash
causal-edge research init SOLUSD
cd research/solusd
causal-edge research run -d "baseline"
```

### Run an experiment
```bash
causal-edge research run --mode exploit -d "added xcorr overlay"
```

### Check progress
```bash
causal-edge research status
```

## What the harness enforces

- K is auto-computed from strategy.py AST
- validate_strategy() runs on every experiment
- KEEP requires PASS
- Look-ahead checks run before execution
