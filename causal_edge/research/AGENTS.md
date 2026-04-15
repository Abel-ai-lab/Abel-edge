# Research Module — Agent Guide

## I want to...

### Start researching a new asset
```bash
causal-edge research init SOLUSD
cd research/solusd
# Edit strategy.py → implement run_strategy()
causal-edge research run -d "baseline"
```

### Run an experiment
```bash
# Edit strategy.py with ONE change
causal-edge research run --mode exploit -d "added xcorr overlay"
# Result is auto-validated, auto-recorded to results.tsv
```

### Check progress
```bash
causal-edge research status
```

## What the harness enforces (you cannot bypass)

- **K auto-computed** from strategy.py AST (tickers × lags)
- **validate_strategy()** runs on every experiment
- **KEEP requires PASS** — append_results_tsv refuses otherwise
- **results.tsv schema** validated on append
- **Look-ahead check** before execution

## What you decide (judgment calls)

- What to write in strategy.py
- Explore vs exploit classification
- When to declare honest failure
- How to interpret validation failures
