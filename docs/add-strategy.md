# Adding a Strategy

## Fastest Path: Validate an Existing Backtest

Already have a CSV with `date` and simple-return `pnl` columns? Skip everything — just validate:

```bash
causal-edge validate --csv my_backtest.csv
```

That's it. You'll get an audited validation report card in 2 seconds. No engine, no YAML, no setup.

Add `position` and `asset_return` columns for Position-Return IC analysis.

Read `docs/validation-audit-matrix.md` for the long-lived timing and validation contract.

## Build a Strategy Engine

### Three starting points

| Path | Copy from | What you get |
|------|-----------|-------------|
| Simple | `examples/sma_crossover/` | 30-line minimal engine |
| ML | `examples/momentum_ml/` | Walk-forward GBDT with shift(1) |
| Causal | `examples/causal_demo/` | Abel graph voting + causal_graph.json |

### Quick Path

1. Copy the example:

```bash
cp -r examples/sma_crossover/ strategies/my_strategy/
# or: cp -r examples/causal_demo/ strategies/my_strategy/
```

2. Edit `strategies/my_strategy/engine.py` — implement your signal logic

3. Add to `strategies.yaml`:

```yaml
strategies:
  - id: my_strategy
    name: "My Strategy"
    asset: ETH
    color: "#FF2D55"
    engine: strategies.my_strategy.engine
    trade_log: "data/trade_log_my_strategy.csv"
    paper_log: "data/paper_log_my_strategy.csv"
    # Optional: default live price source is Abel. Override with CSV if needed.
    # price_data:
    #   source: csv
    #   path: data/prices.csv
```

4. Verify:

```bash
make test                              # structural tests pass
causal-edge run --strategy my_strategy # generates trade log
causal-edge validate --strategy my_strategy  # Abel Proof gate
```

## Engine Interface

Your engine must implement `StrategyEngine` from `causal_edge/engine/base.py`:

```python
class MyEngine(StrategyEngine):
    def compute_signals(self):
        # Optional: load price bars via self.load_bars()
        # Returns: (positions, dates, prices)
        # positions: np.ndarray of daily position sizes (0=flat, 1=long)
        # dates: pd.DatetimeIndex
        # prices: np.ndarray of daily closing prices
        ...

    def get_latest_signal(self):
        # Returns: dict with at least 'position' key
        ...
```

## Rules

- All features must use `shift(1)` — zero look-ahead tolerance
- `rolling().mean()` must be followed by `.shift(1)` before use in decisions
- Clip returns for training features only, use unclipped for PnL
- strategies/ must not import causal_edge/ internals (except base.py)

## Timing Contract

Validation assumes this bar-by-bar relationship:

```text
price[t-1], price[t] -> asset_return[t]
information through t-1 -> position[t]
position[t] * asset_return[t] -> pnl[t]
cumprod(1 + pnl[:t]) - 1 -> cum_return[t]
```

## Audit Checklist

- Every feature used to determine `position[t]` must be lagged by at least one bar.
- No decision path may use `price[t]` or `asset_return[t]` when setting `position[t]`.
- No alignment step may propagate future observations backward into earlier timestamps.
- The emitted trade log must preserve `pnl[t] = position[t] * asset_return[t]`.
- If you enable paper trading, keep live paper rows in `paper_log` so validation and backtests stay isolated.
