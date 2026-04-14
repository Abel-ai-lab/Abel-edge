# Look-Ahead Review Rules

Use these rules when a strategy trips the static or runtime look-ahead checks in
`causal-edge validate` or `causal-edge research run`.

## Rules

### R-SHIFT
Every feature used for today's decision must be shifted first.

```python
# VIOLATION
signal = ret.rolling(20).mean()
positions = np.where(signal > 0, 1, 0)

# CLEAN
signal = ret.rolling(20).mean().shift(1)
positions = np.where(signal > 0, 1, 0)
```

### R-ROLLING
`rolling().mean/std/sum/var/median/min/max/corr()` includes the current row.
Shift the result before using it for decisions.

```python
vol = ret.rolling(20).std().shift(1)
```

### R-GLOBAL
Do not build decision features from full-array statistics such as
`np.mean(series)` or `np.std(series)`.

```python
# CLEAN
z = (ret - ret.expanding().mean().shift(1)) / ret.expanding().std().shift(1)
```

### R-WF
Walk-forward training windows must exclude the current bar.

```python
# CLEAN
X_train = X[:i]
y_train = y[:i]
```

### R-TREND
Trend filters for today's position must use yesterday's values.

```python
# CLEAN
if close[i - 1] < sma[i - 1]:
    positions[i] = 0
```

### R-CORR
Cross-correlations need a second shift after `.corr()`.

```python
# CLEAN
xcorr = parent.shift(14).rolling(60).corr(target.shift(1)).shift(1)
```

### R-EXPANDING
`expanding()` also includes the current row. Shift after the aggregate.

```python
threshold = series.expanding().median().shift(1)
```

## Reporting

Report violations as short findings:

```text
SEMANTIC VIOLATION: R-SHIFT L42 — feature used without shift
SEMANTIC VIOLATION: R-CORR L78 — corr() output missing trailing shift
SEMANTIC CLEAN: no violations found
```
