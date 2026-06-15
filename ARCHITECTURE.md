# Architecture

## Dependency Direction (left to right only)

```
config.py -> engine/ -> validation/
               ^
          strategies/    (implements engine ABC, no framework dependency)
```

- `config.py` depends on nothing (reads YAML)
- `engine/` depends on `config.py`
- `validation/` depends on executed trade logs, not on strategy internals
- `strategies/` implements `engine/base.py` ABC but can run standalone
- `plugins/` is optional — core works without it

## Data Flow

```
strategies.yaml -> config.py -> trader.py -> engine.compute_signals()
                                                 |
                                          trade_log_*.csv
                                                 |
                                          gate.py -> PASS/FAIL
```

## Validation Flow

```
trade_log.csv -> gate.py -> metrics.py -> PASS/FAIL (exit code 0/1)
                                |
                        metric triangle:
                          Lo-adj Sharpe (ratio, optimized)
                          IC (rank, guardrail)
                          Omega (shape, guardrail)
```

## File Size Limits

- No Python file > 400 lines (enforced by test)
- AGENTS.md: root <=80 lines, subsystem <=60 lines (enforced by test)
