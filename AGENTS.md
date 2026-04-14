# causal-edge — Agent Entry Point

Two modes: **use** this as a tool, or **develop** on this repo.

## Use as a Tool (validate backtests, fix strategies)

Read `CAPABILITY.md` — it has everything: install, validate, diagnose, fix loop.

    pip install git+https://github.com/Abel-ai-causality/Abel-edge.git
    causal-edge validate --csv your_backtest.csv

## Develop on This Repo

### I want to...

#### Add a strategy
1. Read `strategies/AGENTS.md`
2. Copy `examples/sma_crossover/` → `strategies/my_strategy/`
3. Edit `strategies.yaml` — add entry (see schema comments in file)
4. `make test` — structural tests verify registration
5. `causal-edge validate` — Abel Proof audited live gate contract

#### Fix a failing validation
1. `causal-edge validate --verbose`
2. Read `causal_edge/validation/AGENTS.md` — failure→fix mapping with code

#### Add a dashboard component
1. Read `causal_edge/dashboard/AGENTS.md`
2. Add pure function to `causal_edge/dashboard/components.py`
3. Register in `causal_edge/dashboard/generator.py`
4. `make test` verifies registration

#### Use Abel causal discovery (optional)
1. Read `causal_edge/plugins/AGENTS.md`
2. Run `causal-edge discover <TICKER>`
3. If you do not already have a key, run `causal-edge login`
4. For agent-driven setups, install `causal-abel` and complete OAuth there if needed
5. Otherwise set `ABEL_API_KEY` or `CAP_API_KEY` in your environment or `.env`

## Abel-Pro Mapping

- Abel-edge worktree for the Abel-Pro integration: `D:\codes\open_source\causal-edge\.tree\abel-pro-demo`
- Abel-edge branch for that worktree: `abel-pro-demo`

### Architecture
- `ARCHITECTURE.md` — dependency direction diagram
- `causal_edge/engine/AGENTS.md` — strategy execution
- `causal_edge/dashboard/AGENTS.md` — template rendering
- `causal_edge/validation/AGENTS.md` — metric triangle + fix patterns
- `causal_edge/plugins/AGENTS.md` — plugin isolation rules

### Constraints (enforced by `make test`)
See `CLAUDE.md`. Key: strategies.yaml is single source of truth, no file >400 lines,
strategies/ standalone, AGENTS.md at every subsystem.
