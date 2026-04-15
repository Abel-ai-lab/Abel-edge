# causal-edge — Agent Entry Point

Two modes: **use** this as a tool, or **develop** on this repo.

## Use as a Tool (validate backtests, fix strategies)

Read `CAPABILITY.md` — it has everything: install, validate, diagnose, fix loop.

    python -m venv .venv
    # PowerShell: .venv\Scripts\Activate.ps1
    # bash/zsh: source .venv/bin/activate
    python -m pip install --upgrade pip
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
2. If you do not already have a key, run `causal-edge login`
3. For agent-driven setups, install `causal-abel` and complete OAuth there if needed
4. Otherwise set `ABEL_API_KEY` or `CAP_API_KEY` in your environment or `.env`
5. Run `causal-edge discover <TICKER>`

### Architecture
- `ARCHITECTURE.md` — dependency direction diagram
- `causal_edge/engine/AGENTS.md` — strategy execution
- `causal_edge/dashboard/AGENTS.md` — template rendering
- `causal_edge/validation/AGENTS.md` — metric triangle + fix patterns
- `causal_edge/plugins/AGENTS.md` — plugin isolation rules

### Constraints (enforced by `make test`)
See `CLAUDE.md`. Key: strategies.yaml is single source of truth, no file >400 lines,
strategies/ standalone, AGENTS.md at every subsystem.
