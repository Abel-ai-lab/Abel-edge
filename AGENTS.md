# abel-edge — Agent Entry Point

Two modes: **use** this as a tool, or **develop** on this repo.

## Use as a Tool (validate backtests, fix strategies)

Read `CAPABILITY.md` — it has everything: install, validate, diagnose, fix loop.
If you use `abel-edge init`, treat it as a standalone project scaffold with
synthetic demos, not as an Abel-alpha branch workspace.

    python -m venv .venv
    # PowerShell: .venv\Scripts\Activate.ps1
    # bash/zsh: source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install git+https://github.com/Abel-ai-causality/Abel-edge.git
    abel-edge validate --csv your_backtest.csv

## Develop on This Repo

### I want to...

#### Add a strategy
1. Read `strategies/AGENTS.md`
2. Copy `examples/sma_crossover/` → `strategies/my_strategy/`
3. Treat bundled examples as synthetic demos for framework exploration, not as research-ready real-data baselines
4. Edit `strategies.yaml` — add entry (see schema comments in file)
5. `make test` — structural tests verify registration
6. `abel-edge validate` — Abel Proof audited live gate contract

#### Fix a failing validation
1. `abel-edge validate --verbose`
2. Read `abel_edge/validation/AGENTS.md` — failure→fix mapping with code

#### Add a dashboard component
1. Read `abel_edge/dashboard/AGENTS.md`
2. Add pure function to `abel_edge/dashboard/components.py`
3. Register in `abel_edge/dashboard/generator.py`
4. `make test` verifies registration

#### Use Abel causal discovery (optional)
1. Read `abel_edge/plugins/AGENTS.md`
2. If you do not already have a key, prefer installing `causal-abel` and completing OAuth there for agent-driven setups
3. `abel-edge` reuses shared auth from project-local `.agents/skills/causal-abel`, known OpenCode/Codex global installs, or `ABEL_AUTH_ENV_FILE`
4. Run `abel-edge login` only when you want the standalone fallback that stores `ABEL_API_KEY` directly for the project
   When you do, surface the authorization URL immediately; `abel-edge login --json --no-browser` now prints it to `stderr` while also emitting JSON events on `stdout`
5. Otherwise set `ABEL_API_KEY` or `CAP_API_KEY` in your environment or `.env`
6. Run `abel-edge discover <TICKER>`

### Architecture
- `ARCHITECTURE.md` — dependency direction diagram
- `abel_edge/engine/AGENTS.md` — strategy execution
- `abel_edge/dashboard/AGENTS.md` — template rendering
- `abel_edge/validation/AGENTS.md` — metric triangle + fix patterns
- `abel_edge/plugins/AGENTS.md` — plugin isolation rules

### Constraints (enforced by `make test`)
See `CLAUDE.md`. Key: strategies.yaml is single source of truth, no file >400 lines,
strategies/ standalone, AGENTS.md at every subsystem.
