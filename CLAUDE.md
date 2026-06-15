# CLAUDE.md — abel-edge development harness

## Constraints (enforced by tests/test_structure.py)

1. **No file > 400 lines.** Split at 350. Enforced by `TestFileSizeLimit`.
2. **strategies/ never imports abel_edge/ internals** (except base.py ABC). Enforced by `TestStrategiesStandalone`.
3. **strategies.yaml is the single source of truth.** Never hardcode strategy names, colors, or tickers.
4. **Every subsystem has AGENTS.md** with "I want to..." decision tree. Enforced by `TestSubsystemAgentsMd` + `TestAgentsMdHasDecisionTree`.
5. **All strategy engines must be importable.** Enforced by `TestEngineModuleImportable`.
6. **No hardcoded absolute paths. No secrets in source.** Enforced by `TestNoHardcodedPaths` + `TestNoSecrets`.
7. **Structural test failure messages include actionable fix instructions.** Pattern: assert message + "Fix:" line.
8. **AGENTS.md size budget.** Root <=80 lines, subsystem <=60 lines. Enforced by `TestAgentsMdSizeBudget`.

## Style

- Formatter: `ruff format` (run via `make fmt`)
- Linter: `ruff check` (run via `make check`)
- All public functions have docstrings

## How to run

```bash
make test        # all tests (must pass before any commit)
make lint        # structural tests only
make test-struct # same as lint
make test-valid  # validation engine tests
make test-cli    # CLI tests
make fmt         # auto-format with ruff
make check       # lint with ruff
make verify-1    # Phase 1 completion check
```

## Navigation

- `AGENTS.md` — project entry point for agents (read first)
- `ARCHITECTURE.md` — dependency direction diagram
- `docs/add-strategy.md` — step-by-step strategy addition guide
