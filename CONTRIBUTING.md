# Contributing to causal-edge

## Quick Start

```bash
git clone https://github.com/Abel-ai-causality/causal-edge.git
cd causal-edge
pip install -e ".[dev]"
make test    # all tests must pass
```

## Development Workflow

1. Fork the repo
2. Create a feature branch: `git checkout -b my-feature`
3. Make changes
4. Run tests: `make test`
5. Format: `make fmt`
6. Lint: `make check`
7. Commit and push
8. Open a Pull Request

## Rules

These are enforced by structural tests — PRs that violate them will fail CI:

- **No file > 400 lines** — split at 350
- **strategies.yaml is the single source of truth** — no hardcoded strategy names
- **Components are pure functions** — data in, string out, no side effects
- **strategies/ never imports causal_edge/** (except engine base ABC)
- **AGENTS.md at every subsystem** with "I want to..." decision tree
- **No hardcoded paths or secrets** in source
- **Structural test assertions include Fix: instructions**

## Adding a Strategy

See [docs/add-strategy.md](docs/add-strategy.md).

## Code Style

- Formatter: `ruff format`
- Linter: `ruff check`
- All public functions have docstrings
- Chart functions return JSON strings

## Tests

```bash
make test        # all tests
make test-struct # structural tests only
make test-valid  # validation engine tests
make test-cli    # CLI tests
```

## Questions?

Open an issue on GitHub.
