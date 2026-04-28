# Agent Developer Guide

How AI agents operate the abel-edge framework.

## The Core Idea

Every subsystem has an `AGENTS.md` file with an "I want to..." decision tree. An agent reads this file and knows exactly what to do — which files to touch, which commands to run, which tests verify the work.

This is not documentation. It is an **executable architecture spec**.

## How Agents Navigate

```
AGENTS.md (root)
├── "Add a strategy"        → strategies/AGENTS.md
├── "Fix a validation"      → abel_edge/validation/AGENTS.md
├── "Add a dashboard chart"  → abel_edge/dashboard/AGENTS.md
└── "Use Abel discovery"     → abel_edge/plugins/AGENTS.md
```

Each subsystem AGENTS.md provides:
1. **Task-based routing** — "I want to X" → numbered steps
2. **File paths** — every referenced file exists (enforced by test)
3. **Verification** — `make test` or specific test class to run
4. **Debug routes** — "this failed, here's why and how to fix"

## Structural Tests = Guardrails

15 structural tests enforce architecture mechanically. When an agent makes a mistake, the test tells it exactly what to fix:

```
FAIL: Strategy 'my_strategy' is missing 'color'.
Fix: Add 'color' to the strategy entry in strategies.yaml.
See: docs/add-strategy.md
```

Every assertion includes a `Fix:` instruction. Agents don't need to guess.

## Key Constraints

These are enforced by `tests/test_structure.py`, not by convention:

| Constraint | Test | What happens if violated |
|-----------|------|------------------------|
| No file > 400 lines | `TestFileSizeLimit` | Fail with file list + "split" instruction |
| AGENTS.md at every subsystem | `TestSubsystemAgentsMd` | Fail with missing path |
| AGENTS.md has decision tree | `TestAgentsMdHasDecisionTree` | Fail with "add I want to..." |
| strategies/ standalone | `TestStrategiesStandalone` | Fail with bad import list |
| No hardcoded paths | `TestNoHardcodedPaths` | Fail with file:line |
| No secrets in source | `TestNoSecrets` | Fail with file path |

## Workflow for Agent Contributors

```
1. Read AGENTS.md → find the right subsystem
2. Read subsystem AGENTS.md → follow the steps
3. make test → verify constraints hold
4. Commit
```

The structural tests catch architectural violations immediately. No human review needed for constraint enforcement — it's mechanical.

## Building on This Pattern

To add the agent-native pattern to your own project:

1. Create `AGENTS.md` at project root with "I want to..." routing
2. Create subsystem `AGENTS.md` files with task-based steps
3. Write structural tests that enforce your constraints
4. Include `Fix:` instructions in every test assertion
5. Use `strategies.yaml` pattern — data as config, not code

The abel-edge framework demonstrates this pattern end-to-end. Study `tests/test_structure.py` for the enforcement approach.
