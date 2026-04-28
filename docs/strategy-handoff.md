# Strategy Handoff Contract

## Purpose

`abel-edge` owns the strategy-compliance decision. External orchestrators such as
`Abel-alpha` may package that decision for downstream workflow handoff, but the
handoff itself must conform to the edge contract or `abel-edge` rejects it.

Validate a handoff with:

```bash
abel-edge validate-handoff path/to/edge-handoff.json
```

Generate an edge-owned handoff while evaluating a strategy with:

```bash
abel-edge evaluate \
  --workdir strategies/my_strategy \
  --output-json edge-result.json \
  --output-md edge-validation.md \
  --output-handoff edge-handoff.json
```

## Contract

The handoff must be a JSON object with exactly these fields:

```json
{
  "contract": "abel-edge.strategy-handoff/v1",
  "strategy_path": "../engine.py",
  "verdict": "PASS",
  "profile": "equity_daily",
  "blocking_failures": [],
  "edge_result_path": "edge-result.json",
  "edge_report_path": "edge-validation.md"
}
```

Rules:

- `contract` must exactly equal `abel-edge.strategy-handoff/v1`
- `verdict` must be one of `PASS`, `FAIL`, `ERROR`
- `blocking_failures` must be a list of strings
- all paths must be relative to the handoff file
- `strategy_path`, `edge_result_path`, and `edge_report_path` must exist
- `verdict`, `profile`, and `blocking_failures` must exactly match the referenced
  `edge_result_path` payload

## Rejection Behavior

`abel-edge validate-handoff` rejects a handoff when any of the following is true:

- missing required fields
- unknown extra fields
- wrong contract version
- invalid field types
- absolute or missing paths
- unreadable JSON
- mismatch between the handoff summary and the referenced edge result

The command prints every rejection reason so the upstream orchestrator can repair the
handoff rather than guessing.
