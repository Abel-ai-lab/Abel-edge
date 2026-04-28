# abel-edge

`abel-edge` is the open-source runtime layer for graph-grounded trading
research.

It gives a strategy project three things:

- a deterministic execution surface built around `DecisionContext`
- validation facts that make look-ahead, overfitting, and weak signal shape
  visible
- CLI and Python APIs that agents and humans can both use without relying on
  repository-local workspace conventions

The core idea is simple: causal structure can be a useful research prior, but
only runtime facts should decide whether a strategy run is valid evidence.

`abel-edge` is research software. It is not financial advice, a broker, or a
guarantee that a strategy will perform in live markets.

## Install

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# bash/zsh: source .venv/bin/activate

python -m pip install --upgrade pip
pip install abel-edge
```

Install from source for development:

```bash
git clone https://github.com/Abel-ai-causality/Abel-edge.git
cd Abel-edge
pip install -e ".[dev]"
```

If PyPI is not available in your environment, install from the public
repository:

```bash
pip install git+https://github.com/Abel-ai-causality/Abel-edge.git
```

## Quickstart

Create a standalone demo project, run the bundled strategies, and validate the
results:

```bash
abel-edge init my-portfolio
cd my-portfolio
abel-edge run
abel-edge validate
abel-edge dashboard
```

Validate an existing backtest CSV:

```bash
abel-edge validate --csv my_backtest.csv
```

Use live Abel discovery or Abel market data by authenticating once:

```bash
abel-edge login
abel-edge discover ETHUSD
```

Discovery and market data default to `https://cap.abel.ai/api`. You can set
`ABEL_API_KEY` directly, use `abel-edge login`, or point at an existing auth
file with `ABEL_AUTH_ENV_FILE`. Override endpoints with `ABEL_CAP_BASE_URL` and
`ABEL_AUTH_BASE_URL` when needed.

## Why This Exists

Most strategy tooling rewards the wrong thing first: a high backtest number.

`abel-edge` puts another layer in front of that number:

1. Was the strategy allowed to observe those inputs at decision time?
2. Did it produce next-position intent rather than same-bar leakage?
3. Did it read the auxiliary feeds it claimed to use?
4. Did validation pass for ratio, rank, and shape rather than one headline
   metric?

That distinction matters for agent-led research. A language model can generate
many variants quickly, so the runtime has to be strict about evidence quality
without telling the agent which strategy idea to try next.

## Causal Graphs As A Research Prior

Correlation is a property of observed data. Causal structure is a hypothesis
about the data-generating process.

In the Abel research stack, causal graph discovery is used as a search prior:
it helps narrow the first set of candidate inputs before strategy variants and
parameter refinements begin. `abel-edge` does not decide which graph path to
explore. It provides the runtime contracts and facts needed to check whether a
graph-supported strategy actually used the selected inputs legally.

The intended research priority is:

```text
causal graph structure first -> strategy variants second -> parameters last
```

Target-only work can still be useful as a control, diagnostic, or explicit
fallback. It just should not be mistaken for candidate causal evidence unless
the runtime facts support that label.

## Agent-Native Runtime Facts

`abel-edge` is designed for both people and AI coding agents.

The agent-facing promise is not that the framework chooses the next research
move. It is that the framework returns structured facts an agent can use:

- validation verdicts and metric failures
- look-ahead and semantic preflight diagnostics
- runtime read facts for target and auxiliary feeds
- effective evaluation windows
- exact handoff contracts for upstream orchestration layers

Agents can start from [`CAPABILITY.md`](CAPABILITY.md). Humans can use the same
CLI and Python APIs directly.

## Abel Edge And Abel Invest

`abel-edge` is the deterministic runtime and contract owner.

It owns:

- strategy execution
- `DecisionContext`
- feed and data contracts
- look-ahead checks
- metric validation
- dashboards
- raw runtime facts
- handoff validation

It does not own:

- research sessions
- branch organization
- evidence ledgers
- research journals
- narrative summaries
- recommendations about what strategy to try next

Those workflow concerns belong in an orchestration layer such as Abel Invest.
That separation is deliberate: the runtime should protect evidence quality,
while the research agent remains responsible for strategy judgment.

## Validation Model

The validation layer focuses on a metric triangle:

```text
               Lo-adjusted Sharpe
                    /      \
                   /        \
     Position-Return IC    Omega
```

Each side catches a different failure mode:

- **Lo-adjusted Sharpe** reduces reward for serial-correlation artifacts.
- **Position-Return IC** checks whether position sizing has rank relationship
  with subsequent returns.
- **Omega** makes clipping and asymmetric loss behavior harder to hide.

The goal is not to turn validation into an optimizer. The goal is to make weak
or suspicious evidence explicit before it becomes a research conclusion.

## Demo Strategies

The scaffolded project includes small demo strategies that run on local sample
data:

| Strategy | What it demonstrates |
| --- | --- |
| `sma_crossover` | Primary target reads and next-position intent |
| `momentum_ml` | Vectorized features and walk-forward ML |
| `feed_overlay_demo` | Declared auxiliary feeds and legal as-of reads |

The repository also includes `examples/causal_demo/` as a graph-shaped example
for named driver feeds.

## CLI

```bash
abel-edge version
abel-edge init <name>
abel-edge login
abel-edge discover <TICKER>

abel-edge run [--strategy ID]
abel-edge paper [--strategy ID]
abel-edge validate [--verbose]
abel-edge validate --csv file.csv
abel-edge dashboard

abel-edge signal-demo --strategy ethusd_causal --output signal-demo-ethusd.html
abel-edge tracking --strategy ethusd_causal --output signal-track-ethusd.html

abel-edge evaluate --workdir strategies/my_strategy
abel-edge evaluate --workdir strategies/my_strategy --start 2020-01-01
abel-edge evaluate --workdir strategies/my_strategy \
  --output-json edge-result.json \
  --output-md edge-validation.md \
  --output-handoff edge-handoff.json
abel-edge validate-handoff edge-handoff.json
```

## Python API

```python
from abel_edge.validation.gate import validate_strategy

result = validate_strategy("my_backtest.csv", positions_col="position")
print(result["verdict"])
print(result["metrics"])
```

New strategies should author against `compute_decisions(self, ctx)` and
`DecisionContext`, not the legacy tuple-return signal contract.

## Public Surface

- PyPI distribution: `abel-edge`
- Python import package: `abel_edge`
- CLI entry point: `abel-edge`
- Supported Python versions: 3.11, 3.12, and 3.13
- Runtime facts contract: `abel-edge.runtime-facts/v1`
- Strategy handoff contract: `abel-edge.strategy-handoff/v1`
- Cache environment variable: `ABEL_EDGE_CACHE_ROOT`

`abel-edge evaluate` emits raw execution facts: verdict, score, metrics,
triangle, failures, profile, K, runtime reads, and the requested/effective
evaluation window. When requested, it also emits a handoff JSON that
orchestration layers can preserve exactly. `abel-edge validate-handoff`
rejects handoffs that do not match the published contract.

## Data And Configuration

Real-price strategies default to Abel price APIs. For local or custom data:

- set `price_data.adapter: csv` for local bar files
- register project-local adapters with `settings.data_adapters.imports`
- use `--config` to point CLI commands at an explicit config file

If both `strategies.local.yaml` and `strategies.yaml` exist, CLI commands prefer
`strategies.local.yaml` automatically.

When a strategy declares `paper_log`, backtests stay in `trade_log` and live
paper rows append to `paper_log`. If `paper_log` is omitted, `abel-edge` falls
back to the single-log format and reads `source=live` rows as paper-trading
data.

## Project Layout

```text
CAPABILITY.md          agent-facing capability guide
AGENTS.md              contributor and agent operating notes
abel_edge/
  validation/          Abel Proof metrics and look-ahead checks
  engine/              StrategyEngine, DecisionContext, execution runtime
  dashboard/           Jinja2 and Plotly static dashboard generation
  plugins/             optional Abel discovery and price integrations
examples/
  sma_crossover/       sample target-only strategy
  momentum_ml/         sample walk-forward ML strategy
  causal_demo/         graph-shaped example with driver feeds
tests/                 runtime, validation, CLI, packaging, and dashboard tests
```

## Documentation

- [`CAPABILITY.md`](CAPABILITY.md) - agent capability guide
- [`docs/add-strategy.md`](docs/add-strategy.md) - CSV, engine, and graph-shaped
  strategy paths
- [`docs/why-causal.md`](docs/why-causal.md) - causal framing and intervention
  invariance
- [`docs/validation-audit-matrix.md`](docs/validation-audit-matrix.md) -
  validation timing and score contract
- [`docs/strategy-handoff.md`](docs/strategy-handoff.md) - exact handoff
  contract and rejection behavior
- [`abel_edge/validation/look_ahead_rules.md`](abel_edge/validation/look_ahead_rules.md) -
  semantic review checklist for leaked features
- [`docs/harness-guide.md`](docs/harness-guide.md) - agent operation guide
- [`docs/releasing.md`](docs/releasing.md) - maintainer release process for PyPI
- [`CONTRIBUTING.md`](CONTRIBUTING.md) - how to contribute
- [`SUPPORT.md`](SUPPORT.md) - usage questions and issue triage
- [`SECURITY.md`](SECURITY.md) - private vulnerability reporting

## Development

```bash
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m build --sdist --wheel
python -m twine check dist/*
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
