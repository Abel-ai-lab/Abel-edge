# causal-edge

**Two edges. One framework.**

**Causal edge** — correlations break when regimes change. Only causal structure survives. Three strategies, same audited validation contract, prove it:

```
Correlation (SMA)    →  Lo = -0.65   dead on arrival
ML (GBDT)            →  Lo = -0.27   still dead
Causal (Abel graph)  →  Lo = +0.55   alive
```

**Agent edge** — your AI agent reads one file, gains the full capability, and works autonomously. No tutorial, no hand-holding, no human in the loop:

```
Agent receives repo URL
  → reads CAPABILITY.md (90 lines)
  → installs, validates, diagnoses failures, applies fixes
  → reports: "your strategy improved under the same audited gate contract"
```

Neither edge exists alone. Causal without agents = a paper you read once. Agents without causal = fast noise. Together = **autonomous discovery of what actually works.**

> **Agents start here → [`CAPABILITY.md`](CAPABILITY.md)**

## 5 Minutes: See Both Edges

```bash
pip install git+https://github.com/Abel-ai-causality/causal-edge.git
causal-edge init my-portfolio
cd my-portfolio
causal-edge run && causal-edge validate
```

Or validate any existing backtest:

```bash
causal-edge validate --csv my_backtest.csv
```

## Why Causal?

Correlation is a property of *data*. Causation is a property of the *data generating process*.

When regimes change (bull→bear, policy shift, crisis):
- Correlations break → correlation-based signals die
- Causal links persist → causal signals survive

This is Pearl's definition: a causal relationship remains invariant under intervention. The causal demo bundles a real causal graph from [Abel](https://abel.ai) — 5 equity parents and 3 children of TONUSD. For live discovery: `causal-edge discover <TICKER>`.

## Why Agent-Native?

Every other quant framework is designed for humans to read docs, write code, run commands.

causal-edge is designed for agents to read `CAPABILITY.md` and operate autonomously:

| What | Human framework | causal-edge |
|------|----------------|-------------|
| Learn the tool | Read 50-page docs | Agent reads 1 file (CAPABILITY.md) |
| Validate a strategy | Configure, run, interpret | `validate_strategy(csv)` → structured result |
| Fix failures | Google the error, guess | Failure→fix table with copy-paste code |
| Iterate | Manual loop, hours | Autonomous loop: validate→fix→revalidate |
| Remember how | Bookmark, forget | Self-internalization (skill/memory/CLAUDE.md) |

## The Metric Triangle

Three leverage-invariant, mathematically orthogonal dimensions:

```
        Lo-adjusted Sharpe (ratio — optimized)
             /           \
       Position-Return IC (rank —  Omega (shape —
        guardrail)          guardrail)
```

No known transformation improves all three simultaneously except genuine signal improvement:
- **Clipping** inflates Sharpe but tanks Omega
- **Serial correlation** inflates Sharpe but Lo catches it
- **Concentration** boosts ratios but Position-Return IC drops

Verified across 38 controlled experiments.

## Demo Strategies

| Strategy | What it is | Validation note | What it proves |
|----------|-----------|-----------------|----------------|
| `sma_crossover` | Simple moving average | Uses the audited live `validate` contract (score denominator depends on applicable gates) | Random signals fail completely |
| `momentum_ml` | Walk-forward GBDT | Uses the audited live `validate` contract (score denominator depends on applicable gates) | ML on noise is still noise |
| `causal_demo` | Abel causal graph voting | Uses the audited live `validate` contract (score denominator depends on applicable gates) | Causal structure produces real edge |

## Commands

```bash
causal-edge init <name>              # scaffold project with 3 demo strategies
causal-edge run [--strategy ID]      # run strategies, write trade logs
causal-edge paper [--strategy ID]    # append latest live paper-trading rows
causal-edge dashboard                # generate dark-theme dashboard HTML
causal-edge dashboard --strategy ethusd_causal --output signal-demo-ethusd.html
                                # generate a single-strategy Signal Demo page
causal-edge tracking --strategy ethusd_causal --output signal-track-ethusd.html
                                # generate a separate tracking page for live rows
causal-edge validate [--verbose]     # Abel Proof validation (audited live gate contract)
causal-edge validate --csv file.csv  # validate any backtest CSV directly
causal-edge validate --export r.txt  # export report for sharing
causal-edge discover <TICKER>        # find causal parents (Abel API key)
```

Real-price strategies default to Abel price APIs. Override per strategy with
`price_data.source: csv` to load local bar data instead.

## Architecture

```
CAPABILITY.md          → agent reads this, gains full capability
AGENTS.md              → "use as tool" or "develop on this repo"
causal_edge/
  validation/          → Abel Proof metric triangle + audited live gate contract
  engine/              → StrategyEngine ABC + execution
  dashboard/           → Jinja2 + Plotly → static HTML
  plugins/             → optional (Abel causal discovery)
examples/
  sma_crossover/       → correlation demo (30 lines)
  momentum_ml/         → ML demo (80 lines)
  causal_demo/         → causal demo (100 lines + graph JSON)
tests/
  test_structure.py    → 15 structural tests enforce architecture
```

## Documentation

- [`CAPABILITY.md`](CAPABILITY.md) — agent capability acquisition (start here)
- [`docs/validation-audit-matrix.md`](docs/validation-audit-matrix.md) — long-lived validation timing/score contract and migration notes
- [Adding a Strategy](docs/add-strategy.md) — three paths: CSV / engine / causal
- [Why Causal?](docs/why-causal.md) — Pearl, DGP, intervention invariance
- [Agent Developer Guide](docs/harness-guide.md) — how agents operate this framework
- [Contributing](CONTRIBUTING.md) — how to contribute

## License

MIT
