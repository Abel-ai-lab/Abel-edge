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
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# bash/zsh: source .venv/bin/activate
python -m pip install --upgrade pip
pip install git+https://github.com/Abel-ai-causality/Abel-edge.git
causal-edge init my-portfolio
cd my-portfolio

# If you want live Abel discovery, authenticate before discover.
causal-edge login

causal-edge run
causal-edge validate
```

If the `git+https` install path is unstable in your network environment, use the same public repo via zip:

```bash
pip install https://github.com/Abel-ai-causality/Abel-edge/archive/refs/heads/main.zip
```

For live discovery, authenticate once before `causal-edge discover <TICKER>`:

```bash
causal-edge login
```

Or validate any existing backtest:

```bash
causal-edge validate --csv my_backtest.csv
```

Abel discovery and market data default to the public CAP base: `https://cap.abel.ai/api`

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
causal-edge login                    # run explicit Abel OAuth and persist ABEL_API_KEY
causal-edge run [--strategy ID]      # run strategies, write backtest trade logs
causal-edge paper [--strategy ID]    # append latest live paper-trading rows
causal-edge dashboard                # generate dark-theme dashboard HTML
causal-edge signal-demo --strategy ethusd_causal --output signal-demo-ethusd.html
                                # generate a single-strategy Signal Demo page
causal-edge tracking --strategy ethusd_causal --output signal-track-ethusd.html
                                # generate a separate tracking page for live rows
causal-edge evaluate --workdir strategies/my_strategy
                                # emit raw validation facts for one strategy workspace
causal-edge evaluate --workdir strategies/my_strategy --start 2020-01-01
                                # pin the requested backtest start for upstream orchestration
causal-edge evaluate --workdir strategies/my_strategy --output-json edge-result.json --output-md edge-validation.md
                                # persist raw JSON + markdown facts for an upstream orchestration layer
causal-edge validate [--verbose]     # Abel Proof validation (audited live gate contract)
causal-edge validate --csv file.csv  # validate any backtest CSV directly
causal-edge validate --export r.txt  # export report for sharing
causal-edge discover <TICKER>        # find causal parents (Abel API key)
```

Real-price strategies default to Abel price APIs. Override per strategy with
`price_data.source: csv` to load local bar data instead. Configure Abel access
with `ABEL_API_KEY` and optionally `ABEL_CAP_BASE_URL`. If you do not already
have an API key, install `causal-abel` and complete its OAuth flow before
running `causal-edge discover <TICKER>` or any workflow that triggers live Abel
discovery:

```bash
npx --yes skills add https://github.com/Abel-ai-causality/Abel-skills/tree/main/skills --skill causal-abel -y
```

For a global install instead of a project-local skill copy, use:

```bash
npx --yes skills add https://github.com/Abel-ai-causality/Abel-skills/tree/main/skills --skill causal-abel -g -y
```

`causal-edge` prefers reusing an existing `causal-abel` auth file before asking you to authorize
again. Use `causal-edge login` when you want a standalone fallback that stores `ABEL_API_KEY`
directly for the current project, or set `ABEL_API_KEY` directly. Override endpoints with
`ABEL_CAP_BASE_URL` and `ABEL_AUTH_BASE_URL` when needed.

After `causal-abel` OAuth succeeds, `causal-edge` checks the current project `.env`,
`ABEL_AUTH_ENV_FILE`, the local `.agents/skills/causal-abel/.env.skill` fallback, and known global
skill installs such as `~/.config/opencode/skills/causal-abel/.env.skill` or
`~/.codex/skills/causal-abel/.env.skill` before failing for a missing key. That lets
agent-driven installs reuse the `causal-abel` auth file without copying the key into each
workspace.

If discovery still reports a missing key after `causal-abel` is installed, run:

```bash
python <causal-abel-skill-root>/scripts/cap_probe.py auth-status --compact
```

For example, the skill root might be one of these locations:

- project-local: `.agents/skills/causal-abel`
- OpenCode global: `~/.config/opencode/skills/causal-abel`
- Codex global: `~/.codex/skills/causal-abel`

If your auth file lives outside those paths, point `causal-edge` at it with `ABEL_AUTH_ENV_FILE`.

For agent-driven setups, `causal-edge login --json --no-browser` emits a JSON
handoff event first, then a final JSON result after authorization completes.

When a strategy declares `paper_log`, backtests stay in `trade_log` and live paper rows
append to `paper_log`. If `paper_log` is omitted, causal-edge falls back to the legacy
single-log format and reads `source=live` rows as paper-trading data.

If both `strategies.local.yaml` and `strategies.yaml` exist, CLI commands now prefer
`strategies.local.yaml` automatically. Use `--config` to point at any explicit file.

`causal-edge evaluate` uses the same audited validation contract as the main CLI and emits
only raw execution facts: verdict, score, metrics, triangle, failures, K, and the requested/effective
evaluation window. It does not own
exploration-session structure, branch organization, or narrative summaries. Those belong to
the upstream orchestration layer such as `Abel-alpha`.

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
- [Look-Ahead Rules](causal_edge/validation/look_ahead_rules.md) — semantic review checklist for leaked features
- [Why Causal?](docs/why-causal.md) — Pearl, DGP, intervention invariance
- [Agent Developer Guide](docs/harness-guide.md) — how agents operate this framework
- [Contributing](CONTRIBUTING.md) — how to contribute

## License

MIT
