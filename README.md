# abel-edge

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

Install the published package:

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# bash/zsh: source .venv/bin/activate
python -m pip install --upgrade pip
pip install abel-edge
abel-edge init my-portfolio
cd my-portfolio

# If you want live Abel discovery, authenticate before discover.
abel-edge login

abel-edge run
abel-edge validate
```

For local development from source:

```bash
git clone https://github.com/Abel-ai-causality/Abel-edge.git
cd Abel-edge
pip install -e ".[dev]"
```

If you need to install from the repository before a PyPI release is available:

```bash
pip install git+https://github.com/Abel-ai-causality/Abel-edge.git
```

If the `git+https` install path is unstable in your network environment, use the same public repo via zip:

```bash
pip install https://github.com/Abel-ai-causality/Abel-edge/archive/refs/heads/main.zip
```

For live discovery, authenticate once before `abel-edge discover <TICKER>`:

```bash
abel-edge login
```

Or validate any existing backtest:

```bash
abel-edge validate --csv my_backtest.csv
```

Abel discovery and market data default to the public CAP base: `https://cap.abel.ai/api`

## Why Causal?

Correlation is a property of *data*. Causation is a property of the *data generating process*.

When regimes change (bull→bear, policy shift, crisis):
- Correlations break → correlation-based signals die
- Causal links persist → causal signals survive

This is Pearl's definition: a causal relationship remains invariant under intervention. The causal demo bundles a real causal graph from [Abel](https://abel.ai) — 5 equity parents and 3 children of TONUSD. For live discovery: `abel-edge discover <TICKER>`.

## Why Agent-Native?

Every other quant framework is designed for humans to read docs, write code, run commands.

abel-edge is designed for agents to read `CAPABILITY.md` and operate autonomously:

| What | Human framework | abel-edge |
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

| Strategy | What it is | Validation note | What it teaches |
|----------|-----------|-----------------|-----------------|
| `sma_crossover` | DecisionContext crossover on local sample bars | Uses the audited live `validate` contract | primary target reads plus next-position intent |
| `momentum_ml` | DecisionContext walk-forward GBDT demo | Uses the audited live `validate` contract | vectorized features and walk-forward training on the target feed |
| `feed_overlay_demo` | Target plus declared auxiliary feeds | Uses the audited live `validate` contract | `ctx.feed(name)` and as-of feed reads |

The repo also includes `examples/causal_demo/` as an optional graph-shaped
example for named driver feeds.

## Commands

```bash
abel-edge init <name>              # scaffold standalone project with 3 local sample-data demo strategies
abel-edge login                    # run explicit Abel OAuth and persist ABEL_API_KEY
abel-edge run [--strategy ID]      # run strategies, write backtest trade logs
abel-edge paper [--strategy ID]    # append latest live paper-trading rows
abel-edge dashboard                # generate dark-theme dashboard HTML
abel-edge signal-demo --strategy ethusd_causal --output signal-demo-ethusd.html
                                # generate a single-strategy Signal Demo page
abel-edge tracking --strategy ethusd_causal --output signal-track-ethusd.html
                                # generate a separate tracking page for live rows
abel-edge evaluate --workdir strategies/my_strategy
                                # emit raw validation facts for one strategy workspace
abel-edge evaluate --workdir strategies/my_strategy --start 2020-01-01
                                # pin the requested backtest start for upstream orchestration
abel-edge evaluate --workdir strategies/my_strategy --output-json edge-result.json --output-md edge-validation.md
                                # persist raw JSON + markdown facts for an upstream orchestration layer
abel-edge evaluate --workdir strategies/my_strategy --output-json edge-result.json --output-md edge-validation.md --output-handoff edge-handoff.json
                                # emit an edge-owned handoff that upstream tools must preserve exactly
abel-edge validate-handoff edge-handoff.json
                                # reject invalid upstream handoffs with explicit reasons
abel-edge validate [--verbose]     # Abel Proof validation (audited live gate contract)
abel-edge validate --csv file.csv  # validate any backtest CSV directly
abel-edge validate --export r.txt  # export report for sharing
abel-edge discover <TICKER>        # find causal parents (Abel API key)
```

## Public Surface

- PyPI distribution: `abel-edge`
- Python import package: `abel_edge`
- CLI entry point: `abel-edge`
- Supported Python versions: 3.11, 3.12, and 3.13
- Runtime facts contract: `abel-edge.runtime-facts/v1`
- Strategy handoff contract: `abel-edge.strategy-handoff/v1`

The bundled `init` scaffold is a standalone demo project, not an Abel-alpha
branch workspace. For real-data branch research inside an Abel-alpha workspace,
stay on the `abel-alpha init-session -> init-branch -> prepare-branch` path.

New strategies should author against `compute_decisions(self, ctx)` and
`DecisionContext`, not against the legacy tuple-return signal contract.

Real-price strategies default to Abel price APIs. Override per strategy with
`price_data.adapter: csv` for local bar files, or register a project-local
adapter via `settings.data_adapters.imports` when your project owns a custom
data backend. The framework still normalizes timestamps, enforces feed/runtime
contracts, and compiles next-position intent after adapter loading. Configure
Abel access with `ABEL_API_KEY` and optionally `ABEL_CAP_BASE_URL`. If you do
not already have an API key, install `causal-abel` and complete its OAuth flow
before running `abel-edge discover <TICKER>` or any workflow that triggers
live Abel discovery:

```bash
npx --yes skills add https://github.com/Abel-ai-causality/Abel-skills/tree/main/skills --skill causal-abel -y
```

For a global install instead of a project-local skill copy, use:

```bash
npx --yes skills add https://github.com/Abel-ai-causality/Abel-skills/tree/main/skills --skill causal-abel -g -y
```

`abel-edge` prefers reusing an existing `causal-abel` auth file before asking you to authorize
again. Use `abel-edge login` when you want a standalone fallback that stores `ABEL_API_KEY`
directly for the current project, or set `ABEL_API_KEY` directly. Override endpoints with
`ABEL_CAP_BASE_URL` and `ABEL_AUTH_BASE_URL` when needed.

After `causal-abel` OAuth succeeds, `abel-edge` checks the current project `.env`,
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

If your auth file lives outside those paths, point `abel-edge` at it with `ABEL_AUTH_ENV_FILE`.

For agent-driven setups, `abel-edge login --json --no-browser` emits a JSON
handoff event first, then a final JSON result after authorization completes.
It also prints the authorization URL and periodic waiting updates to `stderr`
so agent runtimes can surface the browser handoff immediately instead of
appearing hung.

When a strategy declares `paper_log`, backtests stay in `trade_log` and live paper rows
append to `paper_log`. If `paper_log` is omitted, abel-edge falls back to the legacy
single-log format and reads `source=live` rows as paper-trading data.

If both `strategies.local.yaml` and `strategies.yaml` exist, CLI commands now prefer
`strategies.local.yaml` automatically. Use `--config` to point at any explicit file.

`abel-edge evaluate` uses the same audited validation contract as the main CLI and emits
raw execution facts: verdict, score, metrics, triangle, failures, K, profile, and the
requested/effective evaluation window. When asked, it also emits an edge-owned handoff JSON
for upstream orchestrators. `abel-edge validate-handoff` rejects any upstream handoff that
does not exactly match the published contract. `abel-edge` still does not own exploration-session
structure, branch organization, or narrative summaries; those belong to the upstream orchestration
layer such as `Abel-alpha`.

## Architecture

```
CAPABILITY.md          → agent reads this, gains full capability
AGENTS.md              → "use as tool" or "develop on this repo"
abel_edge/
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
- [`docs/strategy-handoff.md`](docs/strategy-handoff.md) — exact upstream handoff contract and rejection behavior
- [`docs/releasing.md`](docs/releasing.md) — maintainer release process for PyPI
- [Adding a Strategy](docs/add-strategy.md) — three paths: CSV / engine / causal
- [Look-Ahead Rules](abel_edge/validation/look_ahead_rules.md) — semantic review checklist for leaked features
- [Why Causal?](docs/why-causal.md) — Pearl, DGP, intervention invariance
- [Agent Developer Guide](docs/harness-guide.md) — how agents operate this framework
- [Contributing](CONTRIBUTING.md) — how to contribute
- [Support](SUPPORT.md) — usage questions and issue triage
- [Security](SECURITY.md) — private vulnerability reporting

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
