# Plugins Subsystem

Optional integrations. Removing this entire directory must not break anything.
`TestPluginsOptional` enforces this mechanically.

## I want to...

### Use Abel causal discovery
1. If your workflow relies on external skills, install `causal-abel` with `npx --yes skills add https://github.com/Abel-ai-causality/Abel-skills/tree/main/skills --skill causal-abel -y`
2. Optionally install it globally with `-g` instead; `abel-edge` also checks known OpenCode/Codex global skill roots for the shared `.env.skill`
3. Reuse the same `causal-abel` OAuth instead of reauthorizing in `abel-edge` when possible
4. Standalone fallback: run `abel-edge login`
   Agent-friendly form: `abel-edge login --json --no-browser`
   This emits a JSON handoff event before polling for completion and prints the authorization URL to `stderr` immediately.
5. Otherwise set `ABEL_API_KEY` or `CAP_API_KEY` in your environment or `.env`
6. Run: `abel-edge discover <TICKER>`
7. Use `--mode parents` or `--mode mb` depending on the discovery need
8. Copy the output YAML into your `strategies.yaml`
9. No API key? Fill `parents:` manually — framework works identically

### Align Abel price APIs
- Default real-price source is Abel market data
- See `docs/abel-price-api.md` for the request/response contract
- Abel currently uses the prod stack for both graph discovery and market data
- Login endpoint base: `https://api.abel.ai/echo`
- CAP endpoint: `POST https://cap.abel.ai/api/cap`
- Market endpoint: `POST https://cap.abel.ai/api/market/day_bar`
- Override the auth base with `ABEL_AUTH_BASE_URL`
- Override the CAP base with `ABEL_CAP_BASE_URL`

### Understand plugin isolation
- Framework uses `try/except ImportError` to detect plugins, not registry
- No plugin code is imported at framework startup
- Core tests pass with `plugins/` directory deleted

### Build a new plugin (future)
- Create `abel_edge/plugins/<name>/` directory
- Expose capabilities via top-level functions
- Framework discovers via `try/except` import in `abel_edge/cli.py`
- No registry pattern until second plugin exists (YAGNI)
