# Plugins Subsystem

Optional integrations. Removing this entire directory must not break anything.
`TestPluginsOptional` enforces this mechanically.

## I want to...

### Use Abel causal discovery
1. Run: `causal-edge login`
   Agent-friendly form: `causal-edge login --json --no-browser`
   This emits a JSON handoff event before polling for completion.
2. Run: `causal-edge discover <TICKER>`
3. Use `--mode parents` or `--mode mb` depending on the discovery need
4. Copy the output YAML into your `strategies.yaml`
5. No API key? Fill `parents:` manually — framework works identically

### Align Abel price APIs
- Default real-price source is Abel market data
- See `docs/abel-price-api.md` for the request/response contract
- Abel currently uses the prod stack for both graph discovery and market data
- Login endpoint base: `https://api.abel.ai/echo`
- CAP endpoint: `POST https://cap.abel.ai/api/cap`
- Market endpoint: `POST https://cap.abel.ai/api/market/day_bar`
- Override the auth base with `ABEL_AUTH_BASE_URL`
- Override the CAP base with `ABEL_CAP_BASE_URL`

## Abel-Pro Mapping

- Abel-edge worktree for the Abel-Pro integration: `D:\codes\open_source\causal-edge\.tree\abel-pro-demo`
- Abel-edge branch for that worktree: `abel-pro-demo`

### Understand plugin isolation
- Framework uses `try/except ImportError` to detect plugins, not registry
- No plugin code is imported at framework startup
- Core tests pass with `plugins/` directory deleted

### Build a new plugin (future)
- Create `causal_edge/plugins/<name>/` directory
- Expose capabilities via top-level functions
- Framework discovers via `try/except` import in `causal_edge/cli.py`
- No registry pattern until second plugin exists (YAGNI)
