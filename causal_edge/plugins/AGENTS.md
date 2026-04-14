# Plugins Subsystem

Optional integrations. Removing this entire directory must not break anything.
`TestPluginsOptional` enforces this mechanically.

## I want to...

### Use Abel causal discovery
1. Run: `causal-edge discover <TICKER>`
2. If you do not already have a key, install `causal-abel` with `npx --yes skills add https://github.com/Abel-ai-causality/Abel-skills/tree/main/skills --skill causal-abel -y`
3. Complete the `causal-abel` OAuth flow, or set `ABEL_API_KEY` / `CAP_API_KEY` in your environment or `.env`
4. Use `--mode parents` or `--mode mb` depending on the discovery need
5. Copy the output YAML into your `strategies.yaml`
6. No API key? Fill `parents:` manually — framework works identically

### Align Abel price APIs
- Default real-price source is Abel market data
- See `docs/abel-price-api.md` for the request/response contract
- Abel currently uses the prod stack for both graph discovery and market data
- CAP endpoint: `POST https://cap.abel.ai/api/cap`
- Market endpoint: `POST https://cap.abel.ai/api/market/day_bar`
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
