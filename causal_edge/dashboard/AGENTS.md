# Dashboard Subsystem

Template-driven HTML pages. Strategies come from `strategies.yaml`, not hardcoded.

## I want to...

### Add a chart component
1. Add pure function to `causal_edge/dashboard/components.py`: data in -> JSON string out
2. Use `_chart_to_json(fig)` to convert Plotly figures
3. Register in `causal_edge/dashboard/generator.py` env.globals
4. `make test` — `TestComponentsRegistered` verifies registration

### Modify a template
Templates are in `causal_edge/dashboard/templates/`. Rules:
- Zero Python logic — only Jinja2 loops/conditionals + component function calls
- One template renders N strategies — never copy-paste per strategy
- All strategy data comes from `strategies.yaml`, not hardcoded

### Debug "component not registered"
1. `make test` — `TestComponentsRegistered` shows which function is unregistered
2. Either add it to `env.globals` dict in `causal_edge/dashboard/generator.py`
3. Or prefix with `_` to mark it private (won't be checked)

## Key Files
- `generator.py` — page generators: dashboard, signal demo, tracking
- `components.py` — stateless Plotly chart builders (pure functions)
- `_helpers.py` — formatting utilities (pnl%, dollar amounts)
- `server.py` — HTTP server for serving generated dashboard
- `templates/` — Jinja2 templates for dashboard, signal demo, tracking
