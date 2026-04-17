"""Command-line interface for causal-edge."""

from __future__ import annotations

import json
from pathlib import Path

import click

from causal_edge import __version__
from causal_edge.cli_support import build_bars_loader
from causal_edge.research.cli import debug_evaluate as debug_evaluate_command
from causal_edge.research.cli import evaluate as evaluate_command
from causal_edge.research.cli import verify_data as verify_data_command
from causal_edge.research.cli import validate_handoff as validate_handoff_command

CONFIG_OPTION_HELP = (
    "Config file path (defaults to strategies.local.yaml if present, else strategies.yaml)"
)


def _get_version() -> str:
    """Return the framework version from the package source."""
    return __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=_get_version(), prog_name="causal-edge")
def main():
    """causal-edge: Agent-native quant framework."""


main.add_command(evaluate_command)
main.add_command(debug_evaluate_command)
main.add_command(verify_data_command)
main.add_command(validate_handoff_command)


@main.command("version")
def version():
    """Show causal-edge version."""
    click.echo(f"causal-edge, version {_get_version()}")


@main.command()
@click.argument("name")
def init(name):
    """Scaffold a new causal-edge project."""
    from causal_edge.scaffold import scaffold_project

    try:
        root = scaffold_project(name)
    except FileExistsError as e:
        raise click.ClickException(str(e))

    click.echo(f"Created {root}/")
    click.echo(f"  strategies.yaml          (3 demo strategies)")
    click.echo(f"  strategies/sma_crossover (simple SMA)")
    click.echo(f"  strategies/momentum_ml   (walk-forward GBDT)")
    click.echo(f"  strategies/causal_demo   (Abel causal graph voting)")
    click.echo(f"  CLAUDE.md + AGENTS.md    (agent harness)")
    click.echo(f"  .env.example             (Abel API key, optional)")
    click.echo(f"")
    click.echo(f"Next:")
    click.echo(f"  cd {name}")
    click.echo(f"  causal-edge run")
    click.echo(f"  causal-edge dashboard")
    click.echo(f"  causal-edge validate")


@main.command()
@click.option(
    "--env-path", default=".env", show_default=True, help="File path for storing ABEL_API_KEY"
)
@click.option(
    "--no-browser", is_flag=True, help="Print the authorization URL without opening a browser"
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit newline-delimited JSON events and the final result",
)
@click.option("--print-token", is_flag=True, help="Include the resolved API key in command output")
@click.option("--force", is_flag=True, help="Ignore any existing ABEL_API_KEY and run OAuth again")
@click.option(
    "--timeout",
    default=300,
    show_default=True,
    type=click.IntRange(min=1),
    help="Authorization timeout in seconds",
)
def login(env_path, no_browser, json_output, print_token, force, timeout):
    """Run explicit Abel OAuth and optionally persist the API key to .env."""
    try:
        from causal_edge.plugins.abel.auth import login_with_oauth
    except ImportError:
        raise click.ClickException("Abel plugin not installed. See: causal_edge/plugins/AGENTS.md")

    def _notify(message: str) -> None:
        click.echo(message, err=True)

    def _emit_handoff(payload: dict[str, object]) -> None:
        if json_output:
            click.echo(json.dumps(payload, sort_keys=True))

    try:
        result = login_with_oauth(
            env_path=env_path,
            open_browser=not no_browser,
            timeout_seconds=timeout,
            force=force,
            notify=None if json_output else _notify,
            on_handoff=_emit_handoff if json_output else None,
        )
    except Exception as e:
        raise click.ClickException(str(e))

    output = dict(result)
    if not print_token:
        output.pop("api_key", None)

    if json_output:
        click.echo(json.dumps(output, sort_keys=True))
        return

    if result["status"] == "already_configured":
        click.echo("Abel API key already configured.")
        if print_token:
            click.echo(output["api_key"])
        return

    click.echo(f"Abel API key saved to {env_path}.")
    if print_token:
        click.echo(output["api_key"])


@main.command()
@click.option("--strategy", default=None, help="Run a specific strategy by ID")
@click.option("--config", default=None, help=CONFIG_OPTION_HELP)
def run(strategy, config):
    """Run strategies and write trade logs."""
    from causal_edge.config import load_config
    from causal_edge.engine.trader import run_all

    cfg = load_config(config)
    if not cfg["strategies"]:
        click.echo("No strategies configured. Add strategies to strategies.yaml.")
        return

    click.echo(f"Running {len(cfg['strategies'])} strategies...")
    results = run_all(cfg, strategy_id=strategy)
    click.echo(f"Done. {len(results)} strategies executed.")


@main.command("paper")
@click.option("--strategy", default=None, help="Paper-trade a specific strategy by ID")
@click.option("--config", default=None, help=CONFIG_OPTION_HELP)
@click.option("--as-of", default=None, help="Only process bars up to this timestamp")
def paper(strategy, config, as_of):
    """Append live paper-trading rows using the latest closed bars."""
    from causal_edge.config import load_config
    from causal_edge.engine.trader import paper_run_all

    cfg = load_config(config)
    if not cfg["strategies"]:
        click.echo("No strategies configured. Add strategies to strategies.yaml.")
        return

    click.echo(f"Paper trading {len(cfg['strategies'])} strategies...")
    results = paper_run_all(cfg, strategy_id=strategy, as_of=as_of)
    click.echo(f"Done. {len(results)} strategies processed.")


@main.command()
@click.option("--config", default=None, help=CONFIG_OPTION_HELP)
@click.option("--output", default="dashboard.html", help="Output HTML path")
def dashboard(config, output):
    """Generate dashboard HTML."""
    from causal_edge.dashboard.generator import generate
    from causal_edge.config import load_config

    cfg = load_config(config)
    bars_loader = build_bars_loader(cfg)
    generate(config, output, bars_loader=bars_loader)
    click.echo(f"Dashboard generated: {output}")


@main.command("signal-demo")
@click.option("--config", default=None, help=CONFIG_OPTION_HELP)
@click.option("--strategy", required=True, help="Render a specific strategy signal demo page")
@click.option("--output", default="signal-demo.html", help="Output HTML path")
def signal_demo(config, strategy, output):
    """Generate a single-strategy Signal Demo page."""
    from causal_edge.dashboard.generator import generate_signal_demo
    from causal_edge.config import load_config

    cfg = load_config(config)
    bars_loader = build_bars_loader(cfg)

    try:
        generate_signal_demo(config, output, strategy_id=strategy, bars_loader=bars_loader)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"Signal demo generated: {output}")


@main.command("tracking")
@click.option("--config", default=None, help=CONFIG_OPTION_HELP)
@click.option("--strategy", required=True, help="Render a specific strategy tracking page")
@click.option("--output", default="tracking.html", help="Output HTML path")
def tracking(config, strategy, output):
    """Generate tracking HTML for a specific strategy."""
    from causal_edge.dashboard.generator import generate_tracking_page
    from causal_edge.config import load_config

    cfg = load_config(config)
    bars_loader = build_bars_loader(cfg)

    try:
        generate_tracking_page(config, output, strategy_id=strategy, bars_loader=bars_loader)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"Tracking page generated: {output}")


@main.command()
@click.option("--strategy", default=None, help="Validate a specific strategy by ID")
@click.option("--verbose", is_flag=True, help="Show detailed failure info")
@click.option(
    "--csv", "csv_path", default=None, help="Validate a standalone CSV (date,pnl columns)"
)
@click.option(
    "--dsr-trials",
    type=int,
    default=None,
    help="Declared strategy exploration count used by DSR (overrides profile default)",
)
@click.option("--export", "export_path", default=None, help="Export report to file")
@click.option("--config", default=None, help=CONFIG_OPTION_HELP)
def validate(strategy, verbose, csv_path, dsr_trials, export_path, config):
    """Run Abel Proof validation on strategies."""
    import io
    import sys

    from causal_edge.validation.gate import validate_strategy, print_validation_report

    results = {}
    old_stdout = None
    capture = None

    if csv_path:
        # Quick path: validate a standalone CSV without strategies.yaml
        if not Path(csv_path).exists():
            raise click.ClickException(f"CSV not found: {csv_path}")
        result = validate_strategy(csv_path, dsr_trials=dsr_trials)
        results[Path(csv_path).stem] = result
    else:
        from causal_edge.config import load_config

        cfg = load_config(config)
        strategies_list = cfg["strategies"]
        if strategy:
            strategies_list = [s for s in strategies_list if s["id"] == strategy]

        if not strategies_list:
            click.echo("No strategies to validate.")
            return

        for s_cfg in strategies_list:
            sid = s_cfg["id"]
            log_path = s_cfg.get("trade_log", "")
            if not Path(log_path).exists():
                results[sid] = {
                    "verdict": "SKIP",
                    "score": "0/0",
                    "failures": [f"Trade log not found: {log_path}. Run 'causal-edge run' first."],
                    "metrics": {},
                    "triangle": {"ratio": 0, "rank": 0, "shape": 0},
                    "profile": "unknown",
                }
                continue
            results[sid] = validate_strategy(log_path, dsr_trials=dsr_trials)

    # Capture output for --export
    if export_path:
        old_stdout = sys.stdout
        sys.stdout = capture = io.StringIO()

    try:
        print_validation_report(results)

        if verbose:
            print()
            for sid, r in results.items():
                if r.get("metrics"):
                    print(f"  {sid} metrics:")
                    m = r["metrics"]
                    for key in (
                        "sharpe",
                        "lo_adjusted",
                        "sortino",
                        "total_return",
                        "max_dd",
                        "calmar",
                        "dsr",
                        "dsr_trials_used",
                        "omega",
                        "position_ic",
                        "position_hit_rate",
                    ):
                        if key.startswith("position_") and not m.get(
                            "position_ic_applicable", False
                        ):
                            continue
                        if key in m:
                            print(f"    {key:20s} {m[key]:.4f}")
                    if "yearly_sharpes" in m:
                        print(f"    yearly_sharpes:")
                        for yr, sh in sorted(m["yearly_sharpes"].items()):
                            print(f"      {yr}: {sh:.2f}")
    finally:
        if export_path:
            sys.stdout = old_stdout
            report_text = capture.getvalue() if capture is not None else ""
            click.echo(report_text, nl=False)  # also print to terminal
            Path(export_path).write_text(report_text, encoding="utf-8")
            click.echo(f"\n  Report exported to {export_path}")

    all_pass = all(r["verdict"] in ("PASS", "SKIP") for r in results.values())
    sys.exit(0 if all_pass else 1)


@main.command()
@click.argument("ticker")
@click.option(
    "--mode",
    type=click.Choice(["parents", "mb", "all"]),
    default="parents",
    show_default=True,
)
@click.option(
    "--limit",
    default=10,
    show_default=True,
    type=int,
    help="Maximum nodes to return (hard cap 20)",
)
@click.option("--json", "json_output", is_flag=True, help="Emit structured JSON output")
def discover(ticker, mode, limit, json_output):
    """Discover causal graph nodes for an asset via Abel API."""
    try:
        from causal_edge.plugins.abel.discover import (
            discover_graph_nodes,
            discover_graph_payload,
        )
    except ImportError:
        raise click.ClickException("Abel plugin not installed. See: causal_edge/plugins/AGENTS.md")
    try:
        if json_output:
            output = discover_graph_payload(ticker, mode=mode, limit=limit)
            click.echo(json.dumps(output, indent=2, sort_keys=True))
            return
        output = discover_graph_nodes(ticker, mode=mode, limit=limit)
    except Exception as e:
        raise click.ClickException(str(e))
    click.echo(output)


@main.command()
@click.option("--config", default=None, help=CONFIG_OPTION_HELP)
def status(config):
    """Show strategy status summary."""
    from causal_edge.config import load_config

    cfg = load_config(config)
    click.echo(f"Strategies: {len(cfg['strategies'])}")
    for s in cfg["strategies"]:
        click.echo(f"  {s['name']:20s}  {s['asset']:6s}  {s.get('badge', '?')}")


if __name__ == "__main__":
    main()
