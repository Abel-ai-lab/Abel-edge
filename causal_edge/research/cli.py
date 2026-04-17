"""CLI subcommands for research evaluation and handoff validation."""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option("--workdir", default=".", show_default=True, help="Directory containing strategy.py")
@click.option("--start", default=None, help="Optional backtest start date passed to run_strategy")
@click.option(
    "--context-json",
    default=None,
    help="Optional JSON file passed to run_strategy(context=...) when supported",
)
@click.option("--output-json", default=None, help="Optional path for raw JSON result")
@click.option("--output-md", default=None, help="Optional path for raw validation markdown")
@click.option("--output-csv", default=None, help="Optional path for metric input CSV")
@click.option("--output-handoff", default=None, help="Optional path for edge-owned handoff JSON")
def evaluate(workdir, start, context_json, output_json, output_md, output_csv, output_handoff):
    """Evaluate one strategy and emit raw validation facts."""
    from causal_edge.research.evaluate import run_evaluation, write_evaluation_outputs

    if output_handoff and (not output_json or not output_md):
        raise click.ClickException("--output-handoff requires both --output-json and --output-md.")

    result = run_evaluation(
        workdir,
        start=start,
        context_json=Path(context_json) if context_json else None,
        output_csv=Path(output_csv) if output_csv else None,
    )
    write_evaluation_outputs(
        result,
        workdir=Path(workdir),
        json_path=Path(output_json) if output_json else None,
        markdown_path=Path(output_md) if output_md else None,
        handoff_path=Path(output_handoff) if output_handoff else None,
    )

    click.echo(f"Verdict: {result.get('verdict', 'ERROR')}")
    click.echo(f"Score:   {result.get('score', '?/?')}")
    click.echo(f"K:       {result.get('K', '?')}")
    if output_json:
        click.echo(f"Raw JSON: {output_json}")
    if output_md:
        click.echo(f"Report:   {output_md}")
    if output_csv:
        click.echo(f"Input CSV: {output_csv}")
    if context_json:
        click.echo(f"Context:  {context_json}")
    if output_handoff:
        click.echo(f"Handoff:  {output_handoff}")
    raise SystemExit(0 if result.get("verdict") == "PASS" else 1)


@click.command("validate-handoff")
@click.argument("handoff_path")
def validate_handoff(handoff_path):
    """Validate an external strategy handoff against the edge contract."""
    from causal_edge.research.handoff import load_strategy_handoff, validate_strategy_handoff

    path = Path(handoff_path)
    if not path.exists():
        raise click.ClickException(f"Handoff not found: {handoff_path}")

    try:
        payload = load_strategy_handoff(path)
    except Exception as exc:
        raise click.ClickException(f"Invalid handoff JSON: {exc}")

    reasons = validate_strategy_handoff(payload, handoff_path=path)
    if reasons:
        click.echo("Handoff rejected:")
        for reason in reasons:
            click.echo(f"- {reason}")
        raise SystemExit(1)

    click.echo("Handoff accepted.")
