"""CLI subcommands for research evaluation and handoff validation."""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option("--workdir", default=".", show_default=True, help="Directory containing engine.py")
@click.option("--start", default=None, help="Optional backtest start date injected into research context")
@click.option(
    "--context-json",
    default=None,
    help="Optional JSON file merged into the research engine context",
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


@click.command("debug-evaluate")
@click.option("--workdir", default=".", show_default=True, help="Directory containing engine.py")
@click.option("--start", default=None, help="Optional backtest start date injected into research context")
@click.option(
    "--context-json",
    default=None,
    help="Optional JSON file merged into the research engine context",
)
@click.option("--output-json", default=None, help="Optional path for raw JSON diagnostics")
def debug_evaluate(workdir, start, context_json, output_json):
    """Run evaluation with a diagnostics-first UX for research debugging."""
    import json

    from causal_edge.research.evaluate import run_preflight

    result = run_preflight(
        workdir,
        start=start,
        context_json=Path(context_json) if context_json else None,
    )
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    diagnostics = result.get("diagnostics") or {}
    signal = diagnostics.get("signal") or {}
    semantic = result.get("semantic") or {}
    click.echo(f"Verdict: {result.get('verdict', 'ERROR')}")
    click.echo(f"Semantic verdict: {semantic.get('verdict', 'unknown')}")
    click.echo(f"Failure signature: {diagnostics.get('failure_signature', 'unknown')}")
    click.echo(f"Runtime stage: {diagnostics.get('runtime_stage', 'unknown')}")
    click.echo(f"Read count: {semantic.get('read_count', 0)}")
    click.echo(
        "Signal activity: "
        f"{signal.get('active_days', 0)} / {signal.get('total_days', 0)} active days"
    )
    if diagnostics.get("hints"):
        click.echo("Hints:")
        for hint in diagnostics["hints"]:
            click.echo(f"- {hint}")
    if output_json:
        click.echo(f"Raw JSON: {output_json}")
    raise SystemExit(1 if result.get("verdict") == "ERROR" else 0)


@click.command("verify-data")
@click.option("--discovery-json", default=None, help="Discovery JSON used to source candidate tickers")
@click.option("--ticker", "tickers", multiple=True, help="Explicit ticker to probe (repeatable)")
@click.option("--start", default=None, help="Requested history start date")
@click.option("--end", default=None, help="Requested history end date")
@click.option("--limit", default=5000, show_default=True, help="Rows requested per ticker probe")
@click.option("--env-path", default=".env", show_default=True, help="Env file used to resolve Abel auth")
@click.option("--output-json", default=None, help="Optional path for structured verification JSON")
def verify_data(discovery_json, tickers, start, end, limit, env_path, output_json):
    """Verify discovered or explicit tickers have usable bar data."""
    import json

    from causal_edge.research.data_readiness import (
        render_data_verification_report,
        run_data_verification,
    )

    report = run_data_verification(
        tickers=list(tickers),
        discovery_json=Path(discovery_json) if discovery_json else None,
        start=start,
        end=end,
        limit=limit,
        env_path=env_path,
    )
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    click.echo(render_data_verification_report(report))
    if output_json:
        click.echo(f"\nRaw JSON: {output_json}")
    summary = report.get("summary") or {}
    raise SystemExit(0 if summary.get("error_count", 0) == 0 else 1)


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
