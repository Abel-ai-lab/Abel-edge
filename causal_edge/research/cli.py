"""Research CLI group."""

from __future__ import annotations

import csv
from pathlib import Path

import click


@click.group()
def research():
    """Autonomous research loop with L1 enforcement."""


@research.command("init")
@click.argument("ticker")
@click.option("--workdir", default=None, help="Workspace directory")
def research_init(ticker, workdir):
    from causal_edge.research.workspace import init_workspace

    workspace = init_workspace(ticker, workdir)
    click.echo(f"Research workspace: {workspace}/")
    click.echo("  strategy.py   - fill in run_strategy()")
    click.echo("  results.tsv   - append-only experiment log")
    click.echo("  memory.md     - agent memory")
    click.echo("  discovery.json - Abel discovery snapshot")
    click.echo()
    click.echo("Next: edit strategy.py, then run:")
    click.echo(f"  causal-edge research run --workdir {workspace}")


@research.command("run")
@click.option("--workdir", default=".", help="Research workspace dir")
@click.option("--mode", default="exploit", type=click.Choice(["exploit", "explore"]))
@click.option("--description", "-d", default="", help="Experiment description")
def research_run(workdir, mode, description):
    from causal_edge.research.evaluate import append_results_tsv, run_evaluation

    workspace = Path(workdir)
    result = run_evaluation(workspace)
    verdict = result.get("verdict", "ERROR")
    score = result.get("score", "?/?")
    k_value = result.get("K", "?")
    triangle = result.get("triangle", {})

    click.echo(f"\n  Verdict: {verdict}")
    click.echo(f"  Score:   {score}")
    click.echo(f"  K:       {k_value} (auto-computed)")
    click.echo(
        f"  Triangle: Lo={triangle.get('ratio', 0):.2f}  "
        f"IC={triangle.get('rank', 0):.3f}  "
        f"Om={triangle.get('shape', 0):.2f}"
    )

    metrics = result.get("metrics", {})
    if metrics:
        click.echo(
            f"  Sharpe={metrics.get('sharpe', 0):.2f}  "
            f"MaxDD={metrics.get('max_dd', 0) * 100:.1f}%  "
            f"PnL={metrics.get('total_return', 0) * 100:.1f}%"
        )

    failures = result.get("failures", [])
    if failures:
        click.echo("\n  Failures:")
        for failure in failures:
            click.echo(f"    - {failure}")

    status = "keep" if verdict == "PASS" else "discard"
    click.echo(f"\n  {'PASS - recording as KEEP' if status == 'keep' else f'{verdict} - recording as DISCARD'}")
    if not description:
        description = f"{mode}: {verdict} {score}"

    append_results_tsv(workspace, result, status, mode, description)
    click.echo("  Appended to results.tsv")


@research.command("status")
@click.option("--workdir", default=".", help="Research workspace dir")
def research_status(workdir):
    workspace = Path(workdir)
    results_path = workspace / "results.tsv"
    if not results_path.exists():
        click.echo("No results.tsv found. Run 'causal-edge research init' first.")
        return

    with results_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    total = len(rows)
    keep_rows = [row for row in rows if row.get("status") == "keep"]
    discard_count = total - len(keep_rows)

    click.echo(f"  Experiments: {total} ({len(keep_rows)} keep, {discard_count} discard)")
    if rows:
        latest = rows[-1]
        click.echo(
            f"  Latest: {latest.get('description', '?')} "
            f"[{latest.get('score', '?')}] {latest.get('status', '?')}"
        )
    if keep_rows:
        best = keep_rows[-1]
        click.echo(
            f"  Baseline: Sharpe={best.get('sharpe', '?')} "
            f"Lo={best.get('lo_adj', '?')} "
            f"IC={best.get('ic', '?')} "
            f"K={best.get('K', '?')} "
            f"[{best.get('score', '?')}]"
        )
