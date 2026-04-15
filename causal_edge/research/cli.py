"""Research CLI group."""

from __future__ import annotations

from pathlib import Path

import click

PLACEHOLDER_SNIPPETS = (
    "Document why this exploration started",
    "Record how many candidate branches this exploration produced",
    "State the next exploration move for this session.",
    "Describe the causal or market thesis.",
    "List the target asset, related assets, and any special data inputs.",
    "List the largest assumptions and failure modes.",
    "fill after discovery",
)


@click.group()
def research():
    """Autonomous research loop with L1 enforcement."""


@research.command("init")
@click.argument("ticker")
@click.option("--workdir", default=None, help="Branch workspace directory")
@click.option("--exp-id", default=None, help="Exploration session id; defaults to UTC timestamp")
@click.option("--branch-id", default="baseline", show_default=True, help="Candidate branch id")
def research_init(ticker, workdir, exp_id, branch_id):
    from causal_edge.research.workspace import init_workspace, resolve_session_dir

    branch_dir = init_workspace(ticker, workdir, exp_id=exp_id, branch_id=branch_id)
    session_dir = resolve_session_dir(branch_dir)

    click.echo(f"Research branch workspace: {branch_dir}/")
    click.echo(f"Exploration session: {session_dir}/")
    click.echo("  README.md                 - session-level exploration narrative")
    click.echo("  discovery.json            - Abel discovery snapshot")
    click.echo("  events.tsv                - append-only session event stream")
    click.echo("  branches/<id>/README.md   - branch summary")
    click.echo("  branches/<id>/thesis.md   - branch thesis")
    click.echo("  branches/<id>/rounds/     - per-round notes")
    click.echo("  branches/<id>/results.tsv - append-only experiment ledger")
    click.echo()
    click.echo("Next:")
    click.echo(f"  causal-edge research run --workdir {branch_dir}")
    click.echo(f"  causal-edge research status --workdir {session_dir}")
    click.echo(f"  causal-edge research check --workdir {session_dir}")


@research.command("run")
@click.option("--workdir", default=".", help="Branch workspace dir")
@click.option("--mode", default="exploit", type=click.Choice(["exploit", "explore"]))
@click.option("--description", "description", "-d", default="", help="Round description")
@click.option("--input-note", default="", help="Input object or context for this round")
@click.option("--hypothesis", default="", help="Core hypothesis for this round")
@click.option("--expected-signal", default="", help="Expected signal or outcome")
@click.option("--summary", default="", help="Human summary for the round note")
@click.option("--next-step", default="", help="Planned next move after this round")
@click.option("--action", "actions", multiple=True, help="Concrete actions taken in this round")
def research_run(
    workdir,
    mode,
    description,
    input_note,
    hypothesis,
    expected_signal,
    summary,
    next_step,
    actions,
):
    from causal_edge.research.evaluate import (
        append_results_tsv,
        decide_research_outcome,
        run_evaluation,
    )
    from causal_edge.research.workspace import (
        append_event,
        load_session_summary,
        next_round_id,
        resolve_branch_dir,
        resolve_session_dir,
        update_branch_memory,
        update_branch_readme,
        update_branch_thesis,
        update_session_readme,
        write_round_record,
        write_validation_summary,
    )

    branch_dir = resolve_branch_dir(workdir)
    session_dir = resolve_session_dir(branch_dir)
    _print_recent_context(load_session_summary(session_dir), branch_dir.name)

    result = run_evaluation(branch_dir)
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

    status, decision = decide_research_outcome(branch_dir, result)
    round_id = next_round_id(branch_dir)
    validation_path = write_validation_summary(branch_dir, round_id, result)
    write_round_record(
        branch_dir,
        round_id=round_id,
        ticker=session_dir.parent.name.upper(),
        exp_id=session_dir.name,
        branch_id=branch_dir.name,
        mode=mode,
        description=description,
        decision=decision,
        result=result,
        user_inputs={
            "input_note": input_note,
            "hypothesis": hypothesis,
            "expected_signal": expected_signal,
            "summary": summary,
            "next_step": next_step,
            "actions": list(actions),
        },
    )

    if not description:
        description = f"{mode}: {verdict} {score}"

    append_results_tsv(
        branch_dir,
        result,
        status,
        mode,
        description,
        exp_id=session_dir.name,
        ticker=session_dir.parent.name.upper(),
        branch_id=branch_dir.name,
        round_id=round_id,
        decision=decision,
        validation_path=str(validation_path.relative_to(branch_dir)),
    )
    append_event(
        session_dir,
        event="round_recorded",
        branch_id=branch_dir.name,
        round_id=round_id,
        mode=mode,
        verdict=verdict,
        decision=decision,
        description=description,
        artifact_path=str(validation_path.relative_to(session_dir)),
    )
    update_branch_memory(branch_dir)
    update_branch_thesis(branch_dir)
    update_branch_readme(branch_dir)
    update_session_readme(session_dir)

    click.echo(f"\n  Decision: {decision.upper()} (recorded as {status})")
    click.echo(f"  Round: {round_id}")
    click.echo(f"  Validation summary: {validation_path}")
    click.echo(f"  Appended to {branch_dir / 'results.tsv'}")


@research.command("status")
@click.option("--workdir", default=".", help="Session dir or branch dir")
def research_status(workdir):
    from causal_edge.research.workspace import load_session_summary, resolve_session_dir

    session_dir = resolve_session_dir(workdir)
    summary = load_session_summary(session_dir)
    branches = summary["branches"]

    click.echo(f"Session: {summary['exp_id']} ({summary['ticker']})")
    click.echo(f"Branches: {len(branches)}")
    click.echo(f"Total rounds: {sum(item['total_rounds'] for item in branches)}")
    for item in branches:
        latest = item["latest"]
        click.echo(
            f"  {item['branch_id']:20s} rounds={item['total_rounds']:2d} "
            f"keep={item['keep_count']:2d} discard={item['discard_count']:2d} "
            f"latest={latest.get('round_id', 'none')} {latest.get('decision', 'pending')}"
        )


@research.command("check")
@click.option("--workdir", default=".", help="Session dir or branch dir")
@click.option(
    "--strict",
    is_flag=True,
    help="Fail on placeholder narrative content as well as traceability issues",
)
def research_check(workdir, strict):
    from causal_edge.research.workspace import (
        load_session_summary,
        read_results_rows,
        resolve_session_dir,
    )

    session_dir = resolve_session_dir(workdir)
    summary = load_session_summary(session_dir)
    failures = []
    warnings = []

    if not (session_dir / "README.md").exists():
        failures.append(f"Missing session README.md in {session_dir}")
    if not (session_dir / "discovery.json").exists():
        failures.append(f"Missing discovery.json in {session_dir}")
    if not (session_dir / "events.tsv").exists():
        failures.append(f"Missing events.tsv in {session_dir}")
    if not summary["branches"]:
        failures.append(f"No branch directories found under {session_dir / 'branches'}")

    warnings.extend(_placeholder_findings(session_dir / "README.md", "session README"))

    session_events = summary["events"]
    for item in summary["branches"]:
        branch_dir = item["branch_dir"]
        for required in ("strategy.py", "results.tsv"):
            if not (branch_dir / required).exists():
                failures.append(f"{branch_dir.name}: missing {required}")
        for required_dir in ("rounds", "outputs"):
            if not (branch_dir / required_dir).exists():
                failures.append(f"{branch_dir.name}: missing {required_dir}/")

        for optional in ("README.md", "thesis.md", "memory.md"):
            if not (branch_dir / optional).exists():
                warnings.append(f"{branch_dir.name}: missing optional {optional}")

        warnings.extend(
            _placeholder_findings(branch_dir / "thesis.md", f"{branch_dir.name} thesis")
        )
        warnings.extend(
            _placeholder_findings(branch_dir / "memory.md", f"{branch_dir.name} memory")
        )

        rows = read_results_rows(branch_dir)
        round_ids = {row.get("round_id") for row in rows if row.get("round_id")}
        round_files = {path.stem for path in item["round_files"]}
        if round_ids != round_files:
            failures.append(
                f"{branch_dir.name}: results.tsv round ids do not match rounds/ files ({sorted(round_ids)} vs {sorted(round_files)})"
            )

        branch_event_rounds = {
            event.get("round_id")
            for event in session_events
            if event.get("event") == "round_recorded"
            and event.get("branch_id") == item["branch_id"]
        }
        if round_ids - branch_event_rounds:
            failures.append(
                f"{branch_dir.name}: missing round_recorded events for {sorted(round_ids - branch_event_rounds)}"
            )

        branch_created = any(
            event.get("event") == "branch_created" and event.get("branch_id") == item["branch_id"]
            for event in session_events
        )
        if not branch_created:
            failures.append(f"{branch_dir.name}: missing branch_created event")

        for row in rows:
            round_id = row.get("round_id", "")
            validation_path = row.get("validation_path", "")
            if not round_id:
                failures.append(f"{branch_dir.name}: row missing round_id")
                continue
            round_path = branch_dir / "rounds" / f"{round_id}.md"
            if not round_path.exists():
                failures.append(f"{branch_dir.name}: missing {round_path.name}")
            if validation_path and not (branch_dir / validation_path).exists():
                failures.append(
                    f"{branch_dir.name}: missing validation artifact {validation_path}"
                )
            if round_path.exists():
                text = round_path.read_text(encoding="utf-8")
                for marker in (
                    "## Goal",
                    "## Inputs And Hypothesis",
                    "## Actions",
                    "## Key Results",
                    "## Conclusion",
                ):
                    if marker not in text:
                        failures.append(
                            f"{branch_dir.name}: {round_path.name} missing section {marker}"
                        )

    if strict:
        failures.extend(warnings)
        warnings = []

    if warnings:
        click.echo("Research check warnings:")
        for warning in warnings:
            click.echo(f"  - {warning}")

    if failures:
        click.echo("Research check failed:")
        for failure in failures:
            click.echo(f"  - {failure}")
        raise SystemExit(1)

    click.echo(f"Research check passed for {session_dir}")


def _print_recent_context(summary: dict, current_branch_id: str) -> None:
    click.echo(f"Session context: {summary['exp_id']} ({summary['ticker']})")
    click.echo(f"Branches: {len(summary['branches'])}")
    click.echo(f"Current branch: {current_branch_id}")
    if summary["events"]:
        click.echo("Recent activity:")
        for event in summary["events"][-5:]:
            click.echo(f"  - {_format_event(event)}")

    branch = next(
        (item for item in summary["branches"] if item["branch_id"] == current_branch_id), None
    )
    if branch and branch["latest"]:
        latest = branch["latest"]
        click.echo(
            f"Latest branch result: {latest.get('round_id', 'none')} "
            f"{latest.get('decision', 'pending')} [{latest.get('score', '?/?')}]"
        )


def _format_event(event: dict[str, str]) -> str:
    parts = [event.get("event", "")]
    if event.get("branch_id"):
        parts.append(event["branch_id"])
    if event.get("round_id"):
        parts.append(event["round_id"])
    if event.get("decision"):
        parts.append(event["decision"])
    detail = event.get("description") or event.get("verdict") or ""
    if detail:
        parts.append(f"- {detail}")
    return " ".join(parts).strip()


def _placeholder_findings(path: Path, label: str) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    findings = []
    for snippet in PLACEHOLDER_SNIPPETS:
        if snippet in text:
            findings.append(f"{label} still contains placeholder text: {snippet}")
    return findings
