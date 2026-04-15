"""Research workspace initialization helpers."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from causal_edge.plugins.abel.client import AbelClient, normalize_public_node_id
from causal_edge.plugins.abel.credentials import resolve_api_key
from causal_edge.research.constants import (
    BRANCHES_DIRNAME,
    EVENTS_COLUMNS,
    EVENTS_HEADER,
    RESULTS_HEADER,
)

STRATEGY_TEMPLATE = '''"""Strategy for {ticker} - experiment baseline.

Fill in run_strategy(). Everything else is handled by causal-edge.
Run: python -m causal_edge.research.evaluate --workdir .
"""

import numpy as np
import pandas as pd


def run_strategy():
    """Your strategy logic. Returns (pnl, dates, positions)."""
    raise NotImplementedError("Fill in run_strategy()")
'''

SESSION_README_TEMPLATE = """# {ticker} Exploration Session {exp_id}

## Executive Summary

{executive_summary}

## Session Summary

- ticker: `{ticker}`
- exp_id: `{exp_id}`
- started_at: `{started_at}`
- discovery_source: `{discovery_source}`
- current_status: `{current_status}`
- branch_count: `{branch_count}`

## Session Goal

{session_goal}

## Selection Narrative

{selection_narrative}

## Branches

{branch_lines}

## Branch Outcome Snapshot

{branch_snapshot_lines}

## Recent Activity

{activity_lines}

## Next Step

{next_step}
"""

BRANCH_README_TEMPLATE = """# {branch_id}

## Basic Info

- branch_id: `{branch_id}`
- ticker: `{ticker}`
- exp_id: `{exp_id}`
- current_status: `exploring`
- total_rounds: `0`
- latest_round: `none`
- validation_status: `not_validated`

## Branch Thesis

See `thesis.md` for the branch hypothesis.

## Latest Conclusion

- decision: `pending`
- summary: `No rounds recorded yet.`
- next_step: `Run causal-edge research run after implementing strategy.py.`

## Round Ledger

`No rounds yet.`
"""

THESIS_TEMPLATE = """# {branch_id} Thesis

## Alpha Source

`Describe the causal or market thesis.`

## Input Universe

`List the target asset, related assets, and any special data inputs.`

## Main Risks

`List the largest assumptions and failure modes.`
"""

MEMORY_TEMPLATE = """# {ticker} Research Memory

## K Budget
- Discovery: {discovery_summary}

## Baseline
- No KEEP baseline yet.

## Exhausted Directions

- none recorded yet

## What Worked

- none recorded yet

## Ideas Not Yet Tried

- none recorded yet
"""


def init_workspace(
    ticker: str,
    workdir: Path | str | None = None,
    *,
    exp_id: str | None = None,
    branch_id: str = "baseline",
) -> Path:
    ticker = ticker.upper()
    branch_id = normalize_branch_id(branch_id)

    if workdir is None:
        exp_id = normalize_exp_id(exp_id or default_exp_id())
        session_dir = Path("research") / ticker.lower() / exp_id
        branch_dir = session_dir / BRANCHES_DIRNAME / branch_id
    else:
        raw_path = Path(workdir)
        if raw_path.parent.name == BRANCHES_DIRNAME:
            branch_dir = raw_path
            session_dir = raw_path.parent.parent
            branch_id = normalize_branch_id(raw_path.name)
        elif raw_path.name == BRANCHES_DIRNAME:
            session_dir = raw_path.parent
            branch_dir = raw_path / branch_id
        else:
            session_dir = raw_path
            branch_dir = session_dir / BRANCHES_DIRNAME / branch_id

    session_dir = session_dir.resolve()
    branch_dir = branch_dir.resolve()
    session_created = not session_dir.exists()
    branch_created = not branch_dir.exists()

    discovery = _load_or_create_discovery(session_dir, ticker)
    _write_session_files(session_dir, ticker, discovery)
    _write_branch_files(branch_dir, ticker, session_dir.name, branch_id)

    if session_created:
        append_event(
            session_dir, event="session_created", description="Initialized research session"
        )
    if branch_created:
        append_event(
            session_dir,
            event="branch_created",
            branch_id=branch_id,
            description="Initialized research branch",
        )

    update_session_readme(session_dir)
    return branch_dir


def normalize_exp_id(exp_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", exp_id.strip())
    cleaned = cleaned.strip("-_")
    if not cleaned:
        raise ValueError("exp_id must contain at least one letter or number")
    return cleaned.lower()


def normalize_branch_id(branch_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", branch_id.strip())
    cleaned = cleaned.strip("-_")
    if not cleaned:
        raise ValueError("branch_id must contain at least one letter or number")
    return cleaned.lower()


def default_exp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def resolve_session_dir(workdir: Path | str | None = None) -> Path:
    path = Path(workdir or ".").resolve()
    if path.name == BRANCHES_DIRNAME and path.parent.exists():
        return path.parent
    if path.parent.name == BRANCHES_DIRNAME:
        return path.parent.parent
    return path


def resolve_branch_dir(workdir: Path | str | None = None) -> Path:
    path = Path(workdir or ".").resolve()
    if path.name == BRANCHES_DIRNAME:
        raise ValueError(
            f"Point --workdir at a branch directory or session directory, not {BRANCHES_DIRNAME}/."
        )
    if path.parent.name == BRANCHES_DIRNAME:
        return path

    branches_dir = path / BRANCHES_DIRNAME
    if branches_dir.exists():
        candidates = sorted(child for child in branches_dir.iterdir() if child.is_dir())
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(f"No branch directories found under {branches_dir}")
        raise ValueError(
            "Multiple branch directories found; pass --workdir to one specific branch."
        )

    return path


def load_session_summary(session_dir: Path) -> dict:
    discovery = {}
    discovery_path = session_dir / "discovery.json"
    if discovery_path.exists():
        discovery = json.loads(discovery_path.read_text(encoding="utf-8"))

    branches = []
    branches_dir = session_dir / BRANCHES_DIRNAME
    if branches_dir.exists():
        for child in sorted(branches_dir.iterdir()):
            if child.is_dir():
                branches.append(load_branch_summary(child))

    return {
        "session_dir": session_dir,
        "ticker": discovery.get("ticker", session_dir.parent.name.upper()),
        "exp_id": session_dir.name,
        "discovery": discovery,
        "branches": branches,
        "events": read_events_rows(session_dir),
    }


def load_branch_summary(branch_dir: Path) -> dict:
    results = read_results_rows(branch_dir)
    round_files = (
        sorted((branch_dir / "rounds").glob("round-*.md"))
        if (branch_dir / "rounds").exists()
        else []
    )
    keep_rows = [row for row in results if row.get("status") == "keep"]
    latest = results[-1] if results else {}
    return {
        "branch_dir": branch_dir,
        "branch_id": branch_dir.name,
        "results": results,
        "total_rounds": len(results),
        "keep_count": len(keep_rows),
        "discard_count": len(results) - len(keep_rows),
        "latest": latest,
        "round_files": round_files,
    }


def read_results_rows(branch_dir: Path) -> list[dict[str, str]]:
    return _read_tsv_rows(branch_dir / "results.tsv")


def read_events_rows(session_dir: Path) -> list[dict[str, str]]:
    return _read_tsv_rows(session_dir / "events.tsv")


def next_round_id(branch_dir: Path) -> str:
    next_index = len(read_results_rows(branch_dir)) + 1
    return f"round-{next_index:03d}"


def append_event(
    session_dir: Path,
    *,
    event: str,
    branch_id: str = "",
    round_id: str = "",
    mode: str = "",
    verdict: str = "",
    decision: str = "",
    description: str = "",
    artifact_path: str = "",
) -> None:
    events_path = session_dir / "events.tsv"
    if not events_path.exists():
        events_path.write_text(EVENTS_HEADER, encoding="utf-8")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "branch_id": branch_id,
        "round_id": round_id,
        "mode": mode,
        "verdict": verdict,
        "decision": decision,
        "description": description,
        "artifact_path": artifact_path,
    }
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(row[column]) for column in EVENTS_COLUMNS) + "\n")


def write_round_record(
    branch_dir: Path,
    *,
    round_id: str,
    ticker: str,
    exp_id: str,
    branch_id: str,
    mode: str,
    description: str,
    decision: str,
    result: dict,
    user_inputs: dict[str, object] | None = None,
) -> Path:
    user_inputs = user_inputs or {}
    metrics = result.get("metrics", {})
    actions = list(user_inputs.get("actions") or [])
    if not actions:
        actions = [
            "Executed strategy.py evaluation",
            "Ran validation and baseline comparison",
        ]

    action_lines = "\n".join(f"1. {action}" for action in actions)
    content = f"""# {round_id}

## Basic Info

- date: `{datetime.now(timezone.utc).strftime("%Y-%m-%d")}`
- ticker: `{ticker}`
- exp_id: `{exp_id}`
- branch_id: `{branch_id}`
- mode: `{mode}`
- decision: `{decision}`
- score: `{result.get("score", "?/?")}`
- verdict: `{result.get("verdict", "ERROR")}`

## Goal

`{description or "No description supplied."}`

## Inputs And Hypothesis

- input: `{user_inputs.get("input_note") or f"Branch {branch_id} entering {round_id}."}`
- hypothesis: `{user_inputs.get("hypothesis") or "No hypothesis supplied."}`
- expected_signal: `{user_inputs.get("expected_signal") or "Improve validation outcome versus the latest KEEP baseline."}`

## Actions

{action_lines}

## Key Results

- lo_adjusted: `{metrics.get("lo_adjusted", 0):.3f}`
- position_ic: `{metrics.get("position_ic", 0):.4f}`
- omega: `{metrics.get("omega", 0):.3f}`
- sharpe: `{metrics.get("sharpe", 0):.3f}`
- total_return: `{metrics.get("total_return", 0) * 100:.1f}%`
- max_dd: `{metrics.get("max_dd", 0) * 100:.1f}%`
- failures: `{"; ".join(result.get("failures", [])) or "none"}`

## Conclusion

- summary: `{user_inputs.get("summary") or f"Recorded {decision} after {result.get("verdict", "ERROR")} {result.get("score", "?/?")}. "}`
- next_step: `{user_inputs.get("next_step") or "Review the validation summary and decide whether to continue or branch."}`
"""

    round_path = branch_dir / "rounds" / f"{round_id}.md"
    round_path.write_text(content, encoding="utf-8")
    return round_path


def write_validation_summary(branch_dir: Path, round_id: str, result: dict) -> Path:
    metrics = result.get("metrics", {})
    triangle = result.get("triangle", {})
    summary = f"""# {round_id} Validation Summary

## Verdict

- verdict: `{result.get("verdict", "ERROR")}`
- score: `{result.get("score", "?/?")}`
- K: `{result.get("K", "?")}`

## Triangle

- lo_ratio: `{triangle.get("ratio", 0):.3f}`
- rank_ic: `{triangle.get("rank", 0):.4f}`
- omega_shape: `{triangle.get("shape", 0):.3f}`

## Metrics

- lo_adjusted: `{metrics.get("lo_adjusted", 0):.3f}`
- position_ic: `{metrics.get("position_ic", 0):.4f}`
- omega: `{metrics.get("omega", 0):.3f}`
- sharpe: `{metrics.get("sharpe", 0):.3f}`
- total_return: `{metrics.get("total_return", 0) * 100:.1f}%`
- max_dd: `{metrics.get("max_dd", 0) * 100:.1f}%`

## Failures

{_format_failure_list(result.get("failures", []))}
"""
    output_path = branch_dir / "outputs" / f"{round_id}-validation.md"
    output_path.write_text(summary, encoding="utf-8")
    return output_path


def update_branch_readme(branch_dir: Path) -> None:
    rows = read_results_rows(branch_dir)
    keep_rows = [row for row in rows if row.get("status") == "keep"]
    latest = rows[-1] if rows else {}
    latest_note = _read_round_note(branch_dir, latest.get("round_id", "")) if latest else {}
    ledger_lines = (
        "\n".join(
            f"1. `{row.get('round_id', '?')}` - {row.get('description', '?')} [{row.get('score', '?')}] {row.get('decision', '?')}"
            for row in rows
        )
        or "`No rounds yet.`"
    )
    rationale_lines = _branch_rationale_lines(rows, latest_note)
    progression_lines = _branch_progression_lines(rows)

    content = f"""# {branch_dir.name}

## Basic Info

- branch_id: `{branch_dir.name}`
- ticker: `{latest.get("ticker", branch_dir.parent.parent.parent.name.upper())}`
- exp_id: `{latest.get("exp_id", branch_dir.parent.parent.name)}`
- current_status: `{_branch_status(rows)}`
- total_rounds: `{len(rows)}`
- latest_round: `{latest.get("round_id", "none")}`
- validation_status: `{latest.get("verdict", "not_validated")}`

## Branch Thesis

See `thesis.md` for the branch hypothesis.

## Latest Conclusion

- decision: `{latest.get("decision", "pending")}`
- summary: `{latest.get("description", "No rounds recorded yet.")}`
- next_step: `Review the latest round note for the next move.`

## Decision Rationale

{rationale_lines}

## Round Ledger

{ledger_lines}

## Metric Progression

{progression_lines}

## Baseline

- keep_rounds: `{len(keep_rows)}`
- latest_keep: `{keep_rows[-1].get("round_id", "none") if keep_rows else "none"}`
"""

    (branch_dir / "README.md").write_text(content, encoding="utf-8")


def update_branch_memory(branch_dir: Path) -> None:
    rows = read_results_rows(branch_dir)
    session_dir = branch_dir.parent.parent
    discovery_path = session_dir / "discovery.json"
    discovery = {}
    if discovery_path.exists():
        discovery = json.loads(discovery_path.read_text(encoding="utf-8"))

    keep_rows = [row for row in rows if row.get("decision") == "keep"]
    discard_rows = [row for row in rows if row.get("decision") == "discard"]
    baseline_line = (
        f"- latest KEEP: {keep_rows[-1].get('round_id', 'none')} "
        f"({keep_rows[-1].get('description', 'n/a')})"
        if keep_rows
        else "- No KEEP baseline yet."
    )
    exhausted_lines = (
        "\n".join(
            f"- {row.get('round_id', '?')} {row.get('description', 'discarded')}"
            for row in discard_rows[-5:]
        )
        if discard_rows
        else "- none recorded yet"
    )
    worked_lines = (
        "\n".join(
            f"- {row.get('round_id', '?')} {row.get('description', 'kept')}"
            for row in keep_rows[-5:]
        )
        if keep_rows
        else "- none recorded yet"
    )

    content = MEMORY_TEMPLATE.format(
        ticker=discovery.get("ticker", branch_dir.parent.parent.parent.name.upper()),
        discovery_summary=_discovery_summary(discovery),
    )
    content += (
        f"\n## Baseline\n{baseline_line}\n\n"
        f"## Exhausted Directions\n{exhausted_lines}\n\n"
        f"## What Worked\n{worked_lines}\n\n"
        "## Ideas Not Yet Tried\n- record the next untested branch idea here\n"
    )
    content = content.replace(
        "## Baseline\n- No KEEP baseline yet.\n\n## Exhausted Directions\n\n- none recorded yet\n\n## What Worked\n\n- none recorded yet\n\n## Ideas Not Yet Tried\n\n- none recorded yet\n",
        "",
    )
    (branch_dir / "memory.md").write_text(content, encoding="utf-8")


def update_branch_thesis(branch_dir: Path) -> None:
    rows = read_results_rows(branch_dir)
    session_dir = branch_dir.parent.parent
    discovery_path = session_dir / "discovery.json"
    discovery = {}
    if discovery_path.exists():
        discovery = json.loads(discovery_path.read_text(encoding="utf-8"))

    latest = rows[-1] if rows else {}
    latest_note = _read_round_note(branch_dir, latest.get("round_id", "")) if latest else {}
    parents = ", ".join(discovery.get("parents", [])[:5]) or "none recorded"
    blanket = ", ".join(discovery.get("blanket_new", [])[:5]) or "none recorded"
    risks = latest_note.get("failures", "none recorded")
    hypothesis = _latest_recorded_hypothesis(branch_dir, rows) or latest.get(
        "description", "Initial branch hypothesis not recorded yet"
    )

    content = f"""# {branch_dir.name} Thesis

## Alpha Source

Branch `{branch_dir.name}` currently assumes: `{hypothesis}`.
Latest decision is `{latest.get("decision", "pending")}` with verdict `{latest.get("verdict", "not_validated")}`.

## Input Universe

- target: `{discovery.get("ticker", branch_dir.parent.parent.parent.name.upper())}`
- discovery_source: `{discovery.get("source", "unknown")}`
- direct_parents: `{parents}`
- blanket_candidates: `{blanket}`

## Main Risks

{_format_risk_lines(risks)}
"""
    (branch_dir / "thesis.md").write_text(content, encoding="utf-8")


def update_session_readme(session_dir: Path) -> None:
    summary = load_session_summary(session_dir)
    discovery = summary["discovery"]
    branch_lines = (
        "\n".join(
            f"1. `{item['branch_id']}` - {item['total_rounds']} rounds, latest `{item['latest'].get('round_id', 'none')}` {item['latest'].get('decision', 'pending')}"
            for item in summary["branches"]
        )
        or "1. `No branches yet.`"
    )
    activity_lines = (
        "\n".join(_format_event_line(event) for event in summary["events"][-5:])
        or "1. `No events yet.`"
    )
    branch_snapshot_lines = _session_branch_snapshot_lines(summary["branches"])

    content = SESSION_README_TEMPLATE.format(
        ticker=summary["ticker"],
        exp_id=summary["exp_id"],
        executive_summary=_session_executive_summary(summary),
        started_at=discovery.get("created_at", "unknown"),
        discovery_source=discovery.get("source", "unknown"),
        current_status=_session_status(summary["branches"]),
        branch_count=len(summary["branches"]),
        session_goal=_session_goal(summary),
        selection_narrative=_selection_narrative(summary),
        branch_lines=branch_lines,
        branch_snapshot_lines=branch_snapshot_lines,
        activity_lines=activity_lines,
        next_step=_session_next_step(summary),
    )
    (session_dir / "README.md").write_text(content, encoding="utf-8")


def _load_or_create_discovery(session_dir: Path, ticker: str) -> dict:
    discovery_path = session_dir / "discovery.json"
    if discovery_path.exists():
        return json.loads(discovery_path.read_text(encoding="utf-8"))

    session_dir.mkdir(parents=True, exist_ok=True)
    discovery = _try_abel_discovery(ticker)
    discovery["created_at"] = datetime.now(timezone.utc).isoformat()
    discovery_path.write_text(json.dumps(discovery, indent=2), encoding="utf-8")
    return discovery


def _write_session_files(session_dir: Path, ticker: str, discovery: dict) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    _write_if_missing(session_dir / "events.tsv", EVENTS_HEADER)
    _write_if_missing(
        session_dir / "README.md",
        SESSION_README_TEMPLATE.format(
            ticker=ticker,
            exp_id=session_dir.name,
            executive_summary="No validated rounds yet. Start the first branch to establish the session baseline.",
            started_at=discovery.get("created_at", "unknown"),
            discovery_source=discovery.get("source", "unknown"),
            current_status="exploring",
            branch_count=0,
            session_goal="Initialize the first branch and record the first validated round for this session.",
            selection_narrative="No branches have run yet. This session will summarize branch outcomes once validation begins.",
            branch_lines="1. `No branches yet.`",
            branch_snapshot_lines="1. `No branch outcomes yet.`",
            activity_lines="1. `No events yet.`",
            next_step="Run `causal-edge research run` on the first branch to establish a baseline.",
        ),
    )


def _write_branch_files(branch_dir: Path, ticker: str, exp_id: str, branch_id: str) -> None:
    branch_dir.mkdir(parents=True, exist_ok=True)
    _write_if_missing(branch_dir / "strategy.py", STRATEGY_TEMPLATE.format(ticker=ticker))
    _write_if_missing(branch_dir / "results.tsv", RESULTS_HEADER)
    session_dir = branch_dir.parent.parent
    discovery = _load_or_create_discovery(session_dir, ticker)
    _write_if_missing(
        branch_dir / "memory.md",
        MEMORY_TEMPLATE.format(
            ticker=ticker,
            discovery_summary=_discovery_summary(discovery),
        ),
    )
    _write_if_missing(
        branch_dir / "README.md",
        BRANCH_README_TEMPLATE.format(ticker=ticker, exp_id=exp_id, branch_id=branch_id),
    )
    _write_if_missing(branch_dir / "thesis.md", THESIS_TEMPLATE.format(branch_id=branch_id))
    (branch_dir / "rounds").mkdir(parents=True, exist_ok=True)
    (branch_dir / "outputs").mkdir(parents=True, exist_ok=True)


def _try_abel_discovery(ticker: str) -> dict:
    api_key = _resolve_workspace_api_key()
    if not api_key:
        return {
            "ticker": ticker,
            "source": "template (no ABEL_API_KEY)",
            "parents": [],
            "blanket_new": [],
            "children": [],
            "K_discovery": 0,
            "note": "Set ABEL_API_KEY or run causal-edge discover manually.",
        }

    try:
        client = AbelClient()
        node_id = normalize_public_node_id(ticker)
        with ThreadPoolExecutor(max_workers=2) as pool:
            parents_future = pool.submit(
                client.discover_parents, node_id=node_id, limit=20, api_key=api_key
            )
            blanket_future = pool.submit(
                client.markov_blanket, node_id=node_id, limit=20, api_key=api_key
            )

        parents = [_pick_node_id(item) for item in parents_future.result()]
        parents = [node for node in parents if node]
        blanket_nodes = [_pick_node_id(item) for item in blanket_future.result()]
        blanket_new = sorted(node for node in blanket_nodes if node and node not in parents)
        k_discovery = len(set(parents + blanket_new))
        return {
            "ticker": ticker,
            "source": "Abel CAP (live)",
            "parents": parents,
            "blanket_new": blanket_new,
            "children": [],
            "K_discovery": k_discovery,
            "note": f"K={k_discovery} tickers from Abel. Scan K = K x n_lags.",
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "source": f"abel_error: {exc}",
            "parents": [],
            "blanket_new": [],
            "children": [],
            "K_discovery": 0,
        }


def _resolve_workspace_api_key() -> str | None:
    token = resolve_api_key(env_path=".env")
    if token:
        return token

    skill_paths = (
        Path.home() / ".agents/skills/causal-abel/.env.skill",
        Path.home() / ".claude/skills/causal-abel/.env.skill",
    )
    for env_path in skill_paths:
        if not env_path.exists():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            if raw.startswith("ABEL_API_KEY="):
                return raw.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= 1:
        return []
    headers = lines[0].split("\t")
    rows: list[dict[str, str]] = []
    for raw in lines[1:]:
        if not raw.strip():
            continue
        values = raw.split("\t")
        rows.append(
            {
                header: values[idx] if idx < len(values) else ""
                for idx, header in enumerate(headers)
            }
        )
    return rows


def _pick_node_id(item: dict) -> str:
    for key in ("node_id", "id", "name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def _format_failure_list(failures: list[str]) -> str:
    if not failures:
        return "- none"
    return "\n".join(f"- {failure}" for failure in failures)


def _branch_status(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "exploring"
    return rows[-1].get("decision", rows[-1].get("status", "exploring"))


def _session_status(branches: list[dict]) -> str:
    if not branches:
        return "exploring"
    if any(branch["latest"].get("decision") == "keep" for branch in branches if branch["latest"]):
        return "has_keep"
    if any(branch["total_rounds"] > 0 for branch in branches):
        return "active"
    return "exploring"


def _format_event_line(event: dict[str, str]) -> str:
    tail = " ".join(
        part
        for part in (
            event.get("branch_id", ""),
            event.get("round_id", ""),
            event.get("decision", ""),
        )
        if part
    )
    detail = event.get("description", "") or event.get("verdict", "") or event.get("event", "")
    return f"1. `{event.get('timestamp', '')}` {event.get('event', '')} {tail} - {detail}".rstrip()


def _session_goal(summary: dict) -> str:
    return (
        f"Explore {summary['ticker']} in session `{summary['exp_id']}` using discovery source "
        f"`{summary['discovery'].get('source', 'unknown')}` and compare candidate branches through validated rounds."
    )


def _selection_narrative(summary: dict) -> str:
    branches = summary["branches"]
    if not branches:
        return "No branches have run yet. Start the first branch to build the exploration record."

    keep_count = sum(1 for item in branches if item["latest"].get("decision") == "keep")
    discard_count = sum(1 for item in branches if item["latest"].get("decision") == "discard")
    pending_count = len(branches) - keep_count - discard_count
    branch_states = ", ".join(
        f"`{item['branch_id']}`={item['latest'].get('decision', 'pending')}" for item in branches
    )
    return (
        f"This session tracks {len(branches)} branch(es): {branch_states}. "
        f"Current outcomes: {keep_count} keep, {discard_count} discard, {pending_count} pending."
    )


def _session_next_step(summary: dict) -> str:
    branches = summary["branches"]
    if not branches:
        return "Run the first branch validation round to create a baseline for the session."

    keep_branches = [item for item in branches if item["latest"].get("decision") == "keep"]
    discard_branches = [item for item in branches if item["latest"].get("decision") == "discard"]

    if keep_branches and discard_branches:
        return f"Continue improving `{keep_branches[-1]['branch_id']}` or branch from the discarded ideas now that both keep and discard outcomes are recorded."
    if keep_branches:
        return f"Continue improving `{keep_branches[-1]['branch_id']}` or open a sibling branch from its latest KEEP baseline."
    return "Revise the discarded branches or open a new branch with a different hypothesis before the next validation round."


def _discovery_summary(discovery: dict) -> str:
    return f"K={discovery.get('K_discovery', 0)} via {discovery.get('source', 'unknown')}"


def _session_executive_summary(summary: dict) -> str:
    branches = summary["branches"]
    if not branches:
        return "No validated rounds yet. Start the first branch to establish the session baseline."

    keep_branches = [item for item in branches if item["latest"].get("decision") == "keep"]
    discard_branches = [item for item in branches if item["latest"].get("decision") == "discard"]
    leader = keep_branches[-1] if keep_branches else branches[0]
    leader_latest = leader["latest"]
    headline = (
        f"Session has {len(branches)} branch(es): {len(keep_branches)} keep and {len(discard_branches)} discard. "
        f"Current lead is `{leader['branch_id']}` at `{leader_latest.get('round_id', 'none')}` with "
        f"Lo {float(leader_latest.get('lo_adj') or 0):.3f}, Sharpe {float(leader_latest.get('sharpe') or 0):.3f}, "
        f"PnL {float(leader_latest.get('pnl') or 0):.1f}%."
    )
    if discard_branches:
        laggard = discard_branches[-1]
        laggard_latest = laggard["latest"]
        headline += (
            f" Discarded branch `{laggard['branch_id']}` stalled at Lo {float(laggard_latest.get('lo_adj') or 0):.3f} "
            f"and PnL {float(laggard_latest.get('pnl') or 0):.1f}%"
        )
        failures = _read_round_note(laggard["branch_dir"], laggard_latest.get("round_id", "")).get(
            "failures"
        )
        if failures and failures != "none":
            headline += f" because `{failures}`."
        else:
            headline += "."
    return headline


def _format_risk_lines(risks: str) -> str:
    cleaned = (risks or "").strip()
    if not cleaned or cleaned == "none" or cleaned == "none recorded":
        return "- no acute validation failures recorded yet"
    parts = [part.strip() for part in cleaned.split(";") if part.strip()]
    return "\n".join(f"- {part}" for part in parts)


def _read_round_note(branch_dir: Path, round_id: str) -> dict[str, str]:
    if not round_id:
        return {}
    round_path = branch_dir / "rounds" / f"{round_id}.md"
    if not round_path.exists():
        return {}

    fields: dict[str, str] = {}
    for raw in round_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        for key in ("hypothesis", "expected_signal", "failures", "summary", "next_step"):
            prefix = f"- {key}: `"
            if line.startswith(prefix) and line.endswith("`"):
                fields[key] = line[len(prefix) : -1]
    return fields


def _latest_recorded_hypothesis(branch_dir: Path, rows: list[dict[str, str]]) -> str:
    for row in reversed(rows):
        note = _read_round_note(branch_dir, row.get("round_id", ""))
        hypothesis = (note.get("hypothesis") or "").strip()
        if hypothesis and hypothesis != "No hypothesis supplied.":
            return hypothesis
    return ""


def _branch_progression_lines(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "`No metric progression yet.`"

    lines = []
    previous = None
    for row in rows:
        lo_adj = float(row.get("lo_adj") or 0)
        sharpe = float(row.get("sharpe") or 0)
        pnl = float(row.get("pnl") or 0)
        delta_text = ""
        if previous is not None:
            delta_text = (
                f" | dLo {lo_adj - previous['lo_adj']:+.3f}"
                f" | dSharpe {sharpe - previous['sharpe']:+.3f}"
                f" | dPnL {pnl - previous['pnl']:+.1f}%"
            )
        lines.append(
            f"1. `{row.get('round_id', '?')}` {row.get('decision', '?')}"
            f" | Lo {lo_adj:.3f} | Sharpe {sharpe:.3f} | PnL {pnl:.1f}%{delta_text}"
        )
        previous = {"lo_adj": lo_adj, "sharpe": sharpe, "pnl": pnl}
    return "\n".join(lines)


def _branch_rationale_lines(rows: list[dict[str, str]], latest_note: dict[str, str]) -> str:
    if not rows:
        return "1. `No rationale recorded yet.`"

    latest = rows[-1]
    lines = [
        f"1. latest_hypothesis: `{latest_note.get('hypothesis', 'not recorded')}`",
        f"1. latest_summary: `{latest_note.get('summary', latest.get('description', 'not recorded'))}`",
    ]
    failures = latest_note.get("failures", "")
    if failures and failures != "none":
        lines.append(f"1. latest_failures: `{failures}`")
    else:
        lines.append("1. latest_failures: `none`")
    return "\n".join(lines)


def _session_branch_snapshot_lines(branches: list[dict]) -> str:
    if not branches:
        return "1. `No branch outcomes yet.`"

    lines = []
    for item in branches:
        rows = item["results"]
        latest = item["latest"]
        latest_note = (
            _read_round_note(item["branch_dir"], latest.get("round_id", "")) if latest else {}
        )
        first = rows[0] if rows else {}
        latest_lo = float(latest.get("lo_adj") or 0)
        latest_sharpe = float(latest.get("sharpe") or 0)
        latest_pnl = float(latest.get("pnl") or 0)
        first_lo = float(first.get("lo_adj") or 0)
        first_sharpe = float(first.get("sharpe") or 0)
        first_pnl = float(first.get("pnl") or 0)
        reason = (
            latest_note.get("hypothesis")
            or latest_note.get("failures")
            or latest.get("description", "")
        )
        lines.append(
            f"1. `{item['branch_id']}` -> `{latest.get('decision', 'pending')}` after {item['total_rounds']} round(s). "
            f"Why: `{reason or 'not recorded'}`. "
            f"Trend: Lo {first_lo:.3f} -> {latest_lo:.3f}, Sharpe {first_sharpe:.3f} -> {latest_sharpe:.3f}, "
            f"PnL {first_pnl:.1f}% -> {latest_pnl:.1f}%."
        )
    return "\n".join(lines)
