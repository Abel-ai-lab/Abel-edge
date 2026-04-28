"""Strategy handoff contract for upstream orchestrators."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

HANDOFF_CONTRACT = "abel-edge.strategy-handoff/v1"
HANDOFF_REQUIRED_FIELDS = (
    "contract",
    "strategy_path",
    "verdict",
    "profile",
    "blocking_failures",
    "edge_result_path",
    "edge_report_path",
)
HANDOFF_ALLOWED_VERDICTS = {"PASS", "FAIL", "ERROR"}


def build_strategy_handoff(
    result: dict[str, Any],
    *,
    strategy_path: Path,
    result_path: Path,
    report_path: Path,
    handoff_path: Path,
) -> dict[str, Any]:
    failures = result.get("failures") or []
    return {
        "contract": HANDOFF_CONTRACT,
        "strategy_path": _relative_path(strategy_path, handoff_path.parent),
        "verdict": str(result.get("verdict", "ERROR")),
        "profile": str(result.get("profile", "unknown")),
        "blocking_failures": [str(item) for item in failures],
        "edge_result_path": _relative_path(result_path, handoff_path.parent),
        "edge_report_path": _relative_path(report_path, handoff_path.parent),
    }


def write_strategy_handoff(payload: dict[str, Any], handoff_path: Path) -> None:
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_strategy_handoff(handoff_path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(handoff_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Handoff payload must be a JSON object.")
    return payload


def validate_strategy_handoff(
    payload: dict[str, Any], *, handoff_path: Path | str | None = None
) -> list[str]:
    reasons: list[str] = []
    keys = set(payload)
    required = set(HANDOFF_REQUIRED_FIELDS)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        reasons.append(f"Missing required fields: {', '.join(missing)}")
    if unknown:
        reasons.append(f"Unknown fields: {', '.join(unknown)}")

    contract = payload.get("contract")
    if contract != HANDOFF_CONTRACT:
        reasons.append(f"contract must be '{HANDOFF_CONTRACT}', got {contract!r}")

    verdict = payload.get("verdict")
    if not isinstance(verdict, str) or verdict not in HANDOFF_ALLOWED_VERDICTS:
        reasons.append("verdict must be one of PASS, FAIL, ERROR.")

    for key in ("strategy_path", "profile", "edge_result_path", "edge_report_path"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"{key} must be a non-empty string.")

    blocking_failures = payload.get("blocking_failures")
    if not isinstance(blocking_failures, list):
        reasons.append("blocking_failures must be a list of strings.")
    elif any(not isinstance(item, str) or not item.strip() for item in blocking_failures):
        reasons.append("blocking_failures must contain only non-empty strings.")

    if handoff_path is None:
        return reasons

    handoff_file = Path(handoff_path)
    base_dir = handoff_file.parent
    for key in ("strategy_path", "edge_result_path", "edge_report_path"):
        raw_value = payload.get(key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        if Path(raw_value).is_absolute():
            reasons.append(f"{key} must be relative to the handoff file.")

    strategy_file = _resolve(base_dir, payload.get("strategy_path"))
    if strategy_file is not None and not strategy_file.exists():
        reasons.append(f"strategy_path target not found: {payload.get('strategy_path')}")

    report_file = _resolve(base_dir, payload.get("edge_report_path"))
    if report_file is not None and not report_file.exists():
        reasons.append(f"edge_report_path target not found: {payload.get('edge_report_path')}")

    result_file = _resolve(base_dir, payload.get("edge_result_path"))
    if result_file is None:
        return reasons
    if not result_file.exists():
        reasons.append(f"edge_result_path target not found: {payload.get('edge_result_path')}")
        return reasons

    try:
        edge_result = load_strategy_handoff(result_file)
    except Exception as exc:
        reasons.append(f"edge_result_path is not valid JSON: {exc}")
        return reasons

    for key in ("verdict", "profile", "failures"):
        if key not in edge_result:
            reasons.append(f"edge result missing {key}")

    if edge_result.get("verdict") != payload.get("verdict"):
        reasons.append("handoff verdict does not match edge_result_path verdict.")
    if edge_result.get("profile") != payload.get("profile"):
        reasons.append("handoff profile does not match edge_result_path profile.")
    if edge_result.get("failures") != payload.get("blocking_failures"):
        reasons.append("blocking_failures does not match edge_result_path failures.")
    return reasons


def _relative_path(target: Path, base_dir: Path) -> str:
    return os.path.relpath(target, start=base_dir)


def _resolve(base_dir: Path, raw_path: Any) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    return (base_dir / raw_path).resolve()
