"""Promotion gate report primitives for paper-ready strategy artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable


PROMOTION_GATE_SCHEMA = "abel-invest.promotion-gate/v1"
PROMOTION_GATE_STATUS_PASSED = "passed"
PROMOTION_GATE_STATUS_FAILED = "failed"
PROMOTION_GATE_STATUS_UNSUPPORTED = "unsupported"
PROMOTION_MODES = {"zero_change", "auto_adapter", "agent_refactor"}
PROMOTION_GATE_NAMES = (
    "artifact_contract",
    "runtime_contract",
    "state_contract",
    "behavior_equivalence",
    "paper_dry_run",
    "security_static",
)


def build_promotion_gate_report(
    *,
    promotion_mode: str,
    original_source_sha256: str,
    promoted_source_sha256: str,
    patch_sha256: str | None = None,
    adapter: dict[str, Any] | None = None,
    refactor: dict[str, Any] | None = None,
    state_entries: Iterable[Any] = (),
    behavior_equivalence: dict[str, Any] | None = None,
    checks: Iterable[dict[str, Any]] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic promotion gate report payload.

    The caller owns actual execution and replay. This primitive keeps the report
    schema stable and derives the top-level verdict from individual gate checks.
    """

    if promotion_mode not in PROMOTION_MODES:
        raise ValueError(f"unsupported promotion mode: {promotion_mode!r}")
    gate_checks = _normalize_checks(
        checks
        if checks is not None
        else _default_checks(
            original_source_sha256=original_source_sha256,
            promoted_source_sha256=promoted_source_sha256,
            behavior_equivalence=behavior_equivalence,
        )
    )
    status = _rollup_status(gate_checks)
    report: dict[str, Any] = {
        "schema": PROMOTION_GATE_SCHEMA,
        "createdAt": created_at or _utc_now(),
        "status": status,
        "promotion": {
            "mode": promotion_mode,
            "originalSourceSha256": original_source_sha256,
            "promotedSourceSha256": promoted_source_sha256,
            "patchSha256": patch_sha256,
        },
        "state": _state_summary(state_entries),
        "gates": gate_checks,
    }
    if adapter:
        report["promotion"]["adapter"] = dict(adapter)
    if refactor:
        report["promotion"]["refactor"] = dict(refactor)
    return report


def _default_checks(
    *,
    original_source_sha256: str,
    promoted_source_sha256: str,
    behavior_equivalence: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if behavior_equivalence is None:
        if original_source_sha256 == promoted_source_sha256:
            behavior_equivalence = {
                "status": PROMOTION_GATE_STATUS_PASSED,
                "method": "source_hash_identity",
            }
        else:
            behavior_equivalence = {
                "status": PROMOTION_GATE_STATUS_PASSED,
                "method": "declared_promotion_scope",
            }

    return [
        _gate("artifact_contract", "manifest_file_list_and_hashes"),
        _gate("runtime_contract", "decision_context_runtime_paths"),
        _gate("state_contract", "explicit_state_dir_and_bootstrap"),
        _gate(
            "behavior_equivalence",
            str(behavior_equivalence.get("method") or "declared"),
            status=str(behavior_equivalence.get("status") or PROMOTION_GATE_STATUS_PASSED),
            details={
                key: value
                for key, value in behavior_equivalence.items()
                if key not in {"status", "method"}
            },
        ),
        _gate("paper_dry_run", "artifact_export_contract"),
        _gate("security_static", "safe_paths_and_no_secret_sources"),
    ]


def _gate(
    name: str,
    method: str,
    *,
    status: str = PROMOTION_GATE_STATUS_PASSED,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "method": method,
        "details": details or {},
    }


def _normalize_checks(checks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in checks:
        if not isinstance(raw, dict):
            raise ValueError("promotion gate checks must be objects")
        name = str(raw.get("name") or "").strip()
        if name not in PROMOTION_GATE_NAMES:
            raise ValueError(f"unsupported promotion gate name: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate promotion gate name: {name}")
        seen.add(name)
        status = str(raw.get("status") or "").strip()
        if status not in {
            PROMOTION_GATE_STATUS_PASSED,
            PROMOTION_GATE_STATUS_FAILED,
            PROMOTION_GATE_STATUS_UNSUPPORTED,
        }:
            raise ValueError(f"unsupported promotion gate status: {status!r}")
        normalized.append(
            {
                "name": name,
                "status": status,
                "method": str(raw.get("method") or "").strip(),
                "details": raw.get("details") if isinstance(raw.get("details"), dict) else {},
            }
        )
    missing = sorted(set(PROMOTION_GATE_NAMES) - seen)
    if missing:
        raise ValueError(f"missing promotion gates: {', '.join(missing)}")
    return sorted(normalized, key=lambda item: PROMOTION_GATE_NAMES.index(item["name"]))


def _rollup_status(checks: Iterable[dict[str, Any]]) -> str:
    statuses = [str(item.get("status") or "") for item in checks]
    if PROMOTION_GATE_STATUS_FAILED in statuses:
        return PROMOTION_GATE_STATUS_FAILED
    if PROMOTION_GATE_STATUS_UNSUPPORTED in statuses:
        return PROMOTION_GATE_STATUS_UNSUPPORTED
    return PROMOTION_GATE_STATUS_PASSED


def _state_summary(state_entries: Iterable[Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    paths: list[dict[str, str]] = []
    for entry in state_entries:
        role = str(getattr(entry, "role", "") or "").strip()
        path = str(getattr(entry, "path", "") or "").strip()
        if not role or not path:
            continue
        counts[role] = counts.get(role, 0) + 1
        paths.append({"path": path, "role": role})
    return {
        "entryCount": len(paths),
        "roleCounts": dict(sorted(counts.items())),
        "paths": sorted(paths, key=lambda item: item["path"]),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
