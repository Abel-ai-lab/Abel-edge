from __future__ import annotations

from dataclasses import dataclass

import pytest

from abel_edge.research.promotion_gate import (
    PROMOTION_GATE_SCHEMA,
    build_promotion_gate_report,
)


@dataclass(frozen=True)
class _StateEntry:
    path: str
    role: str


def test_build_promotion_gate_report_for_auto_adapter() -> None:
    report = build_promotion_gate_report(
        promotion_mode="auto_adapter",
        original_source_sha256="a" * 64,
        promoted_source_sha256="b" * 64,
        patch_sha256="c" * 64,
        adapter={"kind": "state_path_adapter", "scope": "state_path_normalization"},
        state_entries=[_StateEntry("model/latest.joblib", "initial_state")],
        behavior_equivalence={
            "status": "passed",
            "method": "state_path_adapter_static_scope",
            "replacements": [{"path": "model/latest.joblib"}],
        },
        paper_dry_run={
            "status": "passed",
            "method": "promoted_metric_input_replay",
            "rowCount": 252,
        },
        created_at="2026-05-08T00:00:00Z",
    )

    assert report["schema"] == PROMOTION_GATE_SCHEMA
    assert report["createdAt"] == "2026-05-08T00:00:00Z"
    assert report["status"] == "passed"
    assert report["promotion"]["mode"] == "auto_adapter"
    assert report["promotion"]["adapter"]["kind"] == "state_path_adapter"
    assert report["state"]["roleCounts"] == {"initial_state": 1}
    assert [item["name"] for item in report["gates"]] == [
        "artifact_contract",
        "runtime_contract",
        "state_contract",
        "behavior_equivalence",
        "paper_dry_run",
        "security_static",
    ]
    behavior_gate = next(
        item for item in report["gates"] if item["name"] == "behavior_equivalence"
    )
    assert behavior_gate["method"] == "state_path_adapter_static_scope"
    dry_run_gate = next(item for item in report["gates"] if item["name"] == "paper_dry_run")
    assert dry_run_gate["method"] == "promoted_metric_input_replay"
    assert dry_run_gate["details"]["rowCount"] == 252


def test_build_promotion_gate_report_rejects_missing_gate() -> None:
    with pytest.raises(ValueError, match="missing promotion gates"):
        build_promotion_gate_report(
            promotion_mode="zero_change",
            original_source_sha256="a" * 64,
            promoted_source_sha256="a" * 64,
            checks=[
                {
                    "name": "artifact_contract",
                    "status": "passed",
                    "method": "pytest",
                    "details": {},
                }
            ],
        )


def test_build_promotion_gate_report_for_agent_paper_contract() -> None:
    report = build_promotion_gate_report(
        promotion_mode="agent_paper_contract",
        original_source_sha256="a" * 64,
        promoted_source_sha256="b" * 64,
        patch_sha256="c" * 64,
        contract={
            "kind": "hosted_paper_contract",
            "summary": "Declared stateful hosted paper contract.",
            "patchPath": "edge/promotion.patch",
            "reportPath": "edge/paper-contract-report.json",
        },
        behavior_equivalence={
            "status": "passed",
            "method": "agent_declared_hosted_paper_contract",
        },
        created_at="2026-05-08T00:00:00Z",
    )

    assert report["status"] == "passed"
    assert report["promotion"]["mode"] == "agent_paper_contract"
    assert report["promotion"]["contract"]["kind"] == "hosted_paper_contract"
    behavior_gate = next(
        item for item in report["gates"] if item["name"] == "behavior_equivalence"
    )
    assert behavior_gate["method"] == "agent_declared_hosted_paper_contract"
