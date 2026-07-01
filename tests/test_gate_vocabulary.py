from __future__ import annotations

import json

from abel_edge.validation.gate_vocabulary import SCHEMA, gate_dimension_map, list_gate_vocabulary


def test_gate_vocabulary_is_deterministic():
    first = list_gate_vocabulary()
    second = list_gate_vocabulary()

    assert first == second
    assert first["schema"] == SCHEMA
    assert first["vocabulary_hash"].startswith("sha256:")
    assert json.dumps(first, sort_keys=True)


def test_gate_vocabulary_required_dimensions_exist():
    dimensions = gate_dimension_map()

    for dimension in [
        "total_return",
        "sharpe",
        "max_dd",
        "dsr",
        "edge_required_gates",
        "position_bounds",
        "search_width",
    ]:
        assert dimension in dimensions


def test_gate_vocabulary_entries_have_agent_fields():
    required = {
        "id",
        "display_name",
        "meaning",
        "unit",
        "direction",
        "threshold_forms",
        "numeric_meaning",
        "deterministic_check_owner",
        "source_metric",
        "fingerprint",
    }

    for entry in list_gate_vocabulary()["dimensions"]:
        assert required.issubset(entry)
        assert entry["id"]
        assert entry["meaning"]
        assert entry["threshold_forms"]
        assert entry["numeric_meaning"].startswith("Example:")
        assert entry["fingerprint"].startswith("sha256:")


def test_numeric_meaning_does_not_publish_quality_bands():
    banned_terms = {"poor", "usable", "strong"}

    for entry in list_gate_vocabulary()["dimensions"]:
        text = entry["numeric_meaning"].lower()
        assert not any(term in text for term in banned_terms)


def test_gate_dimension_map_returns_dimension_entries():
    dimensions = gate_dimension_map()

    assert dimensions["max_dd"]["direction"] == "higher_is_better"
    assert dimensions["max_dd"]["threshold_forms"] == [
        {"operator": ">=", "value_type": "number"}
    ]
    assert dimensions["edge_required_gates"]["threshold_forms"] == [
        {"operator": "pass_all", "value_type": "boolean"}
    ]
