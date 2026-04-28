from __future__ import annotations

from abel_edge.graph_nodes import (
    GraphNodeRef,
    coerce_graph_node_ref,
    coerce_graph_node_refs,
    graph_node_assets,
    graph_node_runtime_field,
)


def test_coerce_graph_node_ref_from_dict_preserves_field_and_roles() -> None:
    ref = coerce_graph_node_ref(
        {"ticker": "tsla", "field": "volume", "roles": ["neighbor"]},
        extra_roles=["selected"],
    )

    assert ref == GraphNodeRef(
        node_id="TSLA.volume",
        asset="TSLA",
        field="volume",
        roles=("selected", "neighbor"),
    )


def test_coerce_graph_node_refs_deduplicates_by_node_id() -> None:
    refs = coerce_graph_node_refs(
        [
            {"ticker": "AAPL", "field": "price"},
            "AAPL.price",
            {"node_id": "AAPL.volume"},
        ]
    )

    assert [ref.node_id for ref in refs] == ["AAPL.price", "AAPL.volume"]
    assert graph_node_assets(refs) == ["AAPL"]


def test_graph_node_runtime_field_maps_price_and_volume() -> None:
    assert graph_node_runtime_field("price") == "close"
    assert graph_node_runtime_field("volume") == "volume"
