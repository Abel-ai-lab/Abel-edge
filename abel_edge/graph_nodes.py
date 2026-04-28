"""Helpers for graph-node-first session and branch contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

CRYPTO_ALIASES = {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX"}
SUPPORTED_GRAPH_FIELDS = {"price", "volume"}


@dataclass(frozen=True)
class GraphNodeRef:
    """Canonical reference to one graph node."""

    node_id: str
    asset: str
    field: str
    roles: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "node_id": self.node_id,
            "asset": self.asset,
            "field": self.field,
        }
        if self.roles:
            payload["roles"] = list(self.roles)
        return payload

    def label(self, *, include_roles: bool = False) -> str:
        if include_roles and self.roles:
            return f"{self.node_id} ({', '.join(self.roles)})"
        return self.node_id


def coerce_graph_node_ref(
    value: Any,
    *,
    default_field: str = "price",
    extra_roles: Iterable[str] | None = None,
) -> GraphNodeRef | None:
    """Normalize a string/dict payload into a canonical ``GraphNodeRef``."""

    if value is None:
        return None

    roles = list(extra_roles or [])
    node_id = ""
    if isinstance(value, GraphNodeRef):
        if value.roles:
            roles.extend(value.roles)
        roles = _dedupe_strings(roles)
        return GraphNodeRef(
            node_id=value.node_id,
            asset=value.asset,
            field=value.field,
            roles=tuple(roles),
        )
    if isinstance(value, dict):
        raw_node_id = str(value.get("node_id") or "").strip()
        raw_asset = str(
            value.get("asset")
            or value.get("ticker")
            or ""
        ).strip()
        raw_field = str(value.get("field") or default_field).strip().lower() or default_field
        roles.extend(_read_role_values(value))
        if raw_node_id:
            node_id = normalize_graph_node_id(raw_node_id, default_field=default_field)
        elif raw_asset:
            node_id = normalize_graph_node_id(f"{raw_asset}.{raw_field}", default_field=default_field)
    else:
        raw = str(value).strip()
        if raw:
            node_id = normalize_graph_node_id(raw, default_field=default_field)

    if not node_id:
        return None
    asset, field = split_graph_node_id(node_id)
    return GraphNodeRef(
        node_id=node_id,
        asset=asset,
        field=field,
        roles=tuple(_dedupe_strings(roles)),
    )


def coerce_graph_node_refs(
    values: Iterable[Any],
    *,
    default_field: str = "price",
    extra_roles: Iterable[str] | None = None,
) -> list[GraphNodeRef]:
    """Normalize an iterable into unique graph nodes, preserving first occurrence."""

    refs: list[GraphNodeRef] = []
    seen: set[str] = set()
    for value in values:
        ref = coerce_graph_node_ref(
            value,
            default_field=default_field,
            extra_roles=extra_roles,
        )
        if ref is None or ref.node_id in seen:
            continue
        refs.append(ref)
        seen.add(ref.node_id)
    return refs


def graph_node_assets(values: Iterable[Any]) -> list[str]:
    """Return unique assets from an iterable of graph-node-like values."""

    assets: list[str] = []
    seen: set[str] = set()
    for ref in coerce_graph_node_refs(values):
        if ref.asset in seen:
            continue
        assets.append(ref.asset)
        seen.add(ref.asset)
    return assets


def graph_node_runtime_field(value: GraphNodeRef | str) -> str:
    """Map a graph field to the runtime bar field used inside ``DecisionContext``."""

    field = value.field if isinstance(value, GraphNodeRef) else str(value or "").strip().lower()
    if field == "volume":
        return "volume"
    return "close"


def graph_node_label(value: Any, *, include_roles: bool = False) -> str:
    """Render a human-readable label for a node-like value."""

    ref = coerce_graph_node_ref(value)
    if ref is None:
        return ""
    return ref.label(include_roles=include_roles)


def normalize_graph_node_id(value: str, *, default_field: str = "price") -> str:
    """Normalize a public graph node id without importing the optional plugin layer."""

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Ticker or node id cannot be empty.")
    if default_field not in SUPPORTED_GRAPH_FIELDS:
        raise ValueError(f"Unsupported field '{default_field}'.")

    normalized = raw.upper()
    ticker, dot, suffix = normalized.rpartition(".")
    if dot:
        field = suffix.lower()
        if field not in SUPPORTED_GRAPH_FIELDS:
            raise ValueError("Abel node ids must end with .price or .volume.")
        return f"{ticker}.{field}"

    ticker, underscore, suffix = normalized.rpartition("_")
    if underscore:
        field = suffix.lower()
        if field == "close":
            return f"{ticker}.price"
        if field == "volume":
            return f"{ticker}.volume"
        raise ValueError("Abel node ids must use .price or .volume.")

    if normalized in CRYPTO_ALIASES:
        normalized = f"{normalized}USD"
    return f"{normalized}.{default_field}"


def split_graph_node_id(node_id: str) -> tuple[str, str]:
    ticker, _, field = normalize_graph_node_id(node_id).rpartition(".")
    return ticker, field


def _read_role_values(value: dict[str, Any]) -> list[str]:
    roles = value.get("roles")
    if isinstance(roles, list):
        return _dedupe_strings(roles)
    discovery_roles = value.get("discovery_roles")
    if isinstance(discovery_roles, list):
        return _dedupe_strings(discovery_roles)
    if isinstance(roles, str):
        return _dedupe_strings([roles])
    if isinstance(discovery_roles, str):
        return _dedupe_strings([discovery_roles])
    return []


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
    return ordered
