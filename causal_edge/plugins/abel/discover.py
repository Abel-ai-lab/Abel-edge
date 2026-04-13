"""Abel discovery helpers used by the CLI."""

from __future__ import annotations

from typing import Any

from causal_edge.plugins.abel.client import AbelClient, split_public_node_id
from causal_edge.plugins.abel.credentials import MissingAbelApiKeyError, require_api_key


def discover_graph_nodes(
    ticker: str,
    *,
    mode: str = "parents",
    limit: int = 10,
    env_path: str = ".env",
    client: AbelClient | None = None,
) -> str:
    try:
        api_key = require_api_key(env_path=env_path)
    except MissingAbelApiKeyError as e:
        raise MissingAbelApiKeyError(
            f"{e} Optionally set ABEL_CAP_BASE_URL to target a non-default CAP endpoint."
        ) from e
    abel = client or AbelClient()
    limit = min(max(limit, 1), 20)

    if mode == "parents":
        items = abel.discover_parents(node_id=ticker, limit=limit, api_key=api_key)
        return _render_parents(items)
    if mode == "mb":
        items = abel.markov_blanket(node_id=ticker, limit=limit, api_key=api_key)
        return _render_markov_blanket(items)
    raise ValueError(f"Unsupported mode '{mode}'.")


def _render_parents(items: list[dict[str, Any]]) -> str:
    lines = ["parents:"]
    for item in items[:20]:
        node_id = _pick_node_id(item)
        if not node_id:
            continue
        ticker, field = split_public_node_id(node_id)
        lines.append(f"  - ticker: {ticker}")
        lines.append(f"    field: {field}")
    return "\n".join(lines)


def _render_markov_blanket(items: list[dict[str, Any]]) -> str:
    lines = ["markov_blanket:"]
    for item in items[:20]:
        node_id = _pick_node_id(item)
        if not node_id:
            continue
        ticker, field = split_public_node_id(node_id)
        roles = _pick_roles(item)
        lines.append(f"  - ticker: {ticker}")
        lines.append(f"    field: {field}")
        lines.append(f"    roles: [{', '.join(roles)}]")
    return "\n".join(lines)


def _pick_node_id(item: dict[str, Any]) -> str:
    for key in ("node_id", "id", "name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _pick_roles(item: dict[str, Any]) -> list[str]:
    for key in ("roles", "role", "relationship", "type"):
        value = item.get(key)
        if isinstance(value, list):
            return [str(role) for role in value]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    return ["neighbor"]
