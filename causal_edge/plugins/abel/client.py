"""Small Abel CAP client used by the optional plugin."""

from __future__ import annotations

from typing import Any

import requests

from causal_edge.plugins.abel.credentials import resolve_cap_base_url

CRYPTO_ALIASES = {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX"}
SUPPORTED_FIELDS = {"price", "volume"}


def normalize_public_node_id(value: str, *, default_field: str = "price") -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("Ticker or node id cannot be empty.")
    if default_field not in SUPPORTED_FIELDS:
        raise ValueError(f"Unsupported field '{default_field}'.")

    normalized = raw.upper()
    ticker, dot, suffix = normalized.rpartition(".")
    if dot:
        field = suffix.lower()
        if field not in SUPPORTED_FIELDS:
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


def split_public_node_id(node_id: str) -> tuple[str, str]:
    ticker, _, field = normalize_public_node_id(node_id).rpartition(".")
    return ticker, field


def _serialize_timestamp(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


class AbelClient:
    def __init__(
        self,
        *,
        cap_base_url: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.cap_base_url = (cap_base_url or resolve_cap_base_url()).rstrip("/")
        self.session = session or requests.Session()

    def discover_parents(self, *, node_id: str, limit: int, api_key: str) -> list[dict[str, Any]]:
        payload = self._post_cap(
            verb="traverse.parents",
            params={"node_id": normalize_public_node_id(node_id), "top_k": min(limit, 20)},
            api_key=api_key,
        )
        return _extract_items(payload)

    def markov_blanket(self, *, node_id: str, limit: int, api_key: str) -> list[dict[str, Any]]:
        payload = self._post_cap(
            verb="graph.markov_blanket",
            params={"node_id": normalize_public_node_id(node_id), "max_neighbors": min(limit, 20)},
            api_key=api_key,
        )
        return _extract_items(payload)

    def fetch_bars(
        self,
        *,
        symbols: list[str],
        start: str | None,
        end: str | None,
        timeframe: str,
        limit: int | None,
        fields: list[str] | None,
        api_key: str,
    ) -> Any:
        payload = self._post_market(
            endpoint="day_bar",
            body={
                "symbols": [
                    normalize_public_node_id(symbol, default_field="price").split(".")[0]
                    for symbol in symbols
                ],
                "start": _serialize_timestamp(start),
                "end": _serialize_timestamp(end),
                "timeframe": timeframe,
                "limit": limit,
                "fields": fields or ["open", "high", "low", "close", "volume"],
            },
            api_key=api_key,
        )
        items = payload.get("data") or payload.get("result") or []
        if isinstance(items, dict):
            items = items.get("items") or items.get("bars") or []
        return items

    def _post_cap(self, *, verb: str, params: dict[str, Any], api_key: str) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": api_key
            if api_key.lower().startswith("bearer ")
            else f"Bearer {api_key}",
        }
        response = self.session.post(
            f"{self.cap_base_url}/cap",
            json={"verb": verb, "params": params},
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    def _post_market(self, *, endpoint: str, body: dict[str, Any], api_key: str) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": api_key
            if api_key.lower().startswith("bearer ")
            else f"Bearer {api_key}",
        }
        response = self.session.post(
            f"{self.cap_base_url}/market/{endpoint}",
            json=body,
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        return response.json()


def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result")
    if isinstance(result, dict):
        for key in ("items", "nodes", "neighbors", "markov_blanket"):
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []
