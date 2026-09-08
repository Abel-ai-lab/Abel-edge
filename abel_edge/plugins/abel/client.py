"""Small Abel CAP client used by the optional plugin."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from typing import Any
import uuid

import requests

from abel_edge.plugins.abel.credentials import resolve_cap_base_url
from abel_edge.plugins.abel.graph_driver import normalize_graph_query_node_id
from abel_edge.plugins.abel import node_records_client
from abel_edge.plugins.abel.market_contract import (
    normalize_market_fields as _normalize_market_fields,
    normalize_market_symbol as _normalize_market_symbol,
    normalize_public_node_id,
    split_public_node_id as split_public_node_id,
)

CAP_VERSION = "0.2.2"
DEFAULT_GRAPH_ID = "abel-main"
DEFAULT_GRAPH_VERSION = "CausalNodeV3"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_RETRY_ATTEMPTS = 4

def _serialize_daily_bar_bound(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, date):
        return value.isoformat()
    else:
        text = str(value)
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            pass
        try:
            timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
    if timestamp.utcoffset() is not None:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.date().isoformat()


class AbelClient:
    def __init__(
        self,
        *,
        cap_base_url: str | None = None,
        env_path: str | Path = ".env",
        session: requests.Session | None = None,
    ) -> None:
        self.cap_base_url = (cap_base_url or resolve_cap_base_url(env_path=env_path)).rstrip("/")
        self.session = session or requests.Session()
        self._last_cap_provenance: dict[str, Any] = {}

    def _post_json(self, *, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        last_exc = None
        for attempt in range(1, DEFAULT_RETRY_ATTEMPTS + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=DEFAULT_TIMEOUT_SECONDS,
                )
                status_code = getattr(response, "status_code", 200)
                if status_code == 429 and attempt < DEFAULT_RETRY_ATTEMPTS:
                    time.sleep(_retry_delay_seconds(response.headers.get("Retry-After"), attempt))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                if not _should_fallback_to_curl(exc):
                    raise
                return _post_with_curl(url=url, payload=payload, headers=headers)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Abel CAP request exhausted retries without a response.")

    def discover_parents(
        self,
        *,
        node_id: str,
        limit: int,
        api_key: str,
        graph_ref: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        payload = self._post_cap(
            verb="traverse.parents",
            params={
                "node_id": _normalize_graph_node_id(node_id, graph_ref=graph_ref),
                "top_k": min(limit, 20),
            },
            api_key=api_key,
            graph_ref=graph_ref,
        )
        return _extract_items(payload)

    def markov_blanket(
        self,
        *,
        node_id: str,
        limit: int,
        api_key: str,
        graph_ref: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        payload = self._post_cap(
            verb="graph.markov_blanket",
            params={
                "node_id": _normalize_graph_node_id(node_id, graph_ref=graph_ref),
                "max_neighbors": min(limit, 20),
            },
            api_key=api_key,
            graph_ref=graph_ref,
        )
        return _extract_items(payload)

    def cap_methods(self, *, api_key: str) -> list[dict[str, Any]]:
        payload = self._post_cap(
            verb="meta.methods",
            params={"detail": "full", "include_examples": False},
            api_key=api_key,
            graph_ref=None,
        )
        result = payload.get("result") or {}
        methods = result.get("methods") if isinstance(result, dict) else []
        return [item for item in (methods or []) if isinstance(item, dict)]

    def graph_provenance(self) -> dict[str, Any]:
        return deepcopy(self._last_cap_provenance)

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
                    _normalize_market_symbol(symbol)
                    for symbol in symbols
                ],
                "start": _serialize_daily_bar_bound(start),
                "end": _serialize_daily_bar_bound(end),
                "timeframe": timeframe,
                "limit": limit,
                "fields": _normalize_market_fields(fields),
            },
            api_key=api_key,
        )
        items = payload.get("data") or payload.get("result") or []
        if isinstance(items, dict):
            items = items.get("items") or items.get("bars") or []
        return items

    def fetch_node_series(
        self,
        *,
        node_id: str,
        start: str | None,
        end: str | None,
        limit: int | None,
        api_key: str,
    ) -> Any:
        """Fetch one unadjusted scalar series by its exact V4 node id."""

        return node_records_client.fetch_all_node_series(
            fetch_page=self.fetch_node_series_page,
            node_id=node_id,
            start=start,
            end=end,
            limit=limit,
            api_key=api_key,
        )

    def fetch_node_series_page(
        self,
        *,
        node_id: str,
        start: str | None,
        end: str | None,
        limit: int | None,
        cursor_date: str | None,
        api_key: str,
    ) -> dict[str, Any]:
        """Fetch one exact CAP scalar-series page for a V4 node."""

        return node_records_client.fetch_node_series_page(
            post_market=self._post_market,
            node_id=node_id,
            start=start,
            end=end,
            limit=limit,
            cursor_date=cursor_date,
            api_key=api_key,
        )

    def _post_cap(
        self,
        *,
        verb: str,
        params: dict[str, Any],
        api_key: str,
        graph_ref: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": api_key
            if api_key.lower().startswith("bearer ")
            else f"Bearer {api_key}",
        }
        payload = self._post_json(
            url=f"{self.cap_base_url}/cap",
            payload={
                "cap_version": CAP_VERSION,
                "request_id": str(uuid.uuid4()),
                "verb": verb,
                "params": params,
                "context": {
                    "graph_ref": {
                        "graph_id": DEFAULT_GRAPH_ID,
                        "graph_version": DEFAULT_GRAPH_VERSION,
                    }
                    if graph_ref is None
                    else deepcopy(graph_ref)
                },
            },
            headers=headers,
        )
        provenance = payload.get("provenance")
        self._last_cap_provenance = (
            deepcopy(provenance) if isinstance(provenance, dict) else {}
        )
        return payload

    def _post_market(self, *, endpoint: str, body: dict[str, Any], api_key: str) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": api_key
            if api_key.lower().startswith("bearer ")
            else f"Bearer {api_key}",
        }
        return self._post_json(
            url=f"{self.cap_base_url}/market/{endpoint}",
            payload=body,
            headers=headers,
        )


def _should_fallback_to_curl(exc: requests.exceptions.ConnectionError) -> bool:
    if sys.platform != "win32":
        return False
    if shutil.which("curl.exe") is None:
        return False
    message = str(exc).lower()
    return "connectionreseterror" in message or "10054" in message or "connection aborted" in message


def _retry_delay_seconds(retry_after: str | None, attempt: int) -> float:
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass
    return float(min(2 ** (attempt - 1), 8))


def _post_with_curl(*, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    curl_path = shutil.which("curl.exe")
    if curl_path is None:
        raise RuntimeError("curl.exe is required for CAP fallback transport on Windows.")

    command = [
        curl_path,
        "--silent",
        "--show-error",
        "--fail-with-body",
        "-X",
        "POST",
        url,
        "--connect-timeout",
        str(DEFAULT_TIMEOUT_SECONDS),
        "--max-time",
        str(DEFAULT_TIMEOUT_SECONDS),
        "--retry",
        "3",
        "--retry-all-errors",
        "--retry-delay",
        "1",
        "--http1.1",
        "--data-binary",
        json.dumps(payload),
    ]
    for key, value in headers.items():
        command.extend(["-H", f"{key}: {value}"])

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "curl fallback failed").strip()
        raise RuntimeError(f"CAP curl fallback failed: {stderr}")
    return json.loads(result.stdout)


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


def _normalize_graph_node_id(
    value: str,
    *,
    graph_ref: dict[str, str] | None,
) -> str:
    return normalize_graph_query_node_id(
        value,
        graph_version=str(
            (graph_ref or {}).get("graph_version") or DEFAULT_GRAPH_VERSION
        ),
        legacy_normalizer=normalize_public_node_id,
    )
