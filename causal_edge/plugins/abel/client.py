"""Small Abel CAP client used by the optional plugin."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

import requests

from causal_edge.plugins.abel.credentials import resolve_cap_base_url

CAP_VERSION = "0.2.2"
DEFAULT_GRAPH_ID = "abel-main"
DEFAULT_GRAPH_VERSION = "CausalNodeV3"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_RETRY_ATTEMPTS = 4

CRYPTO_ALIASES = {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX"}
SUPPORTED_FIELDS = {"price", "volume"}
SUPPORTED_MARKET_FIELDS = {"open", "high", "low", "close", "volume"}


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


def _normalize_market_fields(fields: list[str] | None) -> list[str]:
    requested = fields or ["open", "high", "low", "close", "volume"]
    normalized = []
    seen = set()
    for field in requested:
        name = str(field).strip().lower()
        if name in {"timestamp", "symbol", "date"}:
            continue
        if name not in SUPPORTED_MARKET_FIELDS:
            continue
        if name not in seen:
            seen.add(name)
            normalized.append(name)
    if not normalized:
        return ["open", "high", "low", "close", "volume"]
    return normalized


class AbelClient:
    def __init__(
        self,
        *,
        cap_base_url: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.cap_base_url = (cap_base_url or resolve_cap_base_url()).rstrip("/")
        self.session = session or requests.Session()

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
                "fields": _normalize_market_fields(fields),
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
        return self._post_json(
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
                },
            },
            headers=headers,
        )

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
