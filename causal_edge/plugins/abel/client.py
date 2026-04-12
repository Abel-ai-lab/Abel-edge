"""Small Abel OAuth + CAP client used by the optional plugin."""

from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path
from typing import Any

import requests

DEFAULT_OAUTH_BASE_URL = "https://api-sit.abel.ai/echo"
DEFAULT_CAP_BASE_URL = "https://cap-sit.abel.ai/api"
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


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def persist_api_key(path: str | Path, api_key: str) -> None:
    env_path = Path(path)
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    updated = False
    for index, line in enumerate(lines):
        if line.startswith("ABEL_API_KEY="):
            lines[index] = f"ABEL_API_KEY={api_key}"
            updated = True
            break
    if not updated:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"ABEL_API_KEY={api_key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class AbelClient:
    def __init__(
        self,
        *,
        oauth_base_url: str = DEFAULT_OAUTH_BASE_URL,
        cap_base_url: str = DEFAULT_CAP_BASE_URL,
        session: requests.Session | None = None,
    ) -> None:
        self.oauth_base_url = oauth_base_url.rstrip("/")
        self.cap_base_url = cap_base_url.rstrip("/")
        self.session = session or requests.Session()

    def ensure_api_key(self, *, env_path: str | Path = ".env", poll_interval: float = 2.0) -> str:
        load_env_file(env_path)
        token = (os.getenv("ABEL_API_KEY") or os.getenv("CAP_API_KEY") or "").strip()
        if token:
            return token.removeprefix("Bearer ").strip()

        auth_data = self._authorize_agent()
        auth_url = auth_data["authUrl"]
        result_url = auth_data.get("resultUrl")
        poll_token = auth_data.get("pollToken")
        click_message = (
            f"Open this URL to authorize Abel access:\n{auth_url}\n"
            "After completing browser authorization, press Enter here to continue."
        )
        print(click_message)
        try:
            webbrowser.open(auth_url)
        except Exception:
            pass
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                input()
            except EOFError:
                pass

        result = self._poll_authorization_result(
            result_url=result_url, poll_token=poll_token, poll_interval=poll_interval
        )
        api_key = result.get("apiKey", "").strip()
        if not api_key:
            raise RuntimeError("Abel authorization succeeded but returned no apiKey.")
        os.environ["ABEL_API_KEY"] = api_key
        persist_api_key(env_path, api_key)
        return api_key

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
                "start": start,
                "end": end,
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

    def _authorize_agent(self) -> dict[str, Any]:
        response = self.session.get(
            f"{self.oauth_base_url}/web/credentials/oauth/google/authorize/agent", timeout=20
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or {}
        if "authUrl" not in data:
            raise RuntimeError("Abel OAuth authorize endpoint did not return authUrl.")
        return data

    def _poll_authorization_result(
        self,
        *,
        result_url: str | None,
        poll_token: str | None,
        poll_interval: float,
    ) -> dict[str, Any]:
        import time

        if result_url:
            url = result_url
        elif poll_token:
            url = (
                f"{self.oauth_base_url}/web/credentials/oauth/google/result?pollToken={poll_token}"
            )
        else:
            raise RuntimeError("OAuth handoff missing resultUrl and pollToken.")

        polls = 0
        while True:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") or {}
            status = data.get("status")
            if status == "authorized":
                return data
            if status == "failed":
                raise RuntimeError(data.get("message") or "Abel authorization failed.")
            if status != "pending":
                raise RuntimeError(f"Unexpected Abel authorization status: {status!r}")
            polls += 1
            if polls % 15 == 0:
                print("  Still waiting for browser authorization... (Ctrl+C to cancel)")
            time.sleep(poll_interval)

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
