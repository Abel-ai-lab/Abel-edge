"""Explicit Abel OAuth helpers for the optional CLI login command."""

from __future__ import annotations

import time
import webbrowser
from typing import Any, Callable

import requests

from causal_edge.plugins.abel.credentials import (
    normalize_api_key,
    persist_env_value,
    resolve_api_key,
    resolve_auth_base_url,
)

Notifier = Callable[[str], None]


class AbelLoginTimeoutError(RuntimeError):
    """Raised when interactive authorization does not complete in time."""


class AbelAuthClient:
    def __init__(
        self,
        *,
        auth_base_url: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.auth_base_url = (auth_base_url or resolve_auth_base_url()).rstrip("/")
        self.session = session or requests.Session()

    def authorize_agent(self) -> dict[str, Any]:
        response = self.session.get(
            f"{self.auth_base_url}/web/credentials/oauth/google/authorize/agent",
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or {}
        if "authUrl" not in data:
            raise RuntimeError("Abel OAuth authorize endpoint did not return authUrl.")
        return data

    def poll_authorization_result(
        self,
        *,
        result_url: str | None,
        poll_token: str | None,
        poll_interval: float,
        timeout_seconds: int,
        notify: Notifier | None = None,
    ) -> dict[str, Any]:
        if result_url:
            url = result_url
        elif poll_token:
            url = f"{self.auth_base_url}/web/credentials/oauth/google/result?pollToken={poll_token}"
        else:
            raise RuntimeError("OAuth handoff missing resultUrl and pollToken.")

        deadline = time.monotonic() + timeout_seconds
        polls = 0
        while True:
            if time.monotonic() > deadline:
                raise AbelLoginTimeoutError(
                    "Timed out waiting for Abel browser authorization to complete."
                )
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
            if notify is not None and polls % 15 == 0:
                notify("Still waiting for browser authorization... (Ctrl+C to cancel)")
            time.sleep(poll_interval)


def login_with_oauth(
    *,
    env_path: str = ".env",
    open_browser: bool = True,
    timeout_seconds: int = 300,
    poll_interval: float = 2.0,
    force: bool = False,
    notify: Notifier | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    existing_api_key = None if force else resolve_api_key(env_path=env_path)
    if existing_api_key:
        return {
            "status": "already_configured",
            "api_key": existing_api_key,
            "env_path": env_path,
            "auth_url": None,
            "opened_browser": False,
            "stored": False,
        }

    auth = AbelAuthClient(session=session, auth_base_url=resolve_auth_base_url(env_path=env_path))
    handoff = auth.authorize_agent()
    auth_url = handoff["authUrl"]
    if notify is not None:
        notify(f"Open this URL to authorize Abel access:\n{auth_url}")

    opened_browser = False
    if open_browser:
        try:
            opened_browser = bool(webbrowser.open(auth_url))
        except Exception:
            opened_browser = False

    result = auth.poll_authorization_result(
        result_url=handoff.get("resultUrl"),
        poll_token=handoff.get("pollToken"),
        poll_interval=poll_interval,
        timeout_seconds=timeout_seconds,
        notify=notify,
    )
    api_key = normalize_api_key(result.get("apiKey"))
    if not api_key:
        raise RuntimeError("Abel authorization succeeded but returned no apiKey.")

    persist_env_value(env_path=env_path, key="ABEL_API_KEY", value=api_key)
    return {
        "status": "authorized",
        "api_key": api_key,
        "env_path": env_path,
        "auth_url": auth_url,
        "opened_browser": opened_browser,
        "stored": True,
    }
