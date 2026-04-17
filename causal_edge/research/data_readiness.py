"""Structured data-readiness probes for research workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from causal_edge.plugins.abel.prices import fetch_bars

DEFAULT_PROBE_LIMIT = 500


def run_data_verification(
    *,
    tickers: list[str] | None = None,
    discovery_json: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = DEFAULT_PROBE_LIMIT,
    env_path: str = ".env",
) -> dict:
    payload = _load_discovery_payload(discovery_json) if discovery_json is not None else None
    requested_start = start or _start_from_discovery(payload)
    candidates = _candidate_tickers(payload, tickers or [])
    if not candidates:
        raise ValueError("No tickers provided. Pass --ticker ... or --discovery-json <path>.")

    results: list[dict[str, object]] = []
    for item in candidates:
        ticker = item["ticker"]
        roles = item["roles"]
        try:
            bars = fetch_bars(
                symbols=[ticker],
                start=requested_start,
                end=end,
                timeframe="1d",
                limit=limit,
                fields=["close"],
                config={"env_path": env_path},
            )
        except Exception as exc:
            results.append(
                {
                    "ticker": ticker,
                    "roles": roles,
                    "status": "error",
                    "usable": False,
                    "full_window": False,
                    "rows": 0,
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "note": str(exc),
                }
            )
            continue

        if bars.empty:
            results.append(
                {
                    "ticker": ticker,
                    "roles": roles,
                    "status": "no_data",
                    "usable": False,
                    "full_window": False,
                    "rows": 0,
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "note": "No bars returned for the requested window.",
                }
            )
            continue

        timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce").dropna()
        if timestamps.empty:
            first_timestamp = None
            last_timestamp = None
        else:
            first_timestamp = timestamps.min().date().isoformat()
            last_timestamp = timestamps.max().date().isoformat()

        full_window = _covers_requested_start(first_timestamp, requested_start)
        status = "full_window" if full_window else "partial_window"
        note = (
            "Ticker has bars covering the requested start."
            if full_window
            else "Ticker has bars, but history starts after the requested window."
        )
        results.append(
            {
                "ticker": ticker,
                "roles": roles,
                "status": status,
                "usable": True,
                "full_window": full_window,
                "rows": int(len(bars)),
                "first_timestamp": first_timestamp,
                "last_timestamp": last_timestamp,
                "note": note,
            }
        )

    summary = {
        "ticker_count": len(results),
        "usable_count": sum(1 for item in results if item["usable"]),
        "full_window_count": sum(1 for item in results if item["status"] == "full_window"),
        "partial_window_count": sum(1 for item in results if item["status"] == "partial_window"),
        "no_data_count": sum(1 for item in results if item["status"] == "no_data"),
        "error_count": sum(1 for item in results if item["status"] == "error"),
    }
    recommendations = _recommended_starts(results)
    return {
        "source": "discovery_json" if payload is not None else "tickers",
        "discovery_path": str(discovery_json.resolve()) if discovery_json is not None else None,
        "probe_limit": limit,
        "requested_window": {"start": requested_start, "end": end},
        "recommended_starts": recommendations,
        "results": results,
        "summary": summary,
    }


def render_data_verification_report(report: dict) -> str:
    requested = report.get("requested_window") or {}
    summary = report.get("summary") or {}
    lines = [
        "Research Data Verification",
        f"Source: {report.get('source', 'unknown')}",
        f"Probe limit: {report.get('probe_limit', 'unknown')}",
        f"Requested window: {requested.get('start', 'latest')} -> {requested.get('end', 'latest')}",
        (
            "Summary: "
            f"{summary.get('ticker_count', 0)} tickers, "
            f"{summary.get('usable_count', 0)} usable, "
            f"{summary.get('full_window_count', 0)} full-window, "
            f"{summary.get('partial_window_count', 0)} partial, "
            f"{summary.get('no_data_count', 0)} no-data, "
            f"{summary.get('error_count', 0)} error"
        ),
        "",
    ]
    recommendations = report.get("recommended_starts") or {}
    target_start = recommendations.get("target_recommended_start")
    common_start = recommendations.get("common_recommended_start")
    if target_start or common_start:
        lines.append(
            "Recommended starts: "
            f"target={target_start or 'n/a'}, common={common_start or 'n/a'}"
        )
        lines.append("")
    for item in report.get("results") or []:
        roles = ",".join(item.get("roles") or [])
        lines.append(
            f"- {item.get('ticker')}: {item.get('status')} "
            f"(rows={item.get('rows', 0)}, first={item.get('first_timestamp') or 'n/a'}, "
            f"last={item.get('last_timestamp') or 'n/a'}, roles={roles or 'unknown'})"
        )
    return "\n".join(lines)


def _load_discovery_payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid discovery JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Invalid discovery JSON: expected an object payload.")
    return payload


def _start_from_discovery(payload: dict | None) -> str | None:
    if not payload:
        return None
    backtest = payload.get("backtest") or {}
    if isinstance(backtest, dict):
        start = backtest.get("start")
        if start:
            return str(start)
    return None


def _candidate_tickers(payload: dict | None, explicit: list[str]) -> list[dict[str, object]]:
    combined: dict[str, set[str]] = {}

    def _remember(ticker: str, role: str) -> None:
        key = str(ticker or "").strip().upper()
        if not key:
            return
        combined.setdefault(key, set()).add(role)

    if payload:
        ticker = payload.get("ticker")
        if ticker:
            _remember(str(ticker), "target")
        for item in payload.get("parents") or []:
            _remember(str((item or {}).get("ticker", "")), "parent")
        for item in payload.get("blanket_new") or []:
            _remember(str((item or {}).get("ticker", "")), "blanket")
        for item in payload.get("children") or []:
            _remember(str((item or {}).get("ticker", "")), "child")

    for ticker in explicit:
        _remember(ticker, "explicit")

    return [
        {"ticker": ticker, "roles": sorted(roles)}
        for ticker, roles in sorted(combined.items())
    ]


def _covers_requested_start(first_timestamp: str | None, requested_start: str | None) -> bool:
    if requested_start is None or first_timestamp is None:
        return first_timestamp is not None
    return first_timestamp <= requested_start


def _recommended_starts(results: list[dict[str, object]]) -> dict[str, str | None]:
    target_start = None
    common_start = None

    usable_starts: list[str] = []
    for item in results:
        if not item.get("usable"):
            continue
        first_timestamp = item.get("first_timestamp")
        if isinstance(first_timestamp, str) and first_timestamp:
            usable_starts.append(first_timestamp)
        roles = item.get("roles") or []
        if "target" in roles and isinstance(first_timestamp, str) and first_timestamp:
            target_start = first_timestamp

    if usable_starts:
        common_start = max(usable_starts)
    return {
        "target_recommended_start": target_start,
        "common_recommended_start": common_start,
    }
