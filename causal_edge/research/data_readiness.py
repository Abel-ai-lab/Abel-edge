"""Structured data-readiness probes for research workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from causal_edge.plugins.abel.prices import fetch_bars

DEFAULT_PROBE_LIMIT = 500
TARGET_CONFIRMATION_LIMITS = (1000, 2000)


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
    target_probe_meta: dict[str, object] | None = None
    target_boundary: dict[str, object] | None = None

    for item in candidates:
        ticker = item["ticker"]
        roles = item["roles"]
        is_target = "target" in roles
        try:
            if is_target:
                bars, target_probe_meta = _probe_target_bars(
                    ticker=ticker,
                    requested_start=requested_start,
                    end=end,
                    limit=limit,
                    env_path=env_path,
                )
            else:
                bars = _fetch_probe_bars(
                    ticker=ticker,
                    start=requested_start,
                    end=end,
                    limit=limit,
                    env_path=env_path,
                )
        except Exception as exc:
            results.append(
                _error_result(
                    ticker=ticker,
                    roles=roles,
                    note=str(exc),
                )
            )
            continue

        result = _result_from_bars(
            ticker=ticker,
            roles=roles,
            bars=bars,
            requested_start=requested_start,
            probe_limit=limit,
            is_target=is_target,
        )
        results.append(result)
        if is_target:
            target_boundary = _target_boundary_from_result(
                result=result,
                requested_start=requested_start,
                probe_meta=target_probe_meta or _default_probe_meta(limit),
            )

    summary = _build_summary(results)
    coverage_hints = _coverage_hints(results, requested_start=requested_start, target_boundary=target_boundary)
    probe = _probe_summary(limit, target_probe_meta)
    return {
        "source": "discovery_json" if payload is not None else "tickers",
        "discovery_path": str(discovery_json.resolve()) if discovery_json is not None else None,
        "probe": probe,
        "requested_window": {"start": requested_start, "end": end},
        "coverage_hints": coverage_hints,
        "target_boundary": target_boundary,
        "results": results,
        "summary": summary,
    }


def render_data_verification_report(report: dict) -> str:
    requested = report.get("requested_window") or {}
    summary = report.get("summary") or {}
    probe = report.get("probe") or {}
    lines = [
        "Research Data Verification",
        f"Source: {report.get('source', 'unknown')}",
        f"Probe limit: {probe.get('limit', 'unknown')}",
        f"Requested window: {requested.get('start', 'latest')} -> {requested.get('end', 'latest')}",
        (
            "Summary: "
            f"{summary.get('ticker_count', 0)} tickers, "
            f"{summary.get('usable_count', 0)} usable, "
            f"{summary.get('start_covered_count', 0)} start-covered, "
            f"{summary.get('partial_window_count', 0)} partial, "
            f"{summary.get('no_data_count', 0)} no-data, "
            f"{summary.get('error_count', 0)} error"
        ),
        f"Probe semantics: left_boundary={probe.get('left_boundary_confidence', 'unknown')}",
        "",
    ]
    target_boundary = report.get("target_boundary") or {}
    if target_boundary:
        lines.append(
            "Target boundary: "
            f"classification={target_boundary.get('classification', 'unknown')}, "
            f"observed_first={target_boundary.get('observed_first_timestamp') or 'n/a'}"
        )
        lines.append("")
    coverage_hints = report.get("coverage_hints") or {}
    target_safe_start = coverage_hints.get("target_safe_start")
    dense_overlap_hint_start = coverage_hints.get("dense_overlap_hint_start")
    if target_safe_start or dense_overlap_hint_start:
        lines.append(
            "Coverage hints: "
            f"target_safe={target_safe_start or 'n/a'}, "
            f"dense_overlap={dense_overlap_hint_start or 'n/a'}"
        )
        lines.append("")
    for item in report.get("results") or []:
        roles = ",".join(item.get("roles") or [])
        lines.append(
            f"- {item.get('ticker')}: {item.get('status')} "
            f"(rows={item.get('rows', 0)}, first={item.get('observed_first_timestamp') or 'n/a'}, "
            f"last={item.get('observed_last_timestamp') or 'n/a'}, "
            f"covers_start={item.get('covers_requested_start')}, "
            f"boundary={item.get('left_boundary_confidence', 'unknown')}, "
            f"roles={roles or 'unknown'})"
        )
    return "\n".join(lines)


def _fetch_probe_bars(
    *,
    ticker: str,
    start: str | None,
    end: str | None,
    limit: int,
    env_path: str,
) -> pd.DataFrame:
    return fetch_bars(
        symbols=[ticker],
        start=start,
        end=end,
        timeframe="1d",
        limit=limit,
        fields=["close"],
        config={"env_path": env_path},
    )


def _probe_target_bars(
    *,
    ticker: str,
    requested_start: str | None,
    end: str | None,
    limit: int,
    env_path: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    bars = _fetch_probe_bars(
        ticker=ticker,
        start=requested_start,
        end=end,
        limit=limit,
        env_path=env_path,
    )
    if bars.empty or requested_start is None:
        return bars, _default_probe_meta(limit)

    rows = len(bars)
    observed_first = _observed_first_timestamp(bars)
    if observed_first is None or observed_first <= requested_start or rows < limit:
        return bars, _default_probe_meta(limit)

    last_bars = bars
    last_limit = limit
    for confirm_limit in TARGET_CONFIRMATION_LIMITS:
        if confirm_limit <= last_limit:
            continue
        candidate = _fetch_probe_bars(
            ticker=ticker,
            start=requested_start,
            end=end,
            limit=confirm_limit,
            env_path=env_path,
        )
        if candidate.empty:
            break
        last_bars = candidate
        last_limit = confirm_limit
        observed_first = _observed_first_timestamp(candidate)
        if observed_first is None or observed_first <= requested_start or len(candidate) < confirm_limit:
            break

    meta = {
        "base_limit": limit,
        "final_limit": last_limit,
        "confirmation_attempted": last_limit != limit,
        "left_boundary_confidence": _left_boundary_confidence(
            rows=len(last_bars),
            probe_limit=last_limit,
            observed_first_timestamp=_observed_first_timestamp(last_bars),
            requested_start=requested_start,
        ),
    }
    return last_bars, meta


def _result_from_bars(
    *,
    ticker: str,
    roles: list[str],
    bars: pd.DataFrame,
    requested_start: str | None,
    probe_limit: int,
    is_target: bool,
) -> dict[str, object]:
    if bars.empty:
        return {
            "ticker": ticker,
            "roles": roles,
            "status": "no_data",
            "usable": False,
            "covers_requested_start": False,
            "rows": 0,
            "is_target": is_target,
            "observed_first_timestamp": None,
            "observed_last_timestamp": None,
            "left_boundary_confidence": "none",
            "note": "No bars returned for the requested window.",
        }

    observed_first = _observed_first_timestamp(bars)
    observed_last = _observed_last_timestamp(bars)
    covers_requested_start = _covers_requested_start(observed_first, requested_start)
    status = "start_covered" if covers_requested_start else "partial_window"
    left_boundary_confidence = _left_boundary_confidence(
        rows=len(bars),
        probe_limit=probe_limit,
        observed_first_timestamp=observed_first,
        requested_start=requested_start,
    )
    note = (
        "Ticker has bars covering the requested start."
        if covers_requested_start
        else "Ticker has bars, but the observed history starts after the requested window."
    )
    if left_boundary_confidence != "confirmed" and not covers_requested_start:
        note += " Probe depth may still be truncating the left boundary."
    return {
        "ticker": ticker,
        "roles": roles,
        "status": status,
        "usable": True,
        "covers_requested_start": covers_requested_start,
        "rows": int(len(bars)),
        "is_target": is_target,
        "observed_first_timestamp": observed_first,
        "observed_last_timestamp": observed_last,
        "left_boundary_confidence": left_boundary_confidence,
        "note": note,
    }


def _error_result(*, ticker: str, roles: list[str], note: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "roles": roles,
        "status": "error",
        "usable": False,
        "covers_requested_start": False,
        "rows": 0,
        "is_target": "target" in roles,
        "observed_first_timestamp": None,
        "observed_last_timestamp": None,
        "left_boundary_confidence": "none",
        "note": note,
    }


def _target_boundary_from_result(
    *,
    result: dict[str, object],
    requested_start: str | None,
    probe_meta: dict[str, object],
) -> dict[str, object]:
    observed_first = result.get("observed_first_timestamp")
    observed_last = result.get("observed_last_timestamp")
    rows = int(result.get("rows", 0) or 0)
    final_limit = int(probe_meta.get("final_limit", 0) or 0)
    classification = "unknown_no_requested_start"
    if requested_start:
        if result.get("covers_requested_start"):
            classification = "confirmed_before_requested_start"
        elif rows == 0:
            classification = "confirmed_after_requested_start"
        elif rows < final_limit:
            classification = "confirmed_after_requested_start"
        else:
            classification = "unknown_probe_truncated"
    return {
        "ticker": result.get("ticker"),
        "requested_start": requested_start,
        "observed_first_timestamp": observed_first,
        "observed_last_timestamp": observed_last,
        "classification": classification,
        "left_boundary_confidence": result.get("left_boundary_confidence", "unknown"),
        "final_probe_limit": final_limit,
        "confirmation_attempted": bool(probe_meta.get("confirmation_attempted", False)),
    }


def _build_summary(results: list[dict[str, object]]) -> dict[str, int]:
    return {
        "ticker_count": len(results),
        "usable_count": sum(1 for item in results if item["usable"]),
        "start_covered_count": sum(
            1 for item in results if item["status"] == "start_covered"
        ),
        "partial_window_count": sum(1 for item in results if item["status"] == "partial_window"),
        "no_data_count": sum(1 for item in results if item["status"] == "no_data"),
        "error_count": sum(1 for item in results if item["status"] == "error"),
    }


def _coverage_hints(
    results: list[dict[str, object]],
    *,
    requested_start: str | None,
    target_boundary: dict[str, object] | None,
) -> dict[str, object]:
    usable = [item for item in results if item.get("usable")]
    usable_starts = [
        str(item.get("observed_first_timestamp"))
        for item in usable
        if isinstance(item.get("observed_first_timestamp"), str) and item.get("observed_first_timestamp")
    ]
    target_safe_start = None
    if target_boundary:
        classification = target_boundary.get("classification")
        observed_first = target_boundary.get("observed_first_timestamp")
        if classification == "confirmed_before_requested_start":
            target_safe_start = requested_start
        elif isinstance(observed_first, str) and observed_first:
            target_safe_start = observed_first
    dense_overlap_hint_start = max(usable_starts) if usable_starts else None
    return {
        "target_safe_start": target_safe_start,
        "dense_overlap_hint_start": dense_overlap_hint_start,
    }

def _probe_summary(limit: int, target_probe_meta: dict[str, object] | None) -> dict[str, object]:
    probe_meta = target_probe_meta or _default_probe_meta(limit)
    return {
        "strategy": "target_boundary_confirm",
        "limit": limit,
        "target_final_limit": int(probe_meta.get("final_limit", limit) or limit),
        "target_confirmation_attempted": bool(probe_meta.get("confirmation_attempted", False)),
        "left_boundary_confidence": probe_meta.get("left_boundary_confidence", "unknown"),
    }


def _default_probe_meta(limit: int) -> dict[str, object]:
    return {
        "base_limit": limit,
        "final_limit": limit,
        "confirmation_attempted": False,
        "left_boundary_confidence": "unknown",
    }


def _observed_first_timestamp(bars: pd.DataFrame) -> str | None:
    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        return None
    return timestamps.min().date().isoformat()


def _observed_last_timestamp(bars: pd.DataFrame) -> str | None:
    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        return None
    return timestamps.max().date().isoformat()


def _left_boundary_confidence(
    *,
    rows: int,
    probe_limit: int,
    observed_first_timestamp: str | None,
    requested_start: str | None,
) -> str:
    if rows <= 0 or observed_first_timestamp is None:
        return "none"
    if requested_start and observed_first_timestamp <= requested_start:
        return "confirmed"
    if probe_limit <= 0:
        return "observed"
    if rows < probe_limit:
        return "confirmed"
    return "observed"


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

    return [{"ticker": ticker, "roles": sorted(roles)} for ticker, roles in sorted(combined.items())]


def _covers_requested_start(first_timestamp: str | None, requested_start: str | None) -> bool:
    if requested_start is None or first_timestamp is None:
        return first_timestamp is not None
    return first_timestamp <= requested_start
