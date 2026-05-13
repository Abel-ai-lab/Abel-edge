"""CLI helpers for warming and inspecting the market-data cache."""

from __future__ import annotations

import json
from pathlib import Path

import click

from abel_edge.engine.adapter_registry import FeedLoadRequest, resolve_adapter
from abel_edge.engine.cache import cache_entry_for_request, load_cached_metadata, resolve_cache_root

WARM_CACHE_MAX_CACHE_AGE_SECONDS = 86400


def warm_cache_payload(
    *,
    symbols: list[str],
    adapter_name: str,
    start: str | None,
    end: str | None,
    timeframe: str,
    profile: str,
    limit: int | None,
    env_path: str | None,
    path: str | None,
) -> dict:
    adapter = resolve_adapter(adapter_name)
    results: list[dict] = []
    cache_root = resolve_cache_root()
    for symbol in symbols:
        normalized_symbol = str(symbol or "").strip().upper()
        options: dict[str, object] = {}
        if env_path:
            options["env_path"] = env_path
        if path:
            options["path"] = path
        options["max_cache_age_seconds"] = WARM_CACHE_MAX_CACHE_AGE_SECONDS
        request = FeedLoadRequest(
            adapter=adapter_name,
            kind="bars",
            symbol=normalized_symbol,
            field=None,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
            profile=profile,
            options=options,
            strategy_id=None,
            feed_name=f"warm-cache:{normalized_symbol}",
        )
        try:
            bars = adapter.load(request)
            entry = cache_entry_for_request(
                adapter=adapter_name,
                symbol=normalized_symbol,
                timeframe=timeframe,
                profile=profile,
                options=request.options,
            )
            metadata = load_cached_metadata(entry)
            results.append(
                {
                    "symbol": normalized_symbol,
                    "ok": True,
                    "row_count": int(len(bars)),
                    "available_range": metadata.get("available_range") or {},
                    "cache_key": metadata.get("cache_key"),
                    "data_path": metadata.get("data_path"),
                    "metadata_path": metadata.get("metadata_path"),
                }
            )
        except Exception as exc:  # pragma: no cover - surfaced to CLI users directly
            results.append(
                {
                    "symbol": normalized_symbol,
                    "ok": False,
                    "error": str(exc),
                }
            )
    return {
        "adapter": adapter_name,
        "path": path,
        "timeframe": timeframe,
        "profile": profile,
        "start": start,
        "end": end,
        "cache_root": str(cache_root),
        "results": results,
    }


@click.command("warm-cache")
@click.option("--adapter", "adapter_name", default="abel", show_default=True)
@click.option("--symbol", "symbols", multiple=True, required=True)
@click.option("--start", default=None, help="Warm data beginning at this start date")
@click.option("--end", default=None, help="Warm data ending at this end date")
@click.option("--timeframe", default="1d", show_default=True)
@click.option("--profile", default="daily", show_default=True)
@click.option("--limit", default=5000, show_default=True, type=click.IntRange(min=1))
@click.option("--env-path", default=None, help="Optional env file used by the adapter")
@click.option("--path", default=None, help="Optional local data path used by adapters such as csv")
@click.option("--output-json", default=None, help="Optional path for the JSON payload")
def warm_cache(
    adapter_name: str,
    symbols: tuple[str, ...],
    start: str | None,
    end: str | None,
    timeframe: str,
    profile: str,
    limit: int | None,
    env_path: str | None,
    path: str | None,
    output_json: str | None,
) -> None:
    """Warm adapter-backed market data into the formal edge cache."""
    payload = warm_cache_payload(
        symbols=list(symbols),
        adapter_name=adapter_name,
        start=start,
        end=end,
        timeframe=timeframe,
        profile=profile,
        limit=limit,
        env_path=env_path,
        path=path,
    )
    failures = [item for item in payload["results"] if not item.get("ok")]
    if output_json:
        Path(output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    click.echo(json.dumps(payload, indent=2))
    if failures:
        raise click.ClickException(
            f"Failed to warm cache for {len(failures)} symbol(s): "
            + ", ".join(str(item.get("symbol")) for item in failures)
        )
