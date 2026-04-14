"""Shared CLI helpers."""

from __future__ import annotations

import click

from causal_edge.engine.price_data import load_bars_from_csv, resolve_price_config


def build_bars_loader(cfg: dict):
    strategies = cfg.get("strategies") or []
    if not strategies:
        return None

    default_source = (
        (cfg.get("settings") or {}).get("price_data", {}).get("default_source", "abel")
    )
    if any(
        (resolve_price_config(cfg.get("settings") or {}, strategy).get("source") == "csv")
        for strategy in strategies
    ):
        return _dispatch_bars_loader(cfg)
    if default_source == "abel":
        return _dispatch_bars_loader(cfg)
    return None


def _dispatch_bars_loader(cfg: dict):
    def _loader(**kwargs):
        config = kwargs.get("config") or {}
        source = config.get("source") or (cfg.get("settings") or {}).get("price_data", {}).get(
            "default_source", "abel"
        )
        if source == "csv":
            path = config.get("path")
            if not path:
                raise click.ClickException("price_data.path is required when source='csv'.")
            return load_bars_from_csv(path, **kwargs)
        if source == "abel":
            try:
                from causal_edge.plugins.abel.credentials import MissingAbelApiKeyError
                from causal_edge.plugins.abel.prices import fetch_bars
            except ImportError as exc:
                raise click.ClickException(
                    "Abel price source is unavailable. See: causal_edge/plugins/AGENTS.md"
                ) from exc
            try:
                return fetch_bars(**kwargs)
            except MissingAbelApiKeyError as exc:
                raise click.ClickException(str(exc)) from exc
        raise click.ClickException(f"Unsupported price_data.source '{source}'.")

    return _loader
