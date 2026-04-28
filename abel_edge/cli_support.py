"""Shared CLI helpers."""

from __future__ import annotations

import click

from abel_edge.engine.feed_loader import load_feed_frame


def build_bars_loader(cfg: dict):
    strategies = cfg.get("strategies") or []
    if not strategies:
        return None

    return _dispatch_bars_loader(cfg)


def _dispatch_bars_loader(cfg: dict):
    def _loader(**kwargs):
        config = kwargs.get("config") or {}
        profile = ((cfg.get("settings") or {}).get("data_contract") or {}).get("profile", "daily")
        feed_cfg = {
            **config,
            "name": str(config.get("name") or "primary"),
            "kind": "bars",
            "profile": profile,
        }
        if "adapter" not in feed_cfg:
            default_adapter = (
                (cfg.get("settings") or {}).get("price_data", {}).get("default_adapter")
                or (cfg.get("settings") or {}).get("price_data", {}).get("default_source", "abel")
            )
            feed_cfg["adapter"] = default_adapter
        try:
            return load_feed_frame(
                feed_cfg,
                start=kwargs.get("start"),
                end=kwargs.get("end"),
                timeframe=kwargs.get("timeframe"),
                limit=kwargs.get("limit"),
                fields=kwargs.get("fields"),
            )
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

    return _loader
