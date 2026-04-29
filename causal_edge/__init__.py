"""Compatibility import path for the current :mod:`abel_edge` package."""

from __future__ import annotations

import importlib
import sys

import abel_edge as _abel_edge
from abel_edge import *  # noqa: F403


def _alias_module(legacy_name: str, current_name: str) -> None:
    sys.modules.setdefault(legacy_name, importlib.import_module(current_name))


for _legacy, _current in {
    "causal_edge.engine": "abel_edge.engine",
    "causal_edge.engine.base": "abel_edge.engine.base",
    "causal_edge.plugins": "abel_edge.plugins",
    "causal_edge.plugins.abel": "abel_edge.plugins.abel",
    "causal_edge.plugins.abel.credentials": "abel_edge.plugins.abel.credentials",
    "causal_edge.plugins.abel.discover": "abel_edge.plugins.abel.discover",
    "causal_edge.research": "abel_edge.research",
    "causal_edge.research.evaluate": "abel_edge.research.evaluate",
    "causal_edge.research.handoff": "abel_edge.research.handoff",
    "causal_edge.validation": "abel_edge.validation",
    "causal_edge.validation.gate_logic": "abel_edge.validation.gate_logic",
    "causal_edge.validation.metrics": "abel_edge.validation.metrics",
}.items():
    _alias_module(_legacy, _current)

__path__ = _abel_edge.__path__
__version__ = _abel_edge.__version__
