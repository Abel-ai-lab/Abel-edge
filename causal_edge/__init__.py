"""Compatibility import path for the current :mod:`abel_edge` package."""

from __future__ import annotations

from abel_edge import *  # noqa: F403
import abel_edge as _abel_edge

__path__ = _abel_edge.__path__
__version__ = _abel_edge.__version__
