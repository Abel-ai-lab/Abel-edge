"""Shared helpers for resolving module-owned strategy engine classes."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def resolve_module_owned_engine(module: ModuleType):
    """Return the module-owned ``StrategyEngine`` subclass defined in ``module``."""
    from abel_edge.engine.base import StrategyEngine

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, StrategyEngine)
            and attr is not StrategyEngine
            and attr.__module__ == module.__name__
        ):
            return attr
    raise ImportError(
        f"No module-owned StrategyEngine subclass found in '{module.__name__}'. "
        "Fix: Ensure engine.py defines its own StrategyEngine subclass instead of only importing one."
    )


def load_engine_from_import_path(engine_path: str):
    """Import an engine module by dotted path and resolve its owned engine class."""
    if engine_path.startswith("strategies.") and (Path.cwd() / "strategies").exists():
        cwd_str = str(Path.cwd())
        if cwd_str not in sys.path:
            sys.path.insert(0, cwd_str)
        stale = [
            name for name in sys.modules if name == "strategies" or name.startswith("strategies.")
        ]
        for name in stale:
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    module = importlib.import_module(engine_path)
    return resolve_module_owned_engine(module)


def load_engine_from_file(engine_path: Path, *, module_name: str = "research_branch_engine"):
    """Load a local ``engine.py`` file and resolve its owned engine class."""
    parent_str = str(engine_path.parent.resolve())
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)
    sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_file_location(module_name, str(engine_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return resolve_module_owned_engine(module)
