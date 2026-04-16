"""Regression tests for strategy engine module loading."""

import importlib
import sys
from pathlib import Path

import pytest

from causal_edge.engine.trader import _load_engine


def _clear_strategy_modules() -> None:
    for name in list(sys.modules):
        if name == "strategies" or name.startswith("strategies."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def test_load_engine_prefers_module_owned_wrapper(tmp_path):
    project_root = tmp_path / "project"
    strategies_dir = project_root / "strategies"
    base_dir = strategies_dir / "base_engine"
    wrapper_dir = strategies_dir / "wrapper_engine"
    base_dir.mkdir(parents=True)
    wrapper_dir.mkdir(parents=True)
    (strategies_dir / "__init__.py").write_text("", encoding="utf-8")
    (base_dir / "__init__.py").write_text("", encoding="utf-8")
    (wrapper_dir / "__init__.py").write_text("", encoding="utf-8")
    (base_dir / "engine.py").write_text(
        "\n".join(
            [
                "from causal_edge.engine.base import StrategyEngine",
                "",
                "class BaseEngine(StrategyEngine):",
                "    def compute_signals(self):",
                "        raise NotImplementedError",
                "",
                "    def get_latest_signal(self):",
                "        return {'position': 0.0}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (wrapper_dir / "engine.py").write_text(
        "\n".join(
            [
                "from strategies.base_engine.engine import BaseEngine",
                "",
                "class WrapperEngine(BaseEngine):",
                "    pass",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    original_cwd = Path.cwd()
    sys.path.insert(0, str(project_root))
    _clear_strategy_modules()
    try:
        os_cwd = str(project_root)
        import os

        os.chdir(project_root)
        engine_cls = _load_engine("strategies.wrapper_engine.engine")
    finally:
        os.chdir(original_cwd)
        sys.path.pop(0)
        _clear_strategy_modules()

    assert engine_cls.__name__ == "WrapperEngine"
    assert engine_cls.__module__ == "strategies.wrapper_engine.engine"


def test_load_engine_rejects_imported_engine_only(tmp_path):
    project_root = tmp_path / "project"
    strategies_dir = project_root / "strategies"
    base_dir = strategies_dir / "base_engine"
    alias_dir = strategies_dir / "alias_engine"
    base_dir.mkdir(parents=True)
    alias_dir.mkdir(parents=True)
    (strategies_dir / "__init__.py").write_text("", encoding="utf-8")
    (base_dir / "__init__.py").write_text("", encoding="utf-8")
    (alias_dir / "__init__.py").write_text("", encoding="utf-8")
    (base_dir / "engine.py").write_text(
        "\n".join(
            [
                "from causal_edge.engine.base import StrategyEngine",
                "",
                "class BaseEngine(StrategyEngine):",
                "    def compute_signals(self):",
                "        raise NotImplementedError",
                "",
                "    def get_latest_signal(self):",
                "        return {'position': 0.0}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (alias_dir / "engine.py").write_text(
        "from strategies.base_engine.engine import BaseEngine\n",
        encoding="utf-8",
    )

    original_cwd = Path.cwd()
    sys.path.insert(0, str(project_root))
    _clear_strategy_modules()
    try:
        import os

        os.chdir(project_root)
        with pytest.raises(ImportError, match="defines its own StrategyEngine subclass"):
            _load_engine("strategies.alias_engine.engine")
    finally:
        os.chdir(original_cwd)
        sys.path.pop(0)
        _clear_strategy_modules()
