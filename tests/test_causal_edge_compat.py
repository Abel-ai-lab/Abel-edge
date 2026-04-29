from __future__ import annotations

import subprocess
import sys


def test_causal_edge_import_path_aliases_abel_edge() -> None:
    import abel_edge
    import causal_edge

    assert causal_edge.__version__ == abel_edge.__version__


def test_causal_edge_submodules_resolve_current_runtime() -> None:
    from causal_edge.plugins.abel.discover import discover_graph_payload
    from causal_edge.research.evaluate import run_evaluation
    from causal_edge.validation.metrics import load_profile

    assert callable(discover_graph_payload)
    assert callable(run_evaluation)
    assert callable(load_profile)


def test_causal_edge_strategy_engine_identity_matches_current_runtime() -> None:
    from abel_edge.engine.base import StrategyEngine as AbelStrategyEngine
    from causal_edge.engine.base import StrategyEngine as CausalStrategyEngine

    assert CausalStrategyEngine is AbelStrategyEngine


def test_causal_edge_cli_module_alias_runs_version() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "causal_edge.cli", "version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "abel-edge" in completed.stdout
