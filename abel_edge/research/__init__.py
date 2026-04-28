"""Raw evaluation helpers for strategy experiments."""

from abel_edge.research.evaluate import (
    check_look_ahead,
    compute_k,
    render_validation_markdown,
    run_evaluation,
    write_evaluation_outputs,
)
from abel_edge.research.handoff import (
    HANDOFF_CONTRACT,
    build_strategy_handoff,
    load_strategy_handoff,
    validate_strategy_handoff,
)

__all__ = [
    "check_look_ahead",
    "compute_k",
    "render_validation_markdown",
    "run_evaluation",
    "write_evaluation_outputs",
    "HANDOFF_CONTRACT",
    "build_strategy_handoff",
    "load_strategy_handoff",
    "validate_strategy_handoff",
]
