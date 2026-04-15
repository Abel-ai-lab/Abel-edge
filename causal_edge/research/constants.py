"""Shared constants for research workflows."""

RESULTS_COLUMNS = (
    "exp_id",
    "ticker",
    "branch_id",
    "round_id",
    "decision",
    "commit",
    "lo_adj",
    "ic",
    "omega",
    "sharpe",
    "max_dd",
    "pnl",
    "K",
    "score",
    "verdict",
    "status",
    "mode",
    "description",
    "validation_path",
)

RESULTS_HEADER = "\t".join(RESULTS_COLUMNS) + "\n"

EVENTS_COLUMNS = (
    "timestamp",
    "event",
    "branch_id",
    "round_id",
    "mode",
    "verdict",
    "decision",
    "description",
    "artifact_path",
)

EVENTS_HEADER = "\t".join(EVENTS_COLUMNS) + "\n"

BRANCHES_DIRNAME = "branches"
