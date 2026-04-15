"""Look-ahead bias checks: static heuristics plus runtime leak detection."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np

from causal_edge.validation._look_ahead_ast import (
    collect_scope_bindings,
    in_string_literal,
    is_bounded_expr,
    is_numpy_reduction,
    node_offset,
    numpy_call_name,
    safe_unparse,
    string_literal_spans,
)


def check_static(source: str) -> list[str]:
    """Run static T2-T5 look-ahead checks against Python source."""
    lines = source.split("\n")
    raw: list[str] = []
    raw.extend(_t2_rolling_without_shift(source))
    raw.extend(_t3_global_stats(source))
    raw.extend(_t4_walk_forward_slicing(source))
    raw.extend(_t5_trend_filter(source))
    return [violation for violation in raw if not _is_suppressed(violation, lines)]


def check_static_file(path: str | Path) -> list[str]:
    return check_static(Path(path).read_text(encoding="utf-8"))


def check_runtime(
    pnl: np.ndarray,
    positions: np.ndarray,
    returns: np.ndarray | None = None,
    *,
    threshold: float = 0.3,
    hit_rate_max: float = 0.70,
) -> list[str]:
    """Run runtime leak checks against aligned pnl/position/return arrays."""
    warnings: list[str] = []
    active = np.abs(positions) > 0.01

    if returns is not None and len(returns) == len(positions):
        if active.sum() > 30:
            pos_mag = np.abs(positions[active])
            ret_mag = np.abs(returns[active])
            if np.std(pos_mag) > 1e-12 and np.std(ret_mag) > 1e-12:
                corr = np.corrcoef(pos_mag, ret_mag)[0, 1]
                if not np.isnan(corr) and abs(corr) > threshold:
                    warnings.append(
                        f"R1: |position| correlates with |same-day return| "
                        f"(corr={corr:.3f}, threshold={threshold}). "
                        f"Positions may be using future information."
                    )

    active_pnl = pnl[active]
    if len(active_pnl) > 30:
        reference = active_pnl
        if returns is not None and len(returns) == len(positions):
            reference = returns[active]
        if np.std(reference) > 1e-12:
            hit_rate = float(np.mean(active_pnl > 0))
            if hit_rate > hit_rate_max:
                warnings.append(
                    f"R2: Hit rate {hit_rate:.0%} on {len(active_pnl)} active days "
                    f"exceeds {hit_rate_max:.0%}. Verify no look-ahead in features."
                )

    return warnings


def _is_suppressed(violation: str, lines: list[str]) -> bool:
    match = re.match(r"^(T[2-5])\s+L(\d+):", violation)
    if not match:
        return False

    code = match.group(1).lower()
    line_num = int(match.group(2))
    candidates: list[str] = []
    if 0 <= line_num - 1 < len(lines):
        candidates.append(lines[line_num - 1])
    if 0 <= line_num - 2 < len(lines):
        candidates.append(lines[line_num - 2])

    for line in candidates:
        noqa = re.search(r"#\s*noqa(?::\s*([A-Za-z0-9_,\s]+))?", line)
        if not noqa:
            continue
        arg = noqa.group(1)
        if arg is None:
            return True
        tokens = {token.strip().lower() for token in arg.split(",")}
        if code in tokens or "lookahead" in tokens:
            return True
    return False


def _t2_rolling_without_shift(source: str) -> list[str]:
    violations: list[str] = []
    spans = string_literal_spans(source)
    pattern = re.compile(
        r"\.rolling\(\s*\d+\s*\)\s*\.\s*(mean|std|sum|var|median|corr|min|max)\s*\([^)]*\)"
    )
    for match in pattern.finditer(source):
        if in_string_literal(match.start(), spans):
            continue
        line_num = source[: match.start()].count("\n") + 1
        after = source[match.end() : match.end() + 50]
        if ".shift(" in after:
            continue
        ctx = source[max(0, match.start() - 30) : match.end() + 50]
        if "shift(-" in ctx or "# target" in ctx.lower() or "# label" in ctx.lower():
            continue
        line_start = source.rfind("\n", 0, match.start()) + 1
        line = source[line_start : match.end() + 50]
        if re.match(r"\s*(y|target|label)\s*=", line):
            continue
        snippet = source[max(0, match.start() - 5) : match.end() + 20].strip()
        violations.append(
            f"T2 L{line_num}: rolling().{match.group(1)}() without .shift(1): {snippet}"
        )
    return violations


def _t3_global_stats(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    violations: list[str] = []
    spans = string_literal_spans(source)
    scope_cache: dict[int, dict[str, ast.AST]] = {}

    def _scan(node: ast.AST, scope: ast.AST) -> None:
        if isinstance(node, ast.Call) and is_numpy_reduction(node) and node.args:
            offset = node_offset(node, source)
            if offset is None or not in_string_literal(offset, spans):
                scope_id = id(scope)
                if scope_id not in scope_cache:
                    scope_cache[scope_id] = collect_scope_bindings(scope)
                if not is_bounded_expr(node.args[0], scope_cache[scope_id]):
                    violations.append(
                        f"T3 L{node.lineno}: "
                        f"np.{numpy_call_name(node)}({safe_unparse(node.args[0])}) "
                        f"on full array - use rolling/expanding or [:i] slice"
                    )
        for child in ast.iter_child_nodes(node):
            new_scope = child if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            _scan(child, new_scope)

    _scan(tree, tree)
    return violations


def _t4_walk_forward_slicing(source: str) -> list[str]:
    violations: list[str] = []
    pattern = re.compile(r"\[\s*:?\s*(\w+)\s*\+\s*1\s*\]")
    for match in pattern.finditer(source):
        line_num = source[: match.start()].count("\n") + 1
        line_start = source.rfind("\n", 0, match.start()) + 1
        line_end = source.find("\n", match.end())
        if line_end == -1:
            line_end = len(source)
        line = source[line_start:line_end]
        if any(keyword in line.lower() for keyword in ("train", "fit", "x_tr", "y_tr", "x_s", "weight")):
            violations.append(
                f"T4 L{line_num}: [:i+1] in training context - use [:i] to exclude current day"
            )
    return violations


def _t5_trend_filter(source: str) -> list[str]:
    violations: list[str] = []
    pattern = re.compile(
        r"(close|price|px|eth_price|sma|ma)\s*\[\s*(\w+)\s*\]"
        r"\s*[<>]=?\s*"
        r"(close|price|px|sma|ma)\s*\[\s*(\w+)\s*\]"
    )
    for match in pattern.finditer(source):
        idx_left, idx_right = match.group(2), match.group(4)
        if idx_left != idx_right or "-1" in idx_left:
            continue
        line_num = source[: match.start()].count("\n") + 1
        line_start = source.rfind("\n", 0, match.start()) + 1
        line_end = source.find("\n", match.end())
        if line_end == -1:
            line_end = len(source)
        line = source[line_start:line_end]
        if "shift(1)" in line or "i-1" in line or "i - 1" in line:
            continue
        violations.append(
            f"T5 L{line_num}: trend filter uses current-day index - use [i-1] or .shift(1)"
        )
    return violations
