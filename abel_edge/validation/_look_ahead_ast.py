"""AST helpers for static look-ahead checks."""

from __future__ import annotations

import ast
from collections.abc import Iterable


def string_literal_spans(source: str) -> list[tuple[int, int]]:
    """Return [start, end) offsets for every string literal in source."""
    spans: list[tuple[int, int]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return spans

    line_starts = _line_start_offsets(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            start = node_offset(node, source, line_starts=line_starts)
            end = node_end_offset(node, source, line_starts=line_starts)
            if start is not None and end is not None:
                spans.append((start, end))
    return spans


def in_string_literal(offset: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start <= offset < end for start, end in spans)


def node_offset(
    node: ast.AST,
    source: str,
    *,
    line_starts: list[int] | None = None,
) -> int | None:
    if not hasattr(node, "lineno") or not hasattr(node, "col_offset"):
        return None
    starts = line_starts or _line_start_offsets(source)
    lineno = getattr(node, "lineno", 0)
    col = getattr(node, "col_offset", 0)
    if lineno <= 0 or lineno > len(starts):
        return None
    return starts[lineno - 1] + col


def node_end_offset(
    node: ast.AST,
    source: str,
    *,
    line_starts: list[int] | None = None,
) -> int | None:
    if not hasattr(node, "end_lineno") or not hasattr(node, "end_col_offset"):
        return None
    starts = line_starts or _line_start_offsets(source)
    lineno = getattr(node, "end_lineno", 0)
    col = getattr(node, "end_col_offset", 0)
    if lineno <= 0 or lineno > len(starts):
        return None
    return starts[lineno - 1] + col


def safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<expr>"


def numpy_call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return "<call>"


def is_numpy_reduction(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in {"mean", "std", "var"}:
        return False
    return isinstance(func.value, ast.Name) and func.value.id in {"np", "numpy"}


def collect_scope_bindings(scope: ast.AST) -> dict[str, ast.AST]:
    bindings: dict[str, ast.AST] = {}
    for stmt in ast.walk(scope):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                bindings[target.id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.value is not None:
                bindings[stmt.target.id] = stmt.value
    return bindings


def is_bounded_expr(node: ast.AST, bindings: dict[str, ast.AST]) -> bool:
    if isinstance(node, ast.Subscript):
        return _contains_slice(node.slice)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.ListComp)):
        return True
    if isinstance(node, ast.Call):
        return any(is_bounded_expr(arg, bindings) for arg in node.args)
    if isinstance(node, ast.Name) and node.id in bindings:
        return is_bounded_expr(bindings[node.id], bindings)
    return False


def _contains_slice(node: ast.AST) -> bool:
    if isinstance(node, ast.Slice):
        return True
    return any(_contains_slice(child) for child in ast.iter_child_nodes(node))


def _line_start_offsets(source: str) -> list[int]:
    starts = [0]
    for idx, char in enumerate(source):
        if char == "\n":
            starts.append(idx + 1)
    return starts
