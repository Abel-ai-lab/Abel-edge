"""Strategy artifact export helpers for hosted paper tracking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from abel_edge.engine.ledger import write_trade_log


STRATEGY_ARTIFACT_SCHEMA = "abel-invest.strategy-artifact/v1"
REQUIRED_ARTIFACT_PATHS = {
    "manifest.json",
    "strategy/strategy.py",
    "edge/edge-result.json",
    "edge/trade-log.csv",
    "runtime/strategy.yaml",
    "runtime/dependencies.json",
    "runtime/data_manifest.json",
}


def write_backtest_trade_log_from_metric_input(
    metric_csv_path: Path | str,
    trade_log_path: Path | str,
) -> dict[str, Any]:
    """Convert an evaluation metric frame into the standard backtest trade log."""

    metric_csv_path = Path(metric_csv_path)
    trade_log_path = Path(trade_log_path)
    frame = pd.read_csv(metric_csv_path)
    required = {"date", "asset_return", "pnl", "position"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"metric input CSV missing columns: {', '.join(missing)}")

    # The hosted artifact needs historical backtest rows only. Existing live rows
    # at the destination would change the artifact hash and admission semantics.
    trade_log_path.unlink(missing_ok=True)
    write_trade_log(
        pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True, format="mixed")),
        frame["asset_return"].to_numpy(dtype=float),
        frame["pnl"].to_numpy(dtype=float),
        frame["position"].to_numpy(dtype=float),
        trade_log_path,
        source="backfill",
        decision_times=_optional_datetime_index(frame, "decision_time"),
        effective_times=_optional_datetime_index(frame, "effective_time"),
        close_prices=_optional_float_array(frame, "close"),
        next_positions=_optional_float_array(frame, "next_position"),
        gross_pnl=_optional_float_array(frame, "gross_pnl"),
        turnover=_optional_float_array(frame, "turnover"),
        execution_cost=_optional_float_array(frame, "execution_cost"),
    )
    return {
        "tradeLogPath": str(trade_log_path),
        "rowCount": int(len(frame)),
        "sha256": sha256_file(trade_log_path),
    }


def export_strategy_artifact_zip(
    manifest: dict[str, Any],
    *,
    output_zip_path: Path | str,
    workdir: Path | str,
    edge_result_path: Path | str,
    trade_log_path: Path | str,
    edge_report_path: Path | str | None = None,
) -> dict[str, Any]:
    """Write artifact.zip using the file list and hashes declared by manifest."""

    _validate_manifest_schema(manifest)
    output_zip_path = Path(output_zip_path)
    source_map = _default_source_map(
        workdir=Path(workdir),
        edge_result_path=Path(edge_result_path),
        trade_log_path=Path(trade_log_path),
        edge_report_path=Path(edge_report_path) if edge_report_path else None,
    )
    file_entries = _manifest_file_entries(manifest)
    _validate_required_artifact_paths(file_entries)
    resolved_files = _resolve_manifest_sources(file_entries, source_map)

    manifest_bytes = canonical_manifest_bytes(manifest)
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip_path, "w", compression=ZIP_DEFLATED) as artifact:
        artifact.writestr("manifest.json", manifest_bytes)
        for artifact_path, source_path in resolved_files:
            artifact.write(source_path, artifact_path)

    return {
        "artifactPath": str(output_zip_path),
        "artifactSha256": sha256_file(output_zip_path),
        "artifactBytes": output_zip_path.stat().st_size,
        "fileCount": len(resolved_files) + 1,
    }


def load_manifest(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact manifest must be a JSON object")
    return payload


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_datetime_index(frame: pd.DataFrame, column: str) -> pd.DatetimeIndex | None:
    if column not in frame.columns:
        return None
    return pd.DatetimeIndex(pd.to_datetime(frame[column], utc=True, format="mixed"))


def _optional_float_array(frame: pd.DataFrame, column: str):
    if column not in frame.columns:
        return None
    return frame[column].to_numpy(dtype=float)


def _validate_manifest_schema(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != STRATEGY_ARTIFACT_SCHEMA:
        raise ValueError(
            "artifact manifest schema must be "
            f"{STRATEGY_ARTIFACT_SCHEMA!r}, got {manifest.get('schema')!r}"
        )


def _manifest_file_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("artifact manifest files must be a non-empty list")
    if not all(isinstance(item, dict) for item in entries):
        raise ValueError("artifact manifest files must contain objects")
    return entries


def _validate_required_artifact_paths(file_entries: list[dict[str, Any]]) -> None:
    declared = {"manifest.json"}
    declared.update(str(item.get("path") or "").strip() for item in file_entries)
    missing = sorted(REQUIRED_ARTIFACT_PATHS - declared)
    if missing:
        raise ValueError(f"artifact manifest missing required files: {', '.join(missing)}")


def _resolve_manifest_sources(
    file_entries: list[dict[str, Any]],
    source_map: dict[str, Path],
) -> list[tuple[str, Path]]:
    resolved: list[tuple[str, Path]] = []
    seen_paths: set[str] = set()
    for entry in file_entries:
        artifact_path = _validate_artifact_path(entry.get("path"))
        if artifact_path in seen_paths:
            raise ValueError(f"duplicate artifact path in manifest: {artifact_path}")
        seen_paths.add(artifact_path)
        source_path = source_map.get(artifact_path)
        if source_path is None:
            raise ValueError(f"no export source registered for artifact path: {artifact_path}")
        if not source_path.is_file():
            raise FileNotFoundError(f"artifact source not found for {artifact_path}: {source_path}")
        expected_bytes = entry.get("bytes")
        if expected_bytes is not None and int(expected_bytes) != source_path.stat().st_size:
            raise ValueError(
                f"artifact source byte size mismatch for {artifact_path}: "
                f"manifest={expected_bytes} actual={source_path.stat().st_size}"
            )
        expected_sha = str(entry.get("sha256") or "").strip().lower()
        if not expected_sha:
            raise ValueError(f"artifact source checksum missing for {artifact_path}")
        actual_sha = sha256_file(source_path)
        if expected_sha != actual_sha:
            raise ValueError(
                f"artifact source checksum mismatch for {artifact_path}: "
                f"manifest={expected_sha} actual={actual_sha}"
            )
        resolved.append((artifact_path, source_path))
    return resolved


def _validate_artifact_path(value: Any) -> str:
    artifact_path = str(value or "").strip()
    path = Path(artifact_path)
    if not artifact_path or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid artifact path: {artifact_path!r}")
    if path.parts[0] == "manifest.json":
        raise ValueError("manifest.json is written by the exporter and cannot be listed")
    return artifact_path


def _default_source_map(
    *,
    workdir: Path,
    edge_result_path: Path,
    trade_log_path: Path,
    edge_report_path: Path | None,
) -> dict[str, Path]:
    source_map = {
        "strategy/strategy.py": workdir / "engine.py",
        "edge/edge-result.json": edge_result_path,
        "edge/trade-log.csv": trade_log_path,
        "runtime/strategy.yaml": workdir / "branch.yaml",
        "runtime/dependencies.json": workdir / "inputs" / "dependencies.json",
        "runtime/data_manifest.json": workdir / "inputs" / "data_manifest.json",
    }
    if edge_report_path is not None:
        source_map["edge/edge-validation.md"] = edge_report_path
    return source_map
