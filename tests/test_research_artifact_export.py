from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from click.testing import CliRunner

from abel_edge.cli import main
from abel_edge.engine.ledger import read_trade_log
from abel_edge.research.artifact_export import (
    canonical_manifest_bytes,
    export_strategy_artifact_zip,
    sha256_file,
    write_backtest_trade_log_from_metric_input,
)
from abel_edge.research.state_intent import validate_state_intent
from abel_edge.runtime_paths import runtime_paths


def _write_sources(tmp_path: Path) -> dict[str, Path]:
    workdir = tmp_path / "branch"
    outputs = workdir / "outputs"
    inputs = workdir / "inputs"
    outputs.mkdir(parents=True)
    inputs.mkdir()
    (workdir / "engine.py").write_text(
        "from abel_edge.engine.base import StrategyEngine\n"
        "class BranchEngine(StrategyEngine):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (workdir / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workdir / "branch.yaml").write_text(
        "version: 2\nbranch_id: momentum_lead\ntarget: TSLA\n",
        encoding="utf-8",
    )
    (inputs / "dependencies.json").write_text(
        json.dumps({"version": 1, "target": "TSLA"}),
        encoding="utf-8",
    )
    (inputs / "data_manifest.json").write_text(
        json.dumps({"version": 1, "target": "TSLA", "feeds": []}),
        encoding="utf-8",
    )
    edge_result = outputs / "round-006-edge-result.json"
    edge_result.write_text(
        json.dumps({"verdict": "PASS", "profile": "equity_daily"}),
        encoding="utf-8",
    )
    edge_report = outputs / "round-006-edge-validation.md"
    edge_report.write_text("# Evaluation Summary\n", encoding="utf-8")
    metric_csv = outputs / "round-006-metric-input.csv"
    metric_csv.write_text(
        "date,asset_return,pnl,position,gross_pnl,turnover,execution_cost,"
        "next_position,decision_time,effective_time,close\n"
        "2020-01-01T00:00:00Z,0,0,0,0,0,0,0,2020-01-01T00:00:00Z,"
        "2020-01-01T00:00:00Z,100\n"
        "2020-01-02T00:00:00Z,0.01,0.01,1,0.01,1,0,1,"
        "2020-01-02T00:00:00Z,2020-01-02T00:00:00Z,101\n",
        encoding="utf-8",
    )
    trade_log = outputs / "round-006-trade-log.csv"
    write_backtest_trade_log_from_metric_input(metric_csv, trade_log)
    return {
        "workdir": workdir,
        "outputs": outputs,
        "edge_result": edge_result,
        "edge_report": edge_report,
        "metric_csv": metric_csv,
        "trade_log": trade_log,
    }


def _manifest(paths: dict[str, Path]) -> dict:
    workdir = paths["workdir"]
    return {
        "schema": "abel-invest.strategy-artifact/v1",
        "files": [
            _file_entry("strategy/strategy.py", workdir / "engine.py"),
            _file_entry("strategy/helper.py", workdir / "helper.py"),
            _file_entry("edge/edge-result.json", paths["edge_result"]),
            _file_entry("edge/trade-log.csv", paths["trade_log"]),
            _file_entry("edge/edge-validation.md", paths["edge_report"]),
            _file_entry("runtime/strategy.yaml", workdir / "branch.yaml"),
            _file_entry("runtime/dependencies.json", workdir / "inputs" / "dependencies.json"),
            _file_entry("runtime/data_manifest.json", workdir / "inputs" / "data_manifest.json"),
        ],
    }


def _file_entry(artifact_path: str, source_path: Path) -> dict:
    return {
        "path": artifact_path,
        "sha256": sha256_file(source_path),
        "bytes": source_path.stat().st_size,
    }


def test_write_backtest_trade_log_from_metric_input_replaces_live_rows(tmp_path: Path) -> None:
    paths = _write_sources(tmp_path)
    paths["trade_log"].write_text(
        "date,pnl,position,cum_return,source\n2026-01-01,1,1,1,live\n",
        encoding="utf-8",
    )

    result = write_backtest_trade_log_from_metric_input(
        paths["metric_csv"],
        paths["trade_log"],
    )

    trade_log = read_trade_log(paths["trade_log"])
    assert result["rowCount"] == 2
    assert len(trade_log) == 2
    assert set(trade_log["source"]) == {"backfill"}
    assert float(trade_log.iloc[-1]["cum_return"]) == pytest.approx(0.01)
    assert result["sha256"] == sha256_file(paths["trade_log"])


def test_export_strategy_artifact_zip_validates_manifest_and_writes_zip(
    tmp_path: Path,
) -> None:
    paths = _write_sources(tmp_path)
    manifest = _manifest(paths)
    output_zip = paths["outputs"] / "artifact.zip"

    result = export_strategy_artifact_zip(
        manifest,
        output_zip_path=output_zip,
        workdir=paths["workdir"],
        edge_result_path=paths["edge_result"],
        trade_log_path=paths["trade_log"],
        edge_report_path=paths["edge_report"],
    )

    assert result["artifactSha256"] == sha256_file(output_zip)
    assert result["fileCount"] == 9
    with ZipFile(output_zip) as artifact:
        assert artifact.namelist() == [
            "manifest.json",
            "strategy/strategy.py",
            "strategy/helper.py",
            "edge/edge-result.json",
            "edge/trade-log.csv",
            "edge/edge-validation.md",
            "runtime/strategy.yaml",
            "runtime/dependencies.json",
            "runtime/data_manifest.json",
        ]
        assert json.loads(artifact.read("manifest.json")) == manifest


def test_export_strategy_artifact_zip_accepts_extra_source_map_for_initial_state(
    tmp_path: Path,
) -> None:
    paths = _write_sources(tmp_path)
    state_file = tmp_path / "model.joblib"
    state_file.write_text("model-state\n", encoding="utf-8")
    manifest = _manifest(paths)
    manifest["files"].append(_file_entry("runtime/initial-state/model.joblib", state_file))
    output_zip = paths["outputs"] / "artifact.zip"

    result = export_strategy_artifact_zip(
        manifest,
        output_zip_path=output_zip,
        workdir=paths["workdir"],
        edge_result_path=paths["edge_result"],
        trade_log_path=paths["trade_log"],
        edge_report_path=paths["edge_report"],
        extra_source_map={"runtime/initial-state/model.joblib": state_file},
    )

    assert result["fileCount"] == 10
    with ZipFile(output_zip) as artifact:
        assert artifact.read("runtime/initial-state/model.joblib") == b"model-state\n"


def test_runtime_paths_uses_explicit_paths_and_context_payload(tmp_path: Path) -> None:
    paths = runtime_paths(
        base_strategy=tmp_path / "strategy",
        runtime=tmp_path / "runtime",
        state=tmp_path / "state",
        create=True,
    )

    assert paths.state.is_dir()
    assert paths.as_context_payload()["state"] == str(paths.state)


def test_validate_state_intent_accepts_declared_state_file(tmp_path: Path) -> None:
    state_file = tmp_path / "model" / "latest.joblib"
    state_file.parent.mkdir()
    state_file.write_text("state\n", encoding="utf-8")

    entries = validate_state_intent(
        {
            "schema": "abel-invest.state-intent/v1",
            "entries": [
                {
                    "path": "model/latest.joblib",
                    "role": "initial_state",
                    "mutableInPaper": True,
                    "requiredForSignal": True,
                    "producedBy": "pytest",
                }
            ],
        },
        root=tmp_path,
    )

    assert entries[0].path == "model/latest.joblib"
    assert entries[0].role == "initial_state"


def test_export_strategy_artifact_zip_rejects_checksum_mismatch(tmp_path: Path) -> None:
    paths = _write_sources(tmp_path)
    manifest = _manifest(paths)
    manifest["files"][1]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="checksum mismatch"):
        export_strategy_artifact_zip(
            manifest,
            output_zip_path=paths["outputs"] / "artifact.zip",
            workdir=paths["workdir"],
            edge_result_path=paths["edge_result"],
            trade_log_path=paths["trade_log"],
            edge_report_path=paths["edge_report"],
        )


def test_export_strategy_artifact_zip_rejects_denylisted_paths(tmp_path: Path) -> None:
    paths = _write_sources(tmp_path)
    manifest = _manifest(paths)
    manifest["files"].append(
        {
            "path": "strategy/__pycache__/helper.pyc",
            "sha256": "0" * 64,
            "bytes": 1,
        }
    )

    with pytest.raises(ValueError, match="denylisted"):
        export_strategy_artifact_zip(
            manifest,
            output_zip_path=paths["outputs"] / "artifact.zip",
            workdir=paths["workdir"],
            edge_result_path=paths["edge_result"],
            trade_log_path=paths["trade_log"],
            edge_report_path=paths["edge_report"],
        )


def test_export_artifact_cli_regenerates_trade_log_and_writes_json_result(
    tmp_path: Path,
) -> None:
    paths = _write_sources(tmp_path)
    manifest = _manifest(paths)
    manifest_path = paths["outputs"] / "manifest.json"
    manifest_path.write_bytes(canonical_manifest_bytes(manifest))
    output_zip = paths["outputs"] / "artifact.zip"

    result = CliRunner().invoke(
        main,
        [
            "export-artifact",
            "--workdir",
            str(paths["workdir"]),
            "--manifest-json",
            str(manifest_path),
            "--edge-result",
            str(paths["edge_result"]),
            "--edge-report",
            str(paths["edge_report"]),
            "--metric-csv",
            str(paths["metric_csv"]),
            "--trade-log",
            str(paths["trade_log"]),
            "--output-zip",
            str(output_zip),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["artifactPath"] == str(output_zip)
    assert output_zip.exists()
