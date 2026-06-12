"""CLI entry point tests."""

import json
import os
from pathlib import Path

from click.testing import CliRunner

from abel_edge import __version__
from abel_edge.cli import main
from abel_edge.engine.ledger import read_trade_log


def test_help():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "abel-edge" in result.output


def test_version():
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_status_empty():
    """Status with empty strategies.yaml should show 0 strategies."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("strategies.yaml").write_text("settings: {}\nstrategies: []\n", encoding="utf-8")
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "Strategies: 0" in result.output


def test_status_prefers_local_overlay_config():
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("strategies.yaml").write_text("settings: {}\nstrategies: []\n", encoding="utf-8")
        Path("strategies.local.yaml").write_text(
            "\n".join(
                [
                    "settings: {}",
                    "strategies:",
                    "  - id: local_demo",
                    '    name: "Local Demo"',
                    "    asset: ETHUSD",
                    '    color: "#2563EB"',
                    "    engine: strategies.local_demo.engine",
                    "    trade_log: data/local_demo.csv",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "Strategies: 1" in result.output
        assert "Local Demo" in result.output


def test_init_creates_project(tmp_path):
    """init should create a project directory with expected files."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["init", "myproject"])
        assert result.exit_code == 0, result.output
        assert "local sample-data demo strategies" in result.output
        root = Path("myproject")
        assert root.is_dir()
        assert (root / "strategies.yaml").exists()
        assert (root / "strategies" / "sma_crossover" / "engine.py").exists()
        assert (root / "strategies" / "sma_crossover" / "__init__.py").exists()
        assert (root / "strategies" / "feed_overlay_demo" / "engine.py").exists()
        assert (root / "data").is_dir()
        assert (root / "data" / "demo_target.csv").exists()
        assert (root / ".env.example").exists()
        assert (root / "CLAUDE.md").exists()
        assert (root / "AGENTS.md").exists()


def test_init_project_runs_sample_data_workflow(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["init", "myproject"])
        assert result.exit_code == 0, result.output

        root = Path("myproject")
        original_cwd = Path.cwd()
        try:
            os.chdir(root)

            result = runner.invoke(main, ["run"])
            assert result.exit_code == 0, result.output
            assert len(read_trade_log("data/trade_log_sma_crossover.csv")) > 0
            assert len(read_trade_log("data/trade_log_momentum_ml.csv")) > 0
            assert len(read_trade_log("data/trade_log_feed_overlay_demo.csv")) > 0
        finally:
            os.chdir(original_cwd)


def test_init_fails_if_dir_exists(tmp_path):
    """init should fail with a clear error if the directory already exists."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(main, ["init", "myproject"])
        result = runner.invoke(main, ["init", "myproject"])
        assert result.exit_code != 0
        assert "already exists" in result.output


def test_run_empty():
    """Run with no strategies should print message, not crash."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("strategies.yaml").write_text("settings: {}\nstrategies: []\n", encoding="utf-8")
        result = runner.invoke(main, ["run"])
        assert result.exit_code == 0
        assert "No strategies" in result.output


def test_warm_cache_supports_csv_path_option(tmp_path):
    runner = CliRunner()
    csv_path = tmp_path / "bars.csv"
    csv_path.write_text(
        "timestamp,symbol,close\n"
        "2020-01-01T00:00:00Z,AAPL,100\n"
        "2020-01-02T00:00:00Z,AAPL,101\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "warm-cache",
            "--adapter",
            "csv",
            "--path",
            str(csv_path),
            "--symbol",
            "AAPL",
            "--output-json",
            str(tmp_path / "warm-cache.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "warm-cache.json").read_text(encoding="utf-8"))
    assert payload["adapter"] == "csv"
    assert payload["path"] == str(csv_path)
    assert payload["results"][0]["ok"] is True


def test_validate_empty():
    """Validate with no strategies should print message, not crash."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("strategies.yaml").write_text("settings: {}\nstrategies: []\n", encoding="utf-8")
        result = runner.invoke(main, ["validate"])
        assert result.exit_code == 0
        assert "No strategies" in result.output


def test_validate_csv_missing_file():
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--csv", "missing.csv"])
    assert result.exit_code != 0
    assert "CSV not found: missing.csv" in result.output
