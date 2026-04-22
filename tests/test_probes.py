from __future__ import annotations

import json

from click.testing import CliRunner

from causal_edge.cli import main
from causal_edge.research.probes import probe_graph_inputs


def _bars(rows: list[tuple[str, float, float]], symbol: str):
    import pandas as pd

    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "close", "volume"])
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([item[0] for item in rows], utc=True),
            "symbol": [symbol] * len(rows),
            "close": [item[1] for item in rows],
            "volume": [item[2] for item in rows],
        }
    )


def test_probe_graph_inputs_supports_cross_calendar_and_volume(monkeypatch):
    from causal_edge.research import probes as probe_module

    def _fake_fetch_bars(*, symbols, start=None, end=None, timeframe="1d", limit=None, fields=None, config=None):
        symbol = symbols[0]
        if symbol == "TSLA":
            return _bars(
                [
                    ("2020-01-03", 100.0, 1000.0),
                    ("2020-01-06", 101.0, 1100.0),
                ],
                symbol,
            )
        if symbol == "BTCUSD":
            return _bars(
                [
                    ("2020-01-03", 9000.0, 2000.0),
                    ("2020-01-04", 9200.0, 2100.0),
                    ("2020-01-05", 9300.0, 2200.0),
                ],
                symbol,
            )
        raise AssertionError(symbol)

    monkeypatch.setattr(probe_module, "fetch_bars", _fake_fetch_bars)

    report = probe_graph_inputs(
        node_ids=["BTCUSD.price", "TSLA.volume"],
        target_node="TSLA.price",
        start="2020-01-01",
    )

    results = {item["node_id"]: item for item in report["results"]}
    btc = results["BTCUSD.price"]
    volume = results["TSLA.volume"]

    assert report["target"]["node_id"] == "TSLA.price"
    assert report["basket"]["dense_overlap_start"].startswith("2020-01-03")
    assert btc["status"] == "full_target_overlap"
    assert btc["target_overlap_days"] == 2
    assert btc["asof_preview"][-1]["value"] == 9300.0
    assert volume["runtime_field"] == "volume"
    assert volume["native_sample"][0]["value"] == 1000.0


def test_probe_data_cli_writes_json(monkeypatch, tmp_path):
    from causal_edge.research import probes as probe_module

    monkeypatch.setattr(
        probe_module,
        "fetch_bars",
        lambda **kwargs: _bars(
            [
                ("2020-01-03", 100.0, 1000.0),
                ("2020-01-06", 101.0, 1100.0),
            ],
            kwargs["symbols"][0],
        ),
    )

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            main,
            [
                "probe-data",
                "--target-node",
                "TSLA.price",
                "--node-id",
                "TSLA.volume",
                "--output-json",
                "probe.json",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Graph Input Probe" in result.output
        payload = json.loads(open("probe.json", "r", encoding="utf-8").read())
        assert payload["target"]["node_id"] == "TSLA.price"
        assert payload["results"][0]["node_id"] == "TSLA.volume"
