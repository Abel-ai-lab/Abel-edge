from __future__ import annotations

import importlib
import json
from pathlib import Path


DEFAULT_STORY = {
    "source_label": "Source",
    "source_value": "Model",
    "lead_label": "Lead",
    "lead_value": "N/A",
    "delay_label": "Delay",
    "delay_value": "N/A",
    "lookback_label": "Lookback",
    "lookback_value": "N/A",
}


def strategy_story(s_cfg: dict) -> dict:
    try:
        module = importlib.import_module(s_cfg["engine"])
        graph_path = Path(module.__file__).resolve().with_name("causal_graph.json")
    except Exception:
        return dict(DEFAULT_STORY)

    if not graph_path.exists():
        return dict(DEFAULT_STORY)

    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(DEFAULT_STORY)

    parent = (graph.get("parents") or [{}])[0]
    if isinstance(parent, str):
        lead_value, delay_value, lookback_value = parent, "1d", "5d"
    else:
        lead_value = parent.get("ticker", "N/A")
        delay_value = f"{int(parent.get('lag', parent.get('tau', 1)))}d"
        lookback_value = f"{int(parent.get('window', 5))}d"

    source = str(graph.get("source", "Model"))
    return {
        **DEFAULT_STORY,
        "source_value": "Abel" if "abel" in source.lower() else source,
        "lead_value": lead_value,
        "delay_value": delay_value,
        "lookback_value": lookback_value,
    }
