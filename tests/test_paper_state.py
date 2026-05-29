import pickle

import pandas as pd
import pytest

from abel_edge.paper_state import PaperStateStore, paper_as_of_key


def _context(tmp_path):
    return {
        "_runtime_paths": {
            "base_strategy": str(tmp_path / "strategy"),
            "runtime": str(tmp_path / "runtime"),
            "state": str(tmp_path / "state"),
        }
    }


def test_paper_as_of_key_normalizes_daily_dates():
    assert paper_as_of_key("2026-05-18T21:30:00+08:00") == "2026-05-18"
    assert paper_as_of_key(pd.Timestamp("2026-05-18T00:00:00Z")) == "2026-05-18"
    assert paper_as_of_key(None) is None


def test_paper_state_store_reads_and_writes_json(tmp_path):
    store = PaperStateStore.from_context(
        _context(tmp_path),
        "strategy/paper_state.json",
    )
    payload = {"schema": "demo/v1", "last_as_of": "2026-05-18", "position": 0.25}

    store.save(payload)

    assert store.path == (tmp_path / "state" / "strategy" / "paper_state.json").resolve()
    assert store.load() == payload
    assert store.is_current(payload, "2026-05-18T10:00:00Z")
    assert not store.is_current(payload, "2026-05-19T00:00:00Z")
    assert store.summary(payload) == {
        "state_file": str(store.path),
        "state_path": "strategy/paper_state.json",
        "state_as_of": "2026-05-18",
        "state_schema": "demo/v1",
    }


def test_paper_state_store_pickle_supports_model_like_objects(tmp_path):
    store = PaperStateStore.from_context(_context(tmp_path))
    model_like = {"coef": [1.0, 2.0], "last_as_of": "2026-05-18"}

    store.save({"schema": "model/v1", "model": model_like})

    with store.path.open("rb") as handle:
        assert pickle.load(handle)["model"] == model_like
    assert store.load()["schema"] == "model/v1"


def test_paper_state_store_signal_preserves_ledger_schema(tmp_path):
    store = PaperStateStore.from_context(_context(tmp_path), "strategy/state.pkl")
    payload = store.mark_current({"schema": "demo/v1"}, "2026-05-18T00:00:00Z")

    signal = store.signal(
        next_position=0.35,
        payload=payload,
        confidence=0.7,
        ignored_object=object(),
    )

    assert signal == {
        "next_position": 0.35,
        "confidence": 0.7,
    }


def test_paper_state_store_rejects_escaping_paths(tmp_path):
    with pytest.raises(ValueError, match="invalid paper state path"):
        PaperStateStore.from_context(_context(tmp_path), "../paper_state.pkl")
