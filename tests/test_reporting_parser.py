"""Unit tests for the reporting snapshot parser utilities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analyzer.reporting import (
    SnapshotRecord,
    load_snapshots,
    parse_snapshots,
)


def _make_raw_snapshot() -> list[dict]:
    return [
        {
            "step": 0,
            "messages": [
                {
                    "flow": "variable_to_factor",
                    "sender": "x1",
                    "recipient": "f12",
                    "values": [0.0, 1.0],
                    "argmin_index": 0,
                    "neutral": False,
                },
                {
                    "flow": "factor_to_variable",
                    "sender": "f12",
                    "recipient": "x2",
                    "values": [0.5, 0.5],
                    "argmin_index": 0,
                    "neutral": True,
                },
            ],
            "assignments": {"x1": 1, "x2": 0},
            "cost": 3.25,
            "neutral_messages": 1,
            "step_neutral": False,
        }
    ]


def test_parse_snapshots_roundtrip() -> None:
    raw = _make_raw_snapshot()
    records = parse_snapshots(raw)
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, SnapshotRecord)
    assert record.step == 0
    assert record.messages[1].neutral is True
    assert record.neutral_messages == 1
    assert record.assignments["x1"] == 1


def test_parse_snapshots_detects_neutral_mismatch() -> None:
    raw = _make_raw_snapshot()
    raw[0]["neutral_messages"] = 0
    with pytest.raises(ValueError):
        parse_snapshots(raw)


def test_load_snapshots(tmp_path: Path) -> None:
    raw = _make_raw_snapshot()
    file_path = tmp_path / "snap.json"
    file_path.write_text(json.dumps(raw))
    loaded = load_snapshots(file_path)
    assert loaded == raw


def test_from_engine_snapshot_manager_neutral_detection() -> None:
    from analyzer.reporting.snapshot_parser import from_engine_snapshot_manager

    class DummyData:
        step = 1
        dom = {"x1": ["0", "1"]}
        Q = {("x1", "f1"): np.array([0.0, 1.0])}
        R = {("f1", "x1"): np.array([0.5, 0.5])}

    class DummyRecord:
        data = DummyData()
        min_idx = {("x1", "f1"): 0}

    record = from_engine_snapshot_manager(DummyRecord())
    assert record.step == 1
    assert record.neutral_messages == 1
    assert record.assignments["x1"] == 0
