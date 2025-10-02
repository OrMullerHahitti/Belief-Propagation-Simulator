import numpy as np

from propflow.snapshots.types import SnapshotData, SnapshotRecord
from propflow.snapshots.utils import (
    latest_cycles,
    latest_jacobians,
    latest_snapshot,
    latest_winners,
    get_snapshot,
)


def _make_record(step: int = 0) -> SnapshotRecord:
    data = SnapshotData(
        step=step,
        lambda_=0.1,
        dom={"x1": ["0", "1"]},
        N_var={"x1": ["f1"]},
        N_fac={"f1": ["x1"]},
        Q={("x1", "f1"): np.array([0.0, 1.0])},
        R={("f1", "x1"): np.array([1.5, 0.3])},
    )
    return SnapshotRecord(data=data, jacobians=None, cycles=None, winners={"w": 1})


def test_latest_snapshot_helpers():
    record = _make_record()

    class DummyEngine:
        def latest_snapshot(self):
            return record

        def get_snapshot(self, idx):
            return record if idx == 0 else None

    engine = DummyEngine()

    assert latest_snapshot(engine) is record
    assert latest_jacobians(engine) is None
    assert latest_cycles(engine) is None
    assert latest_winners(engine) == {"w": 1}
    assert get_snapshot(engine, 0) is record
    assert get_snapshot(engine, 5) is None
