import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from propflow.snapshots.types import SnapshotData, SnapshotRecord
from propflow.snapshots.visualizer import SnapshotVisualizer


def _make_snapshot(step: int, cost: float | None) -> SnapshotRecord:
    data = SnapshotData(
        step=step,
        lambda_=0.0,
        dom={"x1": ["0", "1"]},
        N_var={"x1": ["f1"]},
        N_fac={"f1": ["x1"]},
        Q={("x1", "f1"): np.array([0.0, 0.1])},
        R={("f1", "x1"): np.array([0.0, 0.2])},
        cost={},
        cost_tables={},
        cost_labels={},
        unary={"x1": np.array([0.0, 0.1])},
        beliefs={"x1": float(step)},
        assignments={"x1": step % 2},
        global_cost=cost,
        metadata={},
    )
    return SnapshotRecord(data=data)


def _make_message_snapshot(
    step: int,
    q_messages: dict[tuple[str, str], np.ndarray],
    r_messages: dict[tuple[str, str], np.ndarray],
) -> SnapshotRecord:
    data = SnapshotData(
        step=step,
        lambda_=0.0,
        dom={"x1": ["0", "1"], "x2": ["0", "1"]},
        N_var={"x1": ["f"], "x2": ["f"]},
        N_fac={"f": ["x1", "x2"]},
        Q=q_messages,
        R=r_messages,
        cost={},
        cost_tables={},
        cost_labels={},
        unary={},
        beliefs={},
        assignments={"x1": step % 2, "x2": (step + 1) % 2},
        global_cost=None,
        metadata={},
    )
    return SnapshotRecord(data=data)


def test_plot_global_cost_returns_expected_series(tmp_path) -> None:
    snapshots = [
        _make_snapshot(0, 10.0),
        _make_snapshot(1, 6.0),
        _make_snapshot(2, 8.0),
        _make_snapshot(3, 4.0),
    ]
    visualizer = SnapshotVisualizer(snapshots)

    save_file = tmp_path / "global_cost.png"
    fig, payload = visualizer.plot_global_cost(
        show=False,
        savepath=str(save_file),
        return_data=True,
        rolling_window=2,
    )

    assert payload["steps"] == [0, 1, 2, 3]
    assert payload["costs"] == [10.0, 6.0, 8.0, 4.0]
    expected_rolling = [8.0, 7.0, 6.0]
    assert payload["rolling"]["window"] == 2
    assert payload["rolling"]["steps"] == [1, 2, 3]
    np.testing.assert_allclose(payload["rolling"]["values"], expected_rolling)
    assert save_file.exists()

    plt.close(fig)


def test_plot_message_norms_computes_expected_values(tmp_path) -> None:
    snapshots = [
        _make_message_snapshot(
            0,
            {("x1", "f"): np.array([1.0, 0.0]), ("x2", "f"): np.array([0.5, 0.5])},
            {("f", "x1"): np.array([0.5, 1.5]), ("f", "x2"): np.array([0.0, 0.3])},
        ),
        _make_message_snapshot(
            1,
            {("x1", "f"): np.array([0.0, 2.0]), ("x2", "f"): np.array([0.2, 0.8])},
            {("f", "x1"): np.array([1.0, 1.0]), ("f", "x2"): np.array([0.4, 0.1])},
        ),
    ]

    visualizer = SnapshotVisualizer(snapshots)

    q_save = tmp_path / "message_norms_q.png"
    fig_q, q_payload = visualizer.plot_message_norms(
        message_type="Q",
        pairs=[("x1", "f")],
        norm="l2",
        show=False,
        savepath=str(q_save),
        return_data=True,
    )

    assert q_payload["message_type"] == "Q"
    assert q_payload["norm"] == "l2"
    np.testing.assert_allclose(
        q_payload["series"][("x1", "f")],
        [1.0, 2.0],
    )
    assert q_save.exists()

    plt.close(fig_q)

    r_save = tmp_path / "message_norms_r.png"
    fig_r, r_payload = visualizer.plot_message_norms(
        message_type="R",
        pairs=[("f", "x1")],
        norm="linf",
        show=False,
        savepath=str(r_save),
        return_data=True,
    )

    assert r_payload["message_type"] == "R"
    assert r_payload["norm"] == "linf"
    np.testing.assert_allclose(
        r_payload["series"][("f", "x1")],
        [1.5, 1.0],
    )
    assert r_save.exists()

    plt.close(fig_r)


def test_plot_assignment_heatmap_builds_matrix(tmp_path) -> None:
    assignments_per_step = [
        {"x1": 0, "x2": 1},
        {"x1": 2},
        {"x1": 1, "x2": 0},
    ]

    snapshots = []
    for step, assigns in enumerate(assignments_per_step):
        data = SnapshotData(
            step=step,
            lambda_=0.0,
            dom={"x1": ["0", "1", "2"], "x2": ["0", "1", "2"]},
            N_var={"x1": [], "x2": []},
            N_fac={},
            Q={},
            R={},
            cost={},
            cost_tables={},
            cost_labels={},
            unary={},
            beliefs={},
            assignments=assigns,
            global_cost=None,
            metadata={},
        )
        snapshots.append(SnapshotRecord(data=data))

    visualizer = SnapshotVisualizer(snapshots)

    save_path = tmp_path / "assignment_heatmap.png"
    fig, payload = visualizer.plot_assignment_heatmap(
        vars_filter=["x1", "x2"],
        show=False,
        annotate=False,
        missing_value=-1.0,
        savepath=str(save_path),
        return_data=True,
    )

    assert payload["variables"] == ["x1", "x2"]
    assert payload["steps"] == [0, 1, 2]
    expected_matrix = np.array(
        [
            [0.0, 2.0, 1.0],
            [1.0, -1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(payload["matrix"], expected_matrix)
    assert save_path.exists()

    plt.close(fig)
