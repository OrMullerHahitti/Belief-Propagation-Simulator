"""Checks for the paired node-dynamics experiment's scientific contract."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.dabp_node_dynamics.analysis import (
    StabilityWindow,
    applied_coefficients,
    node_changes,
    node_split_changes,
    paired_sources,
    split_balance,
)
from experiments.dabp_node_dynamics.graph import (
    Settings,
    create_problem,
    fingerprint,
    restore_problem,
)


def small_settings(**kwargs):
    return Settings(
        nodes=4, density=0.5, domain=3, stable_window=4, update_interval=2, **kwargs
    )


def test_saved_problem_preserves_original_tables_and_axes():
    settings = small_settings()
    problem = create_problem(settings)
    assert fingerprint(problem) == fingerprint(create_problem(settings))
    graph = restore_problem(json.loads(json.dumps(problem)))
    for expected, actual in zip(problem["factors"], graph.original_factors):
        np.testing.assert_array_equal(actual.cost_table, expected["table"])
        assert list(actual.connection_number) == expected["variables"]
    assert len(problem["nodes"]) == 4
    assert all(n["degree"] == len(n["neighbors"]) for n in problem["nodes"])


def test_stability_requires_whole_window_not_small_adjacent_steps():
    monitor = StabilityWindow(4, 0.001)
    for iteration in range(8):
        state = monitor.update(np.array([0, 1]), np.array([iteration * 0.0006]))
        assert not state.converged
    for _ in range(4):
        state = monitor.update(np.array([0, 1]), np.array([0.5]))
    assert state.converged
    assert not monitor.update(np.array([1, 1]), np.array([0.5])).converged
    with pytest.raises(ValueError, match="non-finite"):
        monitor.update(np.array([1, 1]), np.array([np.nan]))


def test_opposing_changes_are_not_cancelled_by_averaging():
    series = np.array([[0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
    result = node_changes(series, np.array([0, 0]), 1)[0]
    assert result["initial"] == result["final"] == 0.5
    assert result["movement"] == pytest.approx(0.5)
    assert result["jump"] == pytest.approx(0.3)


def test_split_adjustment_matches_example_and_omits_missing_halves():
    assert split_balance(np.array([0.45]), np.array([10.0]), 0.95)[0] == pytest.approx(
        0.4275 / 0.9275
    )
    meta = {
        "src_fn_idxes": [0, 1, 0],
        "src_trg_idxes": [0, 0, 1],
        "fn_factor_names": ["f", "f"],
        "fn_half": [0, 1],
    }
    assert paired_sources(meta) == [{"target": 0, "factor": "f", "rows": [0, 1]}]


def test_coefficients_use_product_of_head_means():
    damped = np.array([[[0.1, 0.9], [0.9, 0.1]]])
    attention = np.array([[0.2, 0.8], [0.8, 0.2]])
    old, share, coefficient = applied_coefficients(damped, attention, np.array([0, 0]))
    np.testing.assert_allclose(old, [0.5])
    np.testing.assert_allclose(share, [0.5, 0.5])
    np.testing.assert_allclose(coefficient, [0.5, 0.5])
    assert not np.allclose(coefficient, 2 * (attention * damped[0, 0]).mean(axis=1))


def test_node_split_changes_keeps_excursions_and_exact_pair_provenance():
    coefficients = np.array(
        [
            [0.5, 0.5, 0.5, 0.5, 0.2, 0.8],
            [0.9, 0.1, 0.4, 0.6, 0.2, 0.8],
            [0.5, 0.5, 0.7, 0.3, 0.2, 0.8],
        ]
    )
    pairs = [
        {"target": 0, "factor": "a", "rows": [0, 1]},
        {"target": 0, "factor": "b", "rows": [2, 3]},
        {"target": 1, "factor": "c", "rows": [4, 5]},
    ]
    result = node_split_changes(coefficients, pairs, np.array([0, 1]), 3, 0.5)
    assert result[0] == {
        "pair_index": 0,
        "pair_count": 2,
        "initial": 50.0,
        "final": 50.0,
        "departure": 40.0,
        "peak_iteration": 2,
    }
    assert result[1]["initial"] == result[1]["final"] == 20.0
    assert result[1]["departure"] == 0.0
    assert result[2] is None
    assert node_split_changes(coefficients, [], np.array([0, 1]), 3, 0.5) == [None] * 3


def test_reader_encoding_preserves_every_float64_value():
    import base64
    import zlib
    from experiments.dabp_node_dynamics.report import encode_array

    values = np.array([[0.0, 0.5000000000000001], [1e-15, 0.4999999999999999]])
    encoded = encode_array(values)
    restored = (
        np.frombuffer(zlib.decompress(base64.b64decode(encoded["data"])), dtype="<f8")
        .reshape(values.shape[::-1])
        .T
    )
    np.testing.assert_array_equal(restored, values)


def test_reader_entry_opens_the_report_instead_of_exposing_the_template():
    root = Path(__file__).resolve().parents[1] / "experiments" / "dabp_node_dynamics"
    entry = (root / "reader.html").read_text()
    template = (root / "reader.template.html").read_text()
    assert 'href="outputs/20nodes_seed0/report.html"' in entry
    assert 'href="outputs/50nodes_seed0/report.html"' in entry
    assert 'href="http://127.0.0.1:8769/report.html"' in entry
    assert 'href="http://127.0.0.1:8768/report.html"' in entry
    for placeholder in ("__STYLE__", "__DATA__", "__PLOTLY__", "__SCRIPT__"):
        assert placeholder not in entry
        assert placeholder in template


@pytest.mark.parametrize("variant", ["symmetric", "asymmetric"])
def test_coefficients_reconstruct_actual_engine_messages_across_training(variant):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from propflow.integrations.dabp import DABPEngine, DABPEngineSymSplit
    from experiments.dabp_node_dynamics.run import CostSnapshotManager

    torch.set_num_threads(1)
    torch.manual_seed(0)
    cls = DABPEngineSymSplit if variant == "symmetric" else DABPEngine
    engine = cls(
        factor_graph=restore_problem(create_problem(small_settings())),
        device="cpu",
        record_weights=True,
        update_interval=2,
        snapshot_manager=CostSnapshotManager(),
    )
    engine.step(0)
    meta = engine.weight_metadata()
    model = engine._require_abp()
    source_target = np.array(meta["src_trg_idxes"])
    for iteration in range(1, 6):
        old_messages = model.msgs[model.msg_trg_idxes].detach().numpy().copy()
        engine.step(iteration)
        record = engine.weights_log[-1]
        damping, _, coefficient = applied_coefficients(
            record["damped_weights"], record["attention_weight"], source_target
        )
        incoming = model.msgs[model.msg_src_idxes].detach().numpy()
        reconstructed = damping[:, None] * old_messages
        np.add.at(reconstructed, source_target, coefficient[:, None] * incoming)
        reconstructed -= reconstructed.min(axis=1, keepdims=True)
        actual = model.msgs[model.msg_trg_idxes].detach().numpy()
        np.testing.assert_allclose(reconstructed, actual, rtol=1e-12, atol=1e-12)


def test_small_run_archives_every_iteration_and_rebuilds_report(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from experiments.dabp_node_dynamics.run import run_variant
    from experiments.dabp_node_dynamics.report import build_report
    from dataclasses import asdict

    settings = small_settings(max_iterations=5)
    problem = create_problem(settings)
    (tmp_path / "graph.json").write_text(json.dumps(problem))
    manifest = {
        "settings": asdict(settings),
        "graph_sha256": fingerprint(problem),
        "runs": {},
    }
    for variant in ("symmetric", "asymmetric"):
        path = tmp_path / f"{variant}.npz"
        outcome = run_variant(problem, settings, variant, path)
        manifest["runs"][variant] = outcome
        with np.load(path, allow_pickle=False) as archive:
            assert len(archive["damped"]) == outcome["iterations"]
            assert len(archive["attention"]) == len(archive["assignments"])
            np.testing.assert_array_equal(
                archive["iteration"], np.arange(1, outcome["iterations"] + 1)
            )
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    report = build_report(tmp_path)
    text = report.read_text()
    assert "__DATA__" not in text and "__SCRIPT__" not in text
    assert "Split balance" in text
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "asymmetric.npz",
        "graph.json",
        "report.html",
        "run.json",
        "symmetric.npz",
    ]
    problem["nodes"][0]["degree"] += 1
    (tmp_path / "graph.json").write_text(json.dumps(problem))
    with pytest.raises(ValueError, match="checksum"):
        build_report(tmp_path)
