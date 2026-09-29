"""Population stages reuse real evidence and continue split message states exactly."""

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest

from experiments.aamas.late_split import core, domains, population, resume
from propflow import DampingEngine, DampingSCFGEngine


@pytest.mark.parametrize("kind", ["released", "damping", "damped_split"])
def test_runtime_checkpoint_continues_exact_native_state(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(domains, "NUM_AGENTS", 5)
    original = domains.build_dense(2, 20)
    cfg = core.Config(prefix_steps=8, post_steps=12, tail_steps=4)
    if kind == "released":
        engine = core.make_engine(deepcopy(original), cfg, 8)
    else:
        cls = DampingEngine if kind == "damping" else DampingSCFGEngine
        engine = cls(
            deepcopy(original),
            damping_factor=0.9,
            normalize_messages=True,
            anytime=False,
            snapshot_manager=core.TraceSnapshots(),
        )
    snapshots = [core.advance(engine, i) for i in range(13)]
    trace = core.trace_arrays(snapshots, [v.name for v in engine.var_nodes])
    resume.save_runtime(tmp_path, engine, 13, trace)
    restored, saved, next_i = resume.restore_runtime(tmp_path, original, cfg)
    np.testing.assert_array_equal(saved["assignments"], trace["assignments"])
    assert next_i == 13
    for i in range(13, 30):
        a, b = core.advance(engine, i), core.advance(restored, i)
        assert a.assignments == b.assignments
        assert a.global_cost == b.global_cost
    for actual in [engine, restored]:
        state = core.Checkpoint.capture(
            actual, 30, core.input_fingerprint(actual.graph), runtime_graph=True
        )
        serialized = json.dumps(asdict(state), default=core.json_value, sort_keys=True)
        if actual is engine:
            expected = serialized
        else:
            assert serialized == expected


def test_population_partial_rerun_does_not_repeat_finished_bp(tmp_path, monkeypatch):
    monkeypatch.setattr(domains, "NUM_AGENTS", 5)
    cfg = core.Config(prefix_steps=8, post_steps=8, tail_steps=4)
    case = tmp_path / "random_dense_0"
    case.mkdir()
    (tmp_path / "mgm_tail_values").mkdir()
    task = (str(tmp_path), 0, "fixed", cfg)
    first = population.run_bp(task)
    trace = resume.read_arrays(case / "fixed_trace.npz")
    # retain the checkpoint but emulate interruption before writing the phase result
    (case / "fixed_result.json").unlink()
    (case / "fixed_trace.npz").unlink()
    monkeypatch.setattr(population, "advance", lambda *a: pytest.fail("BP repeated"))
    second = population.run_bp(task)
    assert first["final_cost"] == second["final_cost"]
    np.testing.assert_array_equal(
        trace["assignments"],
        resume.read_arrays(case / "fixed_trace.npz")["assignments"],
    )
    assert population.run_mgm(task)["bb_run"] is False
    assert population.run_bp(task) == second
    with (case / "fixed_trace.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="changed phase evidence"):
        population.run_bp(task)


def test_mgm_average_keeps_terminated_seeds_in_denominator():
    from experiments.aamas.late_split.plot_population import align_history

    no_moves = align_history([90.0], 2)
    two_moves = align_history([120.0, 105.0, 95.0], 2)
    np.testing.assert_array_equal(
        np.mean([no_moves, two_moves], axis=0), [105, 97.5, 92.5]
    )


def test_best_phase_replays_checkpoint_captured_by_plain_dms(tmp_path, monkeypatch):
    monkeypatch.setattr(domains, "NUM_AGENTS", 5)
    cfg = core.Config(prefix_steps=15, post_steps=12, tail_steps=4)
    (tmp_path / "random_dense_1").mkdir()
    population.run_bp((str(tmp_path), 1, "fixed", cfg))
    result = population.run_bp((str(tmp_path), 1, "best", cfg))
    prefix = resume.read_arrays(tmp_path / "random_dense_1" / "prefix.npz")
    assert result["split_before_iteration"] == int(np.argmin(prefix["costs"])) + 1
    assert result["checkpoint_replay_max_error"] == 0


def test_interrupted_mid_continuation_reuses_checkpoint_and_preserves_trace(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(domains, "NUM_AGENTS", 5)
    cfg = core.Config(prefix_steps=8, post_steps=520, tail_steps=20)
    case = tmp_path / "random_dense_0"
    case.mkdir()
    real_save = population.save_runtime

    def interrupted_save(directory, *args):
        real_save(directory, *args)
        if directory.name == "000250":
            raise InterruptedError("simulated stop after committed checkpoint")

    monkeypatch.setattr(population, "save_runtime", interrupted_save)
    task = (str(tmp_path), 0, "fixed", cfg)
    with pytest.raises(InterruptedError):
        population.run_bp(task)
    monkeypatch.setattr(population, "save_runtime", real_save)
    population.run_bp(task)
    actual = resume.read_arrays(case / "fixed_trace.npz")
    reference = core.make_engine(core.load_input(case / "input.npz"), cfg, 8)
    core.Checkpoint.load(case / "fixed_checkpoint.json.gz").restore(reference)
    expected = core.trace_arrays(
        [core.advance(reference, i) for i in range(8, 528)],
        [v.name for v in reference.var_nodes],
    )
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
