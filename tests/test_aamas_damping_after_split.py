"""Damping handoff uses actual Q, preserves R, and supports exact partial reruns."""

from copy import deepcopy
from types import MethodType

import numpy as np
import pytest

from propflow import DampingEngine
from experiments.aamas.late_split import core, domains, resume
from experiments.aamas.damping_after_split import run, state


def saved_parent(directory, monkeypatch, checkpoint=True):
    monkeypatch.setattr(domains, "NUM_AGENTS", 5)
    original = domains.build_dense(4, 3)
    cfg = core.Config(prefix_steps=8, post_steps=12, tail_steps=4)
    directory.mkdir(parents=True)
    core.save_input(original, directory / "input.npz")
    engine = core.make_engine(deepcopy(original), cfg, 8)
    for i in range(8):
        core.advance(engine, i)
    core.Checkpoint.capture(engine, 8, core.input_fingerprint(original)).save(
        directory / "best_checkpoint.json.gz"
    )
    capture = state.CaptureLastQ(19)
    engine.snapshot_manager = capture
    snapshots = []
    for i in range(8, 20):
        snapshots.append(core.advance(engine, i))
        if checkpoint and i in [15, 19]:
            resume.save_runtime(
                directory / "checkpoints/best" / f"{i - 7:06d}",
                engine,
                i + 1,
                core.trace_arrays(snapshots, [v.name for v in engine.var_nodes]),
            )
    np.savez_compressed(
        directory / "best_trace.npz",
        **core.trace_arrays(snapshots, [v.name for v in engine.var_nodes]),
    )
    return original, cfg, engine, capture.messages


@pytest.mark.parametrize("checkpoint", [False, True])
def test_suffix_recovery_matches_full_state_and_actual_last_q(
    tmp_path, monkeypatch, checkpoint
):
    case = tmp_path / "case"
    original, cfg, live, last_q = saved_parent(case, monkeypatch, checkpoint)
    recovered, actual_q, next_i, record = state.recover_terminal(case, original, cfg)
    assert next_i == 20
    assert record["replayed_updates"] == (4 if checkpoint else 12)
    assert state.serialized_state(live, 20) == state.serialized_state(recovered, 20)
    for name in last_q:
        for a, b in zip(last_q[name], actual_q[name]):
            assert a.recipient.name == b.recipient.name
            np.testing.assert_array_equal(a.data, b.data)


def test_first_update_uses_damped_q_and_preserves_handoff_state(tmp_path, monkeypatch):
    original, cfg, engine, last_q = saved_parent(tmp_path / "case", monkeypatch)
    trace = resume.read_arrays(tmp_path / "case/best_trace.npz")
    resume.save_runtime(tmp_path / "undamped", engine, 20, trace)
    damped, _, _ = state.restore_damped(tmp_path / "undamped", original)
    state.install_previous_q(damped, last_q)
    damped.damping_factor = cfg.damping
    before = state.serialized_state(damped, 20)
    check = state.first_update_check(damped)
    assert check["formula_max_error"] == 0
    assert check["max_change_from_undamped_q"] > 0
    assert before == state.serialized_state(damped, 20)

    expected = {}
    for var in damped.var_nodes:
        var.compute_messages()
        prior = {m.recipient.name: m.data for m in last_q[var.name]}
        expected[var.name] = {
            m.recipient.name: 0.9 * prior[m.recipient.name] + 0.1 * m.data
            for m in var.mailer.outbox
        }
        var.mailer.prepare()
    capture = state.CaptureLastQ(20)
    damped.snapshot_manager = capture
    core.advance(damped, 20)
    for name, messages in capture.messages.items():
        for message in messages:
            np.testing.assert_allclose(
                message.data,
                expected[name][message.recipient.name],
                rtol=1e-14,
                atol=1e-12,
            )
    assert core.input_fingerprint(damped.graph) == core.input_fingerprint(engine.graph)


def test_live_handoff_matches_serialized_native_damping_and_resume(
    tmp_path, monkeypatch
):
    original, cfg, live, last_q = saved_parent(tmp_path / "case", monkeypatch)
    trace = resume.read_arrays(tmp_path / "case/best_trace.npz")
    resume.save_runtime(tmp_path / "undamped", live, 20, trace)
    restored, _, _ = state.restore_damped(tmp_path / "undamped", original)
    # use the native damping methods directly on the live graph as the reference
    live.step = MethodType(DampingEngine.step, live)
    live.post_var_compute = MethodType(DampingEngine.post_var_compute, live)
    for engine in [live, restored]:
        state.install_previous_q(engine, last_q)
        engine.damping_factor = cfg.damping
    snapshots = []
    for i in range(20, 32):
        actual, expected = core.advance(restored, i), core.advance(live, i)
        snapshots.append(actual)
        assert actual.assignments == expected.assignments
        assert actual.global_cost == expected.global_cost
        assert state.serialized_state(live, i + 1) == state.serialized_state(
            restored, i + 1
        )
        if i == 24:
            resume.save_runtime(
                tmp_path / "resumed",
                restored,
                i + 1,
                core.trace_arrays(snapshots, list(actual.assignments)),
            )
            restored, _, next_i = state.restore_damped(tmp_path / "resumed", original)
            assert next_i == i + 1


def test_interruption_and_completed_seed_reuse_do_not_repeat_updates(
    tmp_path, monkeypatch
):
    source = tmp_path / "parent_evidence/random_dense_0"
    original, cfg, _, _ = saved_parent(source, monkeypatch)
    save = run.save_runtime

    def interrupt(directory, *args):
        save(directory, *args)
        if directory.name == "000005":
            raise InterruptedError("checkpoint committed")

    task = (str(tmp_path), 0, cfg, 12, 5)
    monkeypatch.setattr(run, "save_runtime", interrupt)
    with pytest.raises(InterruptedError):
        run.run_seed(task)
    monkeypatch.setattr(run, "save_runtime", save)
    first = run.run_seed(task)
    trace = resume.read_arrays(tmp_path / "random_dense_0/damped_trace.npz")
    initial, _, next_i = state.restore_damped(
        tmp_path / "random_dense_0/checkpoints/000000", original
    )
    expected = core.trace_arrays(
        [core.advance(initial, i) for i in range(next_i, next_i + 12)],
        list(trace["variable_names"]),
    )
    for key in trace:
        np.testing.assert_array_equal(trace[key], expected[key])
    monkeypatch.setattr(
        run, "advance", lambda *a: pytest.fail("completed seed repeated")
    )
    assert run.run_seed(task) == first
