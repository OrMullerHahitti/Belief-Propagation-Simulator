"""Behavioral checks for the isolated splitting/damping control study."""

import numpy as np
import pytest

from experiments.aamas.aaai_derived_control.code import kernel as runtime
from experiments.aamas.aaai_state_control import core, native_replay, study


def problem():
    return runtime.make_small_problem("k4", "frustrated", 190)


@pytest.mark.parametrize("schedule", core.schedules()[:8])
def test_interventions_preserve_messages_and_original_objective(schedule):
    k = runtime.PairwiseKernel(problem()).advance(64)
    before = [getattr(k, name).copy() for name in ("q", "r", "unary_q", "unary_r")]
    schedule.apply(k)
    for name, expected in zip(("q", "r", "unary_q", "unary_r"), before):
        np.testing.assert_array_equal(getattr(k, name), expected)
    tables = k.tables.reshape(len(k.weights), 2, k.problem.d, k.problem.d)
    np.testing.assert_allclose(tables.sum(axis=1), k.problem.costs, rtol=0, atol=1e-12)


def test_minimizing_rows_match_explicit_next_message_choices():
    k = runtime.PairwiseKernel(runtime.make_problem("bowtie", 311, d=3)).advance(37)
    rows, margins = core.row_state(k)
    trial = k.clone().step()
    for clone, table in enumerate(trial.tables):
        for recipient in range(3):
            assert rows[clone, 0, recipient] == np.argmin(
                table[:, recipient] + trial.q[clone, 0]
            )
            assert rows[clone, 1, recipient] == np.argmin(
                table[recipient, :] + trial.q[clone, 1]
            )
    assert np.all(margins >= 0)


def test_all_messages_enter_undamped_defect():
    k = runtime.PairwiseKernel(problem()).advance(2000)
    before = core.fixed_defect(k)
    k.unary_q[0, 0, 1] += k.problem.scale
    assert core.fixed_defect(k) > before + 0.5


def test_online_update_uses_only_observed_arm():
    model = core.LinearSelector()
    before_a, before_b = model.a.copy(), model.b.copy()
    feature = np.ones(9)
    model.update(3, feature, np.array([-0.1, 0.2]))
    for arm in (0, 1, 2, 4):
        np.testing.assert_array_equal(model.a[arm], before_a[arm])
        np.testing.assert_array_equal(model.b[arm], before_b[arm])
    assert not np.array_equal(model.b[3], before_b[3])


def test_six_online_decisions_receive_feedback_before_later_choices(tmp_path):
    path = tmp_path / "model.npz"
    model = core.LinearSelector()
    model.save(path)
    result = study.run_case(problem(), runtime, "online", 2000, path)
    decisions = result[4]
    assert [d["step"] for d in decisions] == [32, 288, 544, 800, 1056, 1312]
    assert all(d["online_update"] for d in decisions)
    assert all("observed_target" in d for d in decisions)
    observed_steps = {item["step"] for item in result[3]}
    assert all(d["step"] - 8 in observed_steps for d in decisions)
    assert not np.array_equal(result[-1].a, model.a)
    assert study.verify_costs(problem(), result[1]) < 1e-10


def test_frozen_exploration_and_online_share_random_draws(tmp_path):
    path = tmp_path / "model.npz"
    core.LinearSelector().save(path)
    frozen = study.run_case(problem(), runtime, "frozen_explore", 2000, path)
    online = study.run_case(problem(), runtime, "online", 2000, path)
    assert [d["exploration_draw"] for d in frozen[4]] == [
        d["exploration_draw"] for d in online[4]
    ]
    np.testing.assert_array_equal(frozen[-1].a, core.LinearSelector().a)


def test_local_action_only_changes_edges_with_predicted_crossings():
    k = runtime.PairwiseKernel(problem()).advance(24)
    selected = core.action_edges(k, 4)
    before = core.row_state(k)[0]
    after = core.row_state(k, 0.95)[0]
    counts = np.any((before != after).reshape(len(k.weights), -1), axis=1)
    assert np.all(counts[selected])
    assert len(selected) <= int(np.ceil(len(k.weights) / 4))


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        study.run_case(problem(), runtime, "typo", 2000, None)


def test_saved_damping_switch_replays_in_native_engine(tmp_path, monkeypatch):
    for name in ("inputs", "events", "trajectories"):
        (tmp_path / name).mkdir()
    p = problem()
    key = study.save_problem(tmp_path, p)
    method = "pulse_no_damping_during"
    result = study.run_case(p, runtime, method, 300, None)
    np.savez_compressed(tmp_path / "trajectories" / f"{key}_{method}.npz", **result[1])
    study.write_json(
        tmp_path / "events" / f"{key}_{method}.json", {"events": result[2]}
    )
    monkeypatch.setattr(native_replay, "load_runtime", lambda _: runtime)
    report = native_replay.replay(tmp_path, p.family, p.seed, method)
    assert report["passed"], report
