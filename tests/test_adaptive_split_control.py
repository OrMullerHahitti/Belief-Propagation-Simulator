"""Research gates for objective preservation, runtime parity, and online learning."""

import numpy as np
import pytest

from experiments.other.adaptive_split_control.code.lab import (
    Action,
    PairwiseLab,
    TinyScorer,
    gauge,
    make_problem,
    native_parity,
)


@pytest.mark.parametrize("split", [None, 0.5, 0.95])
@pytest.mark.parametrize("damping", [0.0, 0.9])
def test_native_runtime_parity_including_live_intervention(split, damping):
    problem = make_problem("bowtie", "random", 13)
    assert native_parity(problem, split, damping) < 1e-7


def test_dynamic_split_preserves_objective_and_messages():
    problem = make_problem("bowtie", "random", 5)
    lab = PairwiseLab(problem)
    lab.advance(7)
    q, r = lab.q.copy(), lab.r.copy()
    for edge in range(6):
        lab.act(Action(edge, 0.05 + 0.15 * edge, 0.5))
    np.testing.assert_array_equal(lab.q, q)
    np.testing.assert_array_equal(lab.r, r)
    np.testing.assert_allclose(
        lab.tables.reshape(6, 2, 3, 3).sum(axis=1), problem.costs
    )


def test_symmetric_clone_reduction_includes_sibling_return():
    problem = make_problem("bowtie", "random", 7)
    lab = PairwiseLab(problem, 0.5, 0)
    for _ in range(12):
        b = lab.beliefs()
        m = lab.r.reshape(6, 2, 2, 3).sum(axis=1)
        expected = []
        for edge, (u, v) in enumerate(problem.edges):
            left = (problem.costs[edge] + (2 * b[v] - m[edge, 1])[None]).min(axis=1)
            right = (problem.costs[edge] + (2 * b[u] - m[edge, 0])[:, None]).min(axis=0)
            expected.append(gauge(np.stack((left, right))))
        lab.step()
        clones = lab.r.reshape(6, 2, 2, 3)
        np.testing.assert_allclose(clones[:, 0], clones[:, 1], atol=1e-9)
        np.testing.assert_allclose(clones.sum(axis=1), expected, atol=1e-9)


def test_mlp_really_learns_and_has_145_parameters():
    model = TinyScorer(0)
    assert sum(a.size for a in (model.w1, model.b1, model.w2, model.b2)) == 145
    x = np.random.default_rng(1).normal(size=(64, 16))
    y = 0.2 * np.tanh(x[:, 0])
    before = np.mean((model.predict(x) - y) ** 2)
    for _ in range(1000):
        model.update(x, y, rate=0.2)
    assert np.mean((model.predict(x) - y) ** 2) < before / 2


def test_features_do_not_require_optimum(monkeypatch):
    problem = make_problem("bowtie", "random", 5)
    lab = PairwiseLab(problem)
    lab.advance(8)

    def forbidden(*args):
        raise AssertionError("online features must not query the exact solution")

    monkeypatch.setattr(type(problem), "exact", forbidden)
    features = lab.features(lab.actions(), 128)
    assert features.shape == (39, 16)
    assert np.isfinite(features).all()
    assert np.isfinite(lab.reward(lab.costs[0], 8))


@pytest.mark.parametrize("action", [Action(0, 0, 0.5), Action(0, 0.5, 1)])
def test_invalid_actions_do_not_partially_mutate_state(action):
    lab = PairwiseLab(make_problem("bowtie", "random", 0))
    weights, damping = lab.weights.copy(), lab.damping
    with pytest.raises(ValueError):
        lab.act(action)
    np.testing.assert_array_equal(lab.weights, weights)
    assert lab.damping == damping


def test_split_changes_active_regions_while_damping_changes_derivative():
    from experiments.other.adaptive_split_control.code.theory import (
        active_jacobian,
        q_map,
    )

    lab = PairwiseLab(make_problem("bowtie", "random", 17), 0.5, 0)
    q = gauge(np.random.default_rng(4).normal(0, 0.2, lab.q.shape))
    jac, active, margin = active_jacobian(q, lab.tables, lab.ends)
    assert margin > 1e-6
    epsilon = 1e-7
    numeric = np.zeros_like(jac)
    for column in range(jac.shape[1]):
        perturbation = np.zeros_like(q)
        view = perturbation[..., 1:]
        view.flat[column] = epsilon
        plus = q_map(q + perturbation, lab.tables, lab.ends, 0.7)
        minus = q_map(q - perturbation, lab.tables, lab.ends, 0.7)
        numeric[:, column] = ((plus - minus) / (2 * epsilon))[..., 1:].reshape(-1)
    np.testing.assert_allclose(numeric, 0.7 * np.eye(len(jac)) + 0.3 * jac, atol=1e-7)
    lab.act(Action(0, 0.5 + epsilon, 0))
    changed_jac, changed_active, _ = active_jacobian(q, lab.tables, lab.ends)
    np.testing.assert_array_equal(active, changed_active)
    np.testing.assert_array_equal(jac, changed_jac)


def test_settled_online_keeps_split_and_stops_weight_updates(monkeypatch):
    from experiments.other.adaptive_split_control.code.run import run_policy

    problem = make_problem("bowtie", "random", 1)

    def forbidden(*args):
        raise AssertionError("online execution must not query exact solutions")

    monkeypatch.setattr(type(problem), "exact", forbidden)
    lab, trace, _, updates = run_policy(problem, "online", settle_after=16)
    assert updates == 16
    settled = [row for row in trace if row["step"] >= 16]
    assert all(row["phase"] == "settle" for row in settled)
    assert all(row["weights"] == settled[0]["weights"] for row in settled)
    assert lab.t == 128


def test_split_perturbation_is_invisible_until_active_choices_change():
    from experiments.other.adaptive_split_control.code.symmetry_probe import probe

    row = probe(make_problem("bowtie", "random", 17), 0, 1e-6)
    assert row["first_active_change"] is None
    assert row["same_region_max_error"] < 1e-12
    assert row["same_region_clone_change"] > 1e-7
    assert row["final_cost_change"] == 0


def test_frustrated_roundoff_tie_matches_native_runtime():
    from experiments.other.adaptive_split_control.code.lab import decode

    np.testing.assert_array_equal(decode(np.array([[0, -3.47e-18]]), 10), [0])
    assert native_parity(make_problem("bowtie", "frustrated", 2007), None, 0) < 1e-7
