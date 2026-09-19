"""Full-coordinate derivative and scalar-relaxation checks."""

import numpy as np
import pytest
from scipy import sparse

from experiments.aamas.aaai_derived_control.code.kernel import (
    PairwiseKernel,
    gauge,
    make_problem,
)
from experiments.aamas.damping_causality.code.native_experiments import fixtures
from experiments.aamas.damping_generalization.code.local_stability import (
    ActiveRegion,
    active_region,
    exact_nilpotency_index,
    factor_responses,
    nilpotent_basin_certificate,
    relaxation_interval,
    spectral_summary,
    undamped_q_map,
)


@pytest.mark.parametrize("domain", [2, 3, 10])
def test_full_clone_jacobian_matches_independent_coordinate_perturbations(domain):
    problem = make_problem("triangle", 37, d=domain)
    kernel = PairwiseKernel(problem, weights=0.31)
    rng = np.random.default_rng(481)
    kernel.q[:] = rng.normal(0, 17, kernel.q.shape)
    region = active_region(kernel)
    assert region.minimum_margin > 1e-7
    q = gauge(kernel.q)
    epsilon = min(1e-5, region.minimum_margin / 20)
    directions = np.eye(region.jacobian.shape[0])
    numerical = []
    for direction in directions:
        perturbation = np.zeros_like(q)
        perturbation[..., 1:] = direction.reshape(q.shape[0], 2, domain - 1)
        plus = undamped_q_map(problem, kernel.weights, q + epsilon * perturbation)
        minus = undamped_q_map(problem, kernel.weights, q - epsilon * perturbation)
        numerical.append(((plus - minus)[..., 1:] / (2 * epsilon)).ravel())
    np.testing.assert_allclose(
        region.jacobian.toarray(), np.array(numerical).T, atol=1e-7
    )


def test_autonomous_map_matches_kernel_one_step_for_asymmetric_clones():
    kernel = PairwiseKernel(make_problem("k4", 6, d=10), weights=0.79, damping=0)
    kernel.advance(12)
    q = gauge(kernel.q)
    proposed = undamped_q_map(kernel.problem, kernel.weights, q)
    kernel.step()
    np.testing.assert_allclose(gauge(kernel.q), proposed, atol=1e-10)


@pytest.mark.parametrize(
    "eigenvalues,possible,lower,inclusive",
    [
        ([0, 0], True, 0, True),
        ([-1, 0], True, 0, False),
        ([-2], True, 1 / 3, False),
        ([0.5 + 2j, 0.5 - 2j], True, 1 - 1 / 4.25, False),
        ([1], False, None, False),
        ([1 + 1j], False, None, False),
        ([1.1, -4], False, None, False),
    ],
)
def test_scalar_relaxation_interval(eigenvalues, possible, lower, inclusive):
    interval = relaxation_interval(eigenvalues)
    assert interval["exists"] == possible
    assert interval["lower_inclusive"] == inclusive
    if possible:
        assert interval["lower"] == pytest.approx(lower)
        coefficient = (lower + 1) / 2
        transformed = coefficient + (1 - coefficient) * np.asarray(eigenvalues)
        assert np.abs(transformed).max() < 1
        if lower > 0:
            coefficient = lower / 2
            transformed = coefficient + (1 - coefficient) * np.asarray(eigenvalues)
            assert np.abs(transformed).max() >= 1
    else:
        assert interval["lower"] is None


def test_strict_successful_path_region_is_zero_full_clone_jacobian():
    kernel = PairwiseKernel(fixtures()["robust_path"], damping=0.5).advance(100)
    region = active_region(kernel)
    assert region.minimum_margin > 0
    assert region.jacobian.nnz == 0
    summary = spectral_summary(region, 0.5)
    assert summary["structurally_proved_nilpotent"]
    assert summary["nilpotency_index_upper_bound"] == 1
    assert summary["damped_spectral_radius"] == 0.5
    assert summary["relaxation_interval"]["lower_inclusive"]


def test_selector_boundary_reports_zero_margin():
    kernel = PairwiseKernel(fixtures()["robust_path"])
    kernel.q[:, :, 1] = 8
    region = active_region(kernel)
    assert region.minimum_margin == 0
    assert not spectral_summary(region, 0.9)["strict_region"]


def _synthetic_region(matrix):
    return ActiveRegion(
        sparse.csr_matrix(matrix),
        np.zeros((1, 2, 2), dtype=int),
        np.ones((1, 2, 2)),
        sparse.csr_matrix(matrix),
    )


def test_dependency_dag_proves_finite_nilpotency_despite_gain_above_one():
    region = _synthetic_region([[0, 0, 0], [4, 0, 0], [0, -7, 0]])
    summary = spectral_summary(region, 0.9)
    assert summary["nilpotency_index_upper_bound"] == 3
    assert summary["undamped_spectral_radius"] == 0
    assert summary["damped_spectral_radius"] == 0.9


def test_signed_dependency_cycles_can_cancel_exactly():
    matrix = np.array([[1, -1], [1, -1]])
    assert exact_nilpotency_index(matrix) == 2
    summary = spectral_summary(_synthetic_region(matrix), 0.9)
    assert not summary["dependency_dag"]
    assert summary["proved_nilpotent"]
    assert summary["nilpotency_index_upper_bound"] == 2
    assert summary["undamped_spectral_radius"] == 0
    assert summary["relaxation_interval"]["lower_inclusive"]


def test_exact_integer_powers_do_not_overflow():
    matrix = np.array([[2**40, 2**40], [-(2**40), -(2**40)]], dtype=np.int64)
    assert exact_nilpotency_index(matrix) == 2
    with pytest.raises(ValueError):
        exact_nilpotency_index(matrix.astype(float))


def test_cyclic_blocks_are_fully_analyzed_or_explicitly_unresolved():
    region = _synthetic_region([[0, -1], [-1, 0]])
    summary = spectral_summary(region, 0.9)
    assert summary["cyclic_scc_sizes"] == [2]
    assert summary["complete_spectrum"]
    assert summary["damped_spectral_radius"] == pytest.approx(1)
    assert not summary["relaxation_interval"]["exists"]
    limited = spectral_summary(region, 0.9, dense_limit=1)
    assert not limited["complete_spectrum"]
    assert limited["unresolved_scc_sizes"] == [2]
    assert limited["damped_spectral_radius"] is None


def test_invalid_domain_and_nonfinite_values_rejected():
    with pytest.raises(ValueError):
        factor_responses(np.zeros((1, 1, 1)), np.zeros((1, 2, 1)))
    with pytest.raises(ValueError):
        relaxation_interval([float("nan")])


def test_exact_dyadic_basin_certifies_successful_path_and_all_damping_values():
    kernel = PairwiseKernel(fixtures()["robust_path"], damping=0.5).advance(100)
    certificate, numerators = nilpotent_basin_certificate(kernel)
    assert certificate["fixed_point_affine_residual_numerator"] == 0
    assert certificate["fixed_point_strictly_matches_active_region"]
    assert certificate["endpoint_inside_certified_ball"]
    denominator = int(certificate["denominator"])
    fixed = np.asarray(numerators, dtype=float) / denominator
    for damping in (0, 0.01, 0.5, 0.9, 0.99):
        perturbation = np.full_like(
            fixed, certificate["certified_open_ball_radius"] / 4
        )
        perturbation[..., 0] = 0
        point = fixed + perturbation
        for _ in range(10):
            point = damping * point + (1 - damping) * undamped_q_map(
                kernel.problem, kernel.weights, point
            )
        expected = fixed + damping**10 * perturbation
        np.testing.assert_allclose(point, expected, atol=1e-12)


def test_nilpotent_branch_does_not_certify_inconsistent_candidate():
    kernel = PairwiseKernel(fixtures()["robust_path"], damping=0).advance(6)
    assert active_region(kernel).jacobian.nnz == 0
    certificate, _ = nilpotent_basin_certificate(kernel)
    assert not certificate["fixed_point_strictly_matches_active_region"]
    assert not certificate["endpoint_inside_certified_ball"]
    assert int(certificate["minimum_selector_gap_numerator"]) < 0
