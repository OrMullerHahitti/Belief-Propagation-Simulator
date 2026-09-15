"""Full-clone, full-domain active-region stability for pairwise Min-sum.

The Q state follows a factor phase, so R=response(Q), and fixed unary R is
included as an affine input. Reference-label differences remove only message
gauge offsets. No clone synchronization or binary-domain reduction is used.

On a strict active-minimizer region the undamped map is G(q)=Jq+b. Its damped
Jacobian is lambda I+(1-lambda)J. For an eigenvalue mu, strict Schur stability
is equivalent to Re(mu)<1 and

    lambda > 1 - 2*(1-Re(mu))/abs(1-mu)**2,  lambda < 1.

The lower endpoint zero is allowed exactly when all abs(mu)<1. Eigenvalue
mu=1 is never strictly stabilized. This is a local fixed-region statement,
not a global convergence theorem or a criterion for a switching orbit.
Unary Q coordinates are passive: their extra eigenvalues equal lambda.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from experiments.other.aaai_derived_control.code.kernel import (
    PairwiseKernel,
    PairwiseProblem,
    gauge,
)


@dataclass(frozen=True)
class ActiveRegion:
    """Exact integer branch derivative plus floating-point selector margins.

    ``selectors[clone, recipient_endpoint, recipient_label]`` is the minimizing
    label of the other endpoint. Each matrix coordinate is ordered by clone,
    sender endpoint, then labels 1..d-1. A zero margin means the returned
    selected-branch derivative is not a two-sided derivative.
    """

    jacobian: sparse.csr_matrix
    selectors: np.ndarray
    margins: np.ndarray
    response_jacobian: sparse.csr_matrix

    @property
    def minimum_margin(self) -> float:
        return float(self.margins.min())

    @property
    def committed_count(self) -> int:
        return int(np.all(self.selectors == self.selectors[..., :1], axis=-1).sum())


def factor_responses(tables: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute mathematical factor responses, returning reference differences."""
    tables = np.asarray(tables, dtype=float)
    q = np.asarray(q, dtype=float)
    if q.ndim != 3:
        raise ValueError("Q must have three dimensions")
    if any(
        (
            q.shape[1] != 2,
            q.shape[2] < 2,
            tables.shape != (len(q), q.shape[2], q.shape[2]),
            not np.isfinite(q).all(),
            not np.isfinite(tables).all(),
        )
    ):
        raise ValueError("finite clone tables (c,d,d) and Q (c,2,d), d>=2 required")
    r = np.empty_like(q)
    r[:, 0] = (tables + q[:, 1, None, :]).min(axis=2)
    r[:, 1] = (tables + q[:, 0, :, None]).min(axis=1)
    return gauge(r)


def undamped_q_map(
    problem: PairwiseProblem,
    weights: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Evaluate the autonomous mathematical Q map after unary initialization."""
    weights = np.asarray(weights)
    if weights.shape != (len(problem.edges),) or np.any(
        ~np.isfinite(weights) | (weights <= 0) | (weights >= 1)
    ):
        raise ValueError("one split weight strictly inside (0,1) per original edge")
    alpha = np.stack((weights, 1 - weights), axis=1).reshape(-1)
    tables = np.repeat(problem.costs, 2, axis=0) * alpha[:, None, None]
    ends = np.repeat(problem.edges, 2, axis=0)
    r = factor_responses(tables, q)
    beliefs = gauge(problem.unary).copy()
    np.add.at(beliefs, ends.ravel(), r.reshape(-1, problem.d))
    return gauge(beliefs[ends] - r)


def active_region(kernel: PairwiseKernel) -> ActiveRegion:
    """Derive the complete Q-map Jacobian at the kernel's current clone state.

    For one response difference R(y)-R(0), differentiating its two active
    minima gives e[k(y)]-e[k(0)]. Label-zero coordinates are omitted. The
    variable map sums all responses except its recipient factor. Costs are
    read with their original tensor-axis ordering; asymmetric clones are
    analyzed separately. Additive raw-message normalization is irrelevant.
    """
    kernel._validate_controls()
    q, tables, d = gauge(kernel.q), kernel.tables, kernel.problem.d
    if d < 2 or not np.isfinite(q).all():
        raise ValueError("finite Q messages and domain size >=2 required")
    count, size = len(q), len(q) * 2 * (d - 1)
    selectors = np.empty((count, 2, d), dtype=int)
    margins = np.empty((count, 2, d))
    rows, cols, values = [], [], []
    for recipient in (0, 1):
        scores = tables + (q[:, 1, None, :] if recipient == 0 else q[:, 0, :, None])
        if recipient == 1:
            scores = scores.transpose(0, 2, 1)
        chosen = scores.argmin(axis=2)
        selectors[:, recipient] = chosen
        sorted_two = np.partition(scores, 1, axis=2)[..., :2]
        margins[:, recipient] = sorted_two[..., 1] - sorted_two[..., 0]
        for clone in range(count):
            for label in range(1, d):
                row = (2 * clone + recipient) * (d - 1) + label - 1
                for picked, sign in ((chosen[clone, label], 1), (chosen[clone, 0], -1)):
                    if picked != 0:
                        rows.append(row)
                        cols.append((2 * clone + 1 - recipient) * (d - 1) + picked - 1)
                        values.append(sign)
    response = sparse.csr_matrix((values, (rows, cols)), shape=(size, size), dtype=int)
    response.eliminate_zeros()
    rows, cols = [], []
    ends = kernel.ends.ravel()
    for incidence, node in enumerate(ends):
        others = np.flatnonzero(ends == node)
        others = others[others != incidence]
        for label in range(d - 1):
            rows.extend([incidence * (d - 1) + label] * len(others))
            cols.extend((others * (d - 1) + label).tolist())
    aggregate = sparse.csr_matrix(
        (np.ones(len(rows), dtype=int), (rows, cols)), shape=(size, size)
    )
    jacobian = (aggregate @ response).tocsr()
    jacobian.eliminate_zeros()
    return ActiveRegion(jacobian, selectors, margins, response)


def relaxation_interval(eigenvalues: np.ndarray) -> dict:
    """Return the strict scalar-relaxation interval for supplied eigenvalues.

    The criterion is exact algebra. When inputs came from numerical eigensolves,
    its output has the same numerical uncertainty; it is not interval arithmetic.
    No tolerance silently turns a unit eigenvalue into a contracting one.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=complex).reshape(-1)
    if not np.isfinite(eigenvalues).all() or not len(eigenvalues):
        raise ValueError("a nonempty finite eigenvalue vector is required")
    possible = bool(np.all(eigenvalues.real < 1))
    if not possible:
        return {"exists": False, "lower": None, "upper": 1.0, "lower_inclusive": False}
    bounds = 1 - 2 * (1 - eigenvalues.real) / np.abs(1 - eigenvalues) ** 2
    lower = max(0.0, float(bounds.max()))
    return {
        "exists": True,
        "lower": lower,
        "upper": 1.0,
        "lower_inclusive": bool(lower == 0 and np.all(np.abs(eigenvalues) < 1)),
    }


def exact_nilpotency_index(matrix: np.ndarray) -> int | None:
    """Check integer matrix powers with Python integers, avoiding overflow.

    Cayley-Hamilton implies that an n by n nilpotent matrix has index <=n.
    This bounded helper is intended for small strongly connected blocks.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("a square matrix is required")
    if not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError("exact powers require integer matrix entries")
    exact = matrix.astype(object)
    power = exact.copy()
    for index in range(1, len(matrix) + 1):
        if not np.any(power):
            return index
        power = power @ exact
    return None


def spectral_summary(
    region: ActiveRegion, damping: float, dense_limit: int = 512
) -> dict:
    """Analyze SCC diagonal blocks, with an exact structural nilpotency check.

    An acyclic dependency graph proves J nilpotent in exact integer arithmetic.
    Small cyclic integer blocks are also checked for exact nilpotency, since
    signed dependency cycles can cancel. Other blocks of size <=dense_limit
    use all numerical eigenvalues. Larger
    blocks remain explicitly unresolved, avoiding a misleading partial spectrum.
    Schur stability at a nonfixed state does not certify its future trajectory.
    """
    if not np.isfinite(damping) or not 0 <= damping < 1 or dense_limit < 1:
        raise ValueError("valid damping and a positive dense block limit required")
    jacobian = region.jacobian
    count, labels = connected_components(jacobian, directed=True, connection="strong")
    sizes = np.bincount(labels, minlength=count)
    diagonal = jacobian.diagonal()
    cyclic = [
        np.flatnonzero(labels == component)
        for component, size in enumerate(sizes)
        if size > 1 or np.any(diagonal[labels == component] != 0)
    ]
    acyclic = not cyclic
    eigenvalues = [0j]
    unresolved = []
    block_indices = np.ones(count, dtype=int)
    nilpotent_blocks = []
    all_nilpotent = True
    for indices in cyclic:
        if len(indices) > dense_limit:
            unresolved.append(len(indices))
            all_nilpotent = False
        else:
            block = jacobian[indices][:, indices].toarray()
            index = exact_nilpotency_index(block) if len(indices) <= 64 else None
            if index is not None:
                block_indices[labels[indices[0]]] = index
                nilpotent_blocks.append({"size": len(indices), "index": index})
                eigenvalues.append(0j)
            else:
                all_nilpotent = False
                eigenvalues.extend(np.linalg.eigvals(block))
    eigenvalues = np.asarray(eigenvalues)
    complete = not unresolved
    transformed = damping + (1 - damping) * eigenvalues
    result = {
        "pairwise_q_dimension": jacobian.shape[0],
        "jacobian_nonzeros": jacobian.nnz,
        "minimum_selector_margin": region.minimum_margin,
        "strict_region": region.minimum_margin > 0,
        "committed_responses": region.committed_count,
        "total_responses": int(np.prod(region.selectors.shape[:2])),
        "dependency_dag": acyclic,
        "structurally_proved_nilpotent": acyclic,
        "proved_nilpotent": all_nilpotent,
        "exact_nilpotent_cyclic_blocks": nilpotent_blocks,
        "cyclic_scc_sizes": sorted([len(indices) for indices in cyclic], reverse=True),
        "unresolved_scc_sizes": unresolved,
        "complete_spectrum": complete,
        "undamped_spectral_radius": (
            float(np.abs(eigenvalues).max()) if complete else None
        ),
        "damped_spectral_radius": (
            float(np.abs(transformed).max()) if complete else None
        ),
        "maximum_real_eigenvalue": float(eigenvalues.real.max()) if complete else None,
        "relaxation_interval": relaxation_interval(eigenvalues) if complete else None,
        "spectral_arithmetic": (
            "exact integer nilpotent SCCs"
            if all_nilpotent
            else "float64 SCC eigensolves"
        ),
    }
    if all_nilpotent:
        # a path of triangular nilpotent blocks has index <=sum of its block indices.
        nonzero = jacobian.tocoo()
        row_blocks, column_blocks = labels[nonzero.row], labels[nonzero.col]
        crossing = row_blocks != column_blocks
        condensation = sparse.csr_matrix(
            (
                np.ones(int(crossing.sum())),
                (row_blocks[crossing], column_blocks[crossing]),
            ),
            shape=(count, count),
        )
        condensation.eliminate_zeros()
        indegree = np.diff(condensation.indptr).copy()
        outgoing = condensation.tocsc()
        ready = list(np.flatnonzero(indegree == 0))
        distance = block_indices.copy()
        for node in ready:
            start, stop = outgoing.indptr[node], outgoing.indptr[node + 1]
            for target in outgoing.indices[start:stop]:
                distance[target] = max(
                    distance[target], distance[node] + block_indices[target]
                )
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(int(target))
        result["nilpotency_index_upper_bound"] = int(distance.max())
    return result


def load_problem(path: Path, family: str, seed: int) -> PairwiseProblem:
    """Read frozen paper inputs without regenerating their randomness or names."""
    with np.load(path) as data:
        kwargs = {key: data[key].copy() for key in ("edges", "costs", "unary")}
        for key in ("variable_names", "factor_names", "unary_names"):
            if key in data:
                kwargs[key] = tuple(data[key])
    return PairwiseProblem(**kwargs, family=family, seed=seed, topology=family)


def nilpotent_basin_certificate(kernel: PairwiseKernel) -> tuple[dict, np.ndarray]:
    """Prove a local basin for a nilpotent branch using exact dyadic arithmetic.

    Stored float tables, unaries and Q coordinates are interpreted as exact
    dyadic rational numbers. If J^p=0, its candidate fixed point is
    q*=(I+...+J^(p-1))b. A strictly positive exact selector gap m makes that
    candidate self-consistent. Put K=sum(k=0..p-1)||J^k||_infinity. For every
    lambda in [0,1), the ball ||q-q*||_infinity < m/(2K) remains in the region
    and converges to q*. This follows from the binomial expansion of
    (lambda I+(1-lambda)J)^t: each scalar binomial coefficient times its
    lambda factors is <=1, and all terms with J^k for k>=p vanish.

    This proves the mathematical recurrence for the stored dyadic problem;
    it does not replace a rounding-error analysis of infinite machine runs.
    Returned fixed coordinates are exact integer numerators. The common
    denominator is recorded as a decimal string in the certificate.
    """
    region = active_region(kernel)
    summary = spectral_summary(region, kernel.damping)
    if not summary["proved_nilpotent"]:
        raise ValueError("the active Jacobian has no verified nilpotency certificate")
    jacobian = region.jacobian
    p = summary["nilpotency_index_upper_bound"]
    norm_j = int(np.asarray(abs(jacobian).sum(axis=1)).max())
    power = sparse.eye(jacobian.shape[0], dtype=np.int64, format="csr")
    norms = []
    for _ in range(p):
        norm = int(np.asarray(abs(power).sum(axis=1)).max())
        norms.append(norm)
        # each result row's absolute sum is bounded before int64 multiplication.
        if norm * norm_j >= 2**62:
            raise OverflowError("integer sparse-power bound requires wider arithmetic")
        power = (power @ jacobian).tocsr()
        power.eliminate_zeros()
        if power.nnz == 0:
            break
    if power.nnz:
        raise AssertionError("the claimed nilpotency bound did not annihilate J")
    p, bound = len(norms), sum(norms)
    arrays = [kernel.tables, kernel.problem.unary, gauge(kernel.q)]
    ratios = [
        [float(value).as_integer_ratio() for value in array.ravel()] for array in arrays
    ]
    denominator = max(den for source in ratios for _, den in source)
    exact = [
        np.array(
            [num * (denominator // den) for num, den in source], dtype=object
        ).reshape(array.shape)
        for source, array in zip(ratios, arrays)
    ]
    tables, unary, endpoint = exact
    clone_indices = np.arange(len(tables))[:, None]
    labels = np.arange(kernel.problem.d)[None, :]
    r_intercept = np.empty_like(endpoint)
    r_intercept[:, 0] = tables[clone_indices, labels, region.selectors[:, 0]]
    r_intercept[:, 1] = tables[clone_indices, region.selectors[:, 1], labels]
    r_intercept -= r_intercept[..., :1]
    belief = unary - unary[:, :1]
    np.add.at(belief, kernel.ends.ravel(), r_intercept.reshape(-1, kernel.problem.d))
    intercept = belief[kernel.ends] - r_intercept
    b = intercept[..., 1:].ravel()

    def apply_exact(vector: np.ndarray) -> np.ndarray:
        result = np.empty(len(vector), dtype=object)
        for row in range(len(vector)):
            start, stop = jacobian.indptr[row], jacobian.indptr[row + 1]
            result[row] = sum(
                int(jacobian.data[index]) * vector[jacobian.indices[index]]
                for index in range(start, stop)
            )
        return result

    fixed, term = b.copy(), b.copy()
    for _ in range(1, p):
        term = apply_exact(term)
        fixed += term
    affine_residual = int(max(abs(apply_exact(fixed) + b - fixed)))
    if affine_residual:
        raise AssertionError("exact reconstructed affine fixed point is inconsistent")
    q_star = np.zeros_like(endpoint)
    q_star[..., 1:] = fixed.reshape(q_star.shape[0], 2, q_star.shape[2] - 1)
    minimum_gap = None
    for recipient in (0, 1):
        scores = tables + (
            q_star[:, 1, None, :] if recipient == 0 else q_star[:, 0, :, None]
        )
        if recipient == 1:
            scores = scores.transpose(0, 2, 1)
        selected = scores[clone_indices, labels, region.selectors[:, recipient]]
        gaps = scores - selected[..., None]
        other_labels = np.arange(kernel.problem.d)[None, None, :]
        competitors = other_labels != region.selectors[:, recipient, :, None]
        gap = int(min(gaps[competitors]))
        minimum_gap = gap if minimum_gap is None else min(minimum_gap, gap)
    distance = int(max(abs(endpoint - q_star).ravel()))
    strict = minimum_gap > 0
    inside = bool(strict and 2 * bound * distance < minimum_gap)
    certificate = {
        "arithmetic": "exact Python integer numerators over a common dyadic denominator",
        "denominator": str(denominator),
        "nilpotency_index": p,
        "integer_power_infinity_norms": norms,
        "power_norm_sum_K": bound,
        "fixed_point_affine_residual_numerator": affine_residual,
        "minimum_selector_gap_numerator": str(minimum_gap),
        "minimum_selector_gap": minimum_gap / denominator,
        "fixed_point_strictly_matches_active_region": strict,
        "endpoint_distance_numerator": str(distance),
        "endpoint_distance": distance / denominator,
        "certified_open_ball_radius": (
            minimum_gap / (2 * bound * denominator) if strict else None
        ),
        "endpoint_inside_certified_ball": inside,
        "valid_old_q_damping": "all lambda in [0,1), fixed after this state",
        "scope": (
            "pairwise Q mathematical recurrence; passive unary Q also converges; "
            "infinite floating-point rounding is not modeled"
        ),
    }
    return certificate, q_star


def certify_saved(out: Path) -> None:
    """Reconstruct and prove local basins for saved nilpotent final states."""
    records_path = out / "records.json"
    records = json.loads(records_path.read_text())
    last_step = max(record["step"] for record in records)
    root = Path(__file__).resolve().parents[4]
    source = root / "results/aaai_derived_control_20260915"
    certificates = []
    folder = out / "basin_certificates"
    folder.mkdir(exist_ok=False)
    hashes = {str(records_path): hashlib.sha256(records_path.read_bytes()).hexdigest()}
    for record in records:
        if record["step"] != last_step or not record["proved_nilpotent"]:
            continue
        family, seed, method = record["family"], record["seed"], record["method"]
        part = "paper_confirmation" if family == "random_sparse" else "paper_validation"
        input_path = source / part / "inputs" / f"{family}_{seed}.npz"
        problem = load_problem(input_path, family, seed)
        key = f"{family}_{seed}_{method}_{last_step}"
        state_path = out / f"{key}.npz"
        state = np.load(state_path)
        kernel = PairwiseKernel(problem, weights=state["weights"], damping=0.9)
        kernel.q = state["q"]
        certificate, numerator = nilpotent_basin_certificate(kernel)
        certificate.update(family=family, seed=seed, method=method, step=last_step)
        certificates.append(certificate)
        np.savez_compressed(
            folder / f"{key}_fixed_point.npz",
            numerator=numerator.astype(str),
            denominator=certificate["denominator"],
        )
        for path in (input_path, state_path):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(json.dumps(certificate), flush=True)
    (folder / "certificates.json").write_text(json.dumps(certificates, indent=2) + "\n")
    provenance = {
        "input_sha256": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {"out": str(out), "certify_saved": True},
    }
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def release_saved(out: Path) -> None:
    """Replay exact saved endpoints, then remove damping in the actual kernel."""
    certificate_path = out / "basin_certificates/certificates.json"
    certificates = json.loads(certificate_path.read_text())
    root = Path(__file__).resolve().parents[4]
    source = root / "results/aaai_derived_control_20260915"
    records = []
    for certificate in certificates:
        family, seed = certificate["family"], certificate["seed"]
        method, steps = certificate["method"], certificate["step"]
        part = "paper_confirmation" if family == "random_sparse" else "paper_validation"
        input_path = source / part / "inputs" / f"{family}_{seed}.npz"
        kernel = PairwiseKernel(load_problem(input_path, family, seed), damping=0.9)
        for step in range(steps):
            if method.startswith("split095") and step in (64, 256):
                kernel.weights[:] = 0.95 if step == 64 else 0.5
            kernel.step()
        state = np.load(out / f"{family}_{seed}_{method}_{steps}.npz")
        np.testing.assert_array_equal(gauge(kernel.q), state["q"])
        initial_assignment, initial_cost = kernel.assignment.copy(), kernel.cost
        kernel.damping = 0
        snapshots = []
        changed = 0
        max_cost_delta = 0.0
        for step in range(1, 17):
            kernel.step()
            changed += int(np.any(kernel.assignment != initial_assignment))
            max_cost_delta = max(max_cost_delta, abs(kernel.cost - initial_cost))
            if step in (4, 16):
                proposed = undamped_q_map(
                    kernel.problem, kernel.weights, gauge(kernel.q)
                )
                snapshots.append(
                    {
                        "updates_after_damping_removal": step,
                        "original_cost": kernel.cost,
                        "all_message_step_residual": kernel.message_residual,
                        "undamped_pairwise_q_defect": float(
                            np.abs(proposed - gauge(kernel.q)).max()
                        ),
                    }
                )
        record = {
            "family": family,
            "seed": seed,
            "method": method,
            "release_update": steps,
            "original_cost_before_release": initial_cost,
            "initial_assignment": initial_assignment.tolist(),
            "changed_assignment_updates_in_next_16": changed,
            "maximum_original_cost_change_in_next_16": max_cost_delta,
            "checkpoints": snapshots,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (out / "damping_release.json").write_text(json.dumps(records, indent=2) + "\n")
    provenance = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "certificate_sha256": hashlib.sha256(certificate_path.read_bytes()).hexdigest(),
        "arguments": {"out": str(out), "release_saved": True},
        "scope": "native-equivalent PairwiseKernel with its raw message arithmetic and normalization schedule",
    }
    (out / "damping_release_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def release_controls_saved(out: Path) -> None:
    """Isolate raw-offset rounding with per-step normalization and gauge controls."""
    certificate_path = out / "basin_certificates/certificates.json"
    certificates = json.loads(certificate_path.read_text())
    root = Path(__file__).resolve().parents[4]
    source = root / "results/aaai_derived_control_20260915"
    folder = out / "release_controls"
    folder.mkdir(exist_ok=False)
    records, summaries = [], []
    for certificate in certificates:
        family, seed = certificate["family"], certificate["seed"]
        method, steps = certificate["method"], certificate["step"]
        part = "paper_confirmation" if family == "random_sparse" else "paper_validation"
        input_path = source / part / "inputs" / f"{family}_{seed}.npz"
        initial = PairwiseKernel(load_problem(input_path, family, seed), damping=0.9)
        for step in range(steps):
            if method.startswith("split095") and step in (64, 256):
                initial.weights[:] = 0.95 if step == 64 else 0.5
            initial.step()
        key = f"{family}_{seed}_{method}_{steps}"
        state = np.load(out / f"{key}.npz")
        np.testing.assert_array_equal(gauge(initial.q), state["q"])
        for variant in ("native_raw", "normalize_every_step", "direct_gauge_map"):
            kernel = initial.clone()
            kernel.damping = 0
            point = gauge(kernel.q)
            trace = {name: [] for name in ("q", "r", "unary_q", "unary_r")}
            changed, max_cost_delta = 0, 0.0
            rows = []
            for step in range(17):
                if step:
                    if variant == "direct_gauge_map":
                        point = undamped_q_map(kernel.problem, kernel.weights, point)
                    else:
                        if variant == "normalize_every_step":
                            for name in trace:
                                message = getattr(kernel, name)
                                message -= message.min(axis=-1, keepdims=True)
                        kernel.step()
                        point = gauge(kernel.q)
                mathematical_r = factor_responses(kernel.tables, point)
                if variant == "direct_gauge_map":
                    belief = gauge(kernel.problem.unary)
                    np.add.at(
                        belief,
                        kernel.ends.ravel(),
                        mathematical_r.reshape(-1, kernel.problem.d),
                    )
                    assignment = belief.argmin(axis=1)
                    cost = kernel.problem.cost(assignment)
                    max_offsets = {
                        "q": float(np.abs(point).max()),
                        "r": float(np.abs(mathematical_r).max()),
                        "unary_q": None,
                        "unary_r": None,
                    }
                    r_error, unary_r_error = 0.0, 0.0
                else:
                    assignment, cost = kernel.assignment, kernel.cost
                    max_offsets = {
                        name: float(np.abs(getattr(kernel, name)).max())
                        for name in trace
                    }
                    r_error = float(np.abs(gauge(kernel.r) - mathematical_r).max())
                    expected_unary = gauge(kernel.problem.unary[:, None, :] * 0.5)
                    unary_r_error = float(
                        np.abs(gauge(kernel.unary_r) - expected_unary).max()
                    )
                    for name in trace:
                        trace[name].append(getattr(kernel, name).copy())
                changed += int(np.any(assignment != initial.assignment))
                max_cost_delta = max(max_cost_delta, abs(cost - initial.cost))
                proposed = undamped_q_map(kernel.problem, kernel.weights, point)
                row = {
                    "family": family,
                    "seed": seed,
                    "method": method,
                    "variant": variant,
                    "release_step": step,
                    "cost": cost,
                    "assignment_changed": bool(
                        np.any(assignment != initial.assignment)
                    ),
                    "assignment": assignment.tolist(),
                    "maximum_absolute_raw_messages": max_offsets,
                    "maximum_factor_response_arithmetic_error": r_error,
                    "maximum_unary_response_arithmetic_error": unary_r_error,
                    "undamped_q_defect": float(np.abs(proposed - point).max()),
                }
                rows.append(row)
            records.extend(rows)
            if variant == "native_raw" and changed:
                np.savez_compressed(
                    folder / f"{key}_{variant}_full_trace.npz",
                    **{name: np.array(values) for name, values in trace.items()},
                    assignments=np.array([row["assignment"] for row in rows]),
                    costs=np.array([row["cost"] for row in rows]),
                )
            summary = {
                "family": family,
                "seed": seed,
                "method": method,
                "variant": variant,
                "changed_assignments_in_16_steps": changed,
                "maximum_cost_change": max_cost_delta,
                "maximum_raw_q_magnitude": max(
                    row["maximum_absolute_raw_messages"]["q"] for row in rows
                ),
                "maximum_factor_response_arithmetic_error": max(
                    row["maximum_factor_response_arithmetic_error"] for row in rows
                ),
                "final_undamped_q_defect": rows[-1]["undamped_q_defect"],
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (folder / "steps.json").write_text(json.dumps(records, indent=2) + "\n")
    (folder / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    provenance = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "certificate_sha256": hashlib.sha256(certificate_path.read_bytes()).hexdigest(),
        "arguments": {"out": str(out), "release_controls": True},
        "scope": (
            "native raw arithmetic versus the same kernel normalized "
            "before every update versus direct gauge recurrence"
        ),
    }
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    """Replay fixed representative inputs and save exact active-dependency evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--certify-saved", action="store_true")
    parser.add_argument("--release-saved", action="store_true")
    parser.add_argument("--release-controls", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--sparse-seeds", type=int, nargs="+", default=[6000, 6002, 6003]
    )
    parser.add_argument("--dense-seeds", type=int, nargs="+", default=[5100])
    args = parser.parse_args()
    if sum((args.certify_saved, args.release_saved, args.release_controls)) > 1:
        parser.error("choose one saved-state operation at a time")
    if args.certify_saved:
        certify_saved(args.out)
        return
    if args.release_saved:
        release_saved(args.out)
        return
    if args.release_controls:
        release_controls_saved(args.out)
        return
    if args.steps < 1:
        parser.error("steps must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[4]
    original = root / "results/aaai_derived_control_20260915"
    cases = [
        ("random_sparse", "paper_confirmation", seed) for seed in args.sparse_seeds
    ]
    cases += [("random_dense", "paper_validation", seed) for seed in args.dense_seeds]
    records, hashes = [], {}
    for family, part, seed in cases:
        input_path = original / part / "inputs" / f"{family}_{seed}.npz"
        hashes[str(input_path.relative_to(root))] = hashlib.sha256(
            input_path.read_bytes()
        ).hexdigest()
        problem = load_problem(input_path, family, seed)
        for method in ("split05_d09", "split095_pulse64_256_d09"):
            kernel = PairwiseKernel(problem, damping=0.9)
            evidence_path = (
                original / part / "trajectories" / f"{family}_{seed}_{method}.csv"
            )
            evidence = None
            if evidence_path.exists():
                evidence = np.loadtxt(evidence_path, delimiter=",", skiprows=1)
                hashes[str(evidence_path.relative_to(root))] = hashlib.sha256(
                    evidence_path.read_bytes()
                ).hexdigest()
            parity_cost_error, parity_assignment_mismatches = 0.0, 0
            for step in range(args.steps):
                if method.startswith("split095") and step in (64, 256):
                    kernel.weights[:] = 0.95 if step == 64 else 0.5
                kernel.step()
                if evidence is not None and step < len(evidence):
                    parity_cost_error = max(
                        parity_cost_error, abs(kernel.cost - evidence[step, 1])
                    )
                    parity_assignment_mismatches += int(
                        np.sum(kernel.assignment != evidence[step, 2:])
                    )
                if step + 1 in (64, 256, 512, args.steps):
                    region = active_region(kernel)
                    summary = spectral_summary(region, kernel.damping)
                    proposed = undamped_q_map(problem, kernel.weights, gauge(kernel.q))
                    record = {
                        "family": family,
                        "seed": seed,
                        "method": method,
                        "step": step + 1,
                        "cost": kernel.cost,
                        "undamped_q_defect": float(
                            np.abs(proposed - gauge(kernel.q)).max()
                        ),
                        **summary,
                    }
                    key = f"{family}_{seed}_{method}_{step + 1}"
                    np.savez_compressed(
                        args.out / f"{key}.npz",
                        q=gauge(kernel.q),
                        selectors=region.selectors,
                        margins=region.margins,
                        weights=kernel.weights,
                    )
                    sparse.save_npz(args.out / f"{key}_J.npz", region.jacobian)
                    records.append(record)
                    print(json.dumps(record), flush=True)
            records[-1]["saved_trajectory_max_cost_error"] = (
                parity_cost_error if evidence is not None else None
            )
            records[-1]["saved_trajectory_assignment_mismatches"] = (
                parity_assignment_mismatches if evidence is not None else None
            )
            if evidence is not None and (
                parity_cost_error > 1e-8 or parity_assignment_mismatches
            ):
                raise AssertionError(
                    "replayed trajectory differs from saved paper experiment"
                )
    (args.out / "records.json").write_text(json.dumps(records, indent=2) + "\n")
    keys = [
        "family",
        "seed",
        "method",
        "step",
        "cost",
        "undamped_q_defect",
        "minimum_selector_margin",
        "committed_responses",
        "total_responses",
        "jacobian_nonzeros",
        "structurally_proved_nilpotent",
        "proved_nilpotent",
        "complete_spectrum",
        "undamped_spectral_radius",
        "damped_spectral_radius",
        "maximum_real_eigenvalue",
    ]
    with (args.out / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    provenance = json.dumps(
        {
            "input_sha256": hashes,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "full-clone full-domain gauge Q state; unary Q adds only damping eigenvalues",
            "interpretation": "frozen active-region derivative only; small damped steps do not prove convergence",
        },
        indent=2,
    )
    (args.out / "provenance.json").write_text(provenance + "\n")


if __name__ == "__main__":
    main()
