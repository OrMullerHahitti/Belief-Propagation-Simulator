"""Check the decoded-label gauge on the six exact paper fixed-point certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from experiments.other.aaai_derived_control.code.kernel import PairwiseKernel
from .local_stability import load_problem


def coordinate_transforms(references: np.ndarray, domain: int) -> tuple:
    """Return exact maps between label-zero and per-message reference gauges.

    Each message's new coordinates use increasing label order excluding its
    selected reference. Returned matrices T,S satisfy new=T old, old=S new.
    Only a linear gauge change is performed; no message coordinates are lost.
    """
    references = np.asarray(references)
    if references.ndim != 1 or not np.issubdtype(references.dtype, np.integer):
        raise ValueError("references must be an integer vector")
    if domain < 2 or np.any((references < 0) | (references >= domain)):
        raise ValueError("valid reference labels and domain >=2 required")
    rows_t, cols_t, vals_t, rows_s, cols_s, vals_s = [], [], [], [], [], []
    for incidence, reference in enumerate(references):
        labels = [label for label in range(domain) if label != reference]
        local = {label: index for index, label in enumerate(labels)}
        base = incidence * (domain - 1)
        for index, label in enumerate(labels):
            for old_label, sign in ((label, 1), (reference, -1)):
                if old_label != 0:
                    rows_t.append(base + index)
                    cols_t.append(base + old_label - 1)
                    vals_t.append(sign)
        for label in range(1, domain):
            for new_label, sign in ((label, 1), (0, -1)):
                if new_label != reference:
                    rows_s.append(base + label - 1)
                    cols_s.append(base + local[new_label])
                    vals_s.append(sign)
    size = len(references) * (domain - 1)
    transform = sparse.csr_matrix(
        (vals_t, (rows_t, cols_t)), shape=(size, size), dtype=int
    )
    inverse = sparse.csr_matrix(
        (vals_s, (rows_s, cols_s)), shape=(size, size), dtype=int
    )
    transform.eliminate_zeros()
    inverse.eliminate_zeros()
    difference = transform @ inverse - sparse.eye(size, dtype=int, format="csr")
    difference.eliminate_zeros()
    if difference.nnz:
        raise AssertionError("the reference-coordinate transforms are not inverses")
    return transform, inverse


def dag_nilpotency_index(matrix: sparse.csr_matrix) -> int | None:
    """Return longest dependency path plus one, or None when a cycle exists."""
    indegree = np.diff(matrix.indptr).copy()
    outgoing = matrix.tocsc()
    ready = list(np.flatnonzero(indegree == 0))
    depth = np.ones(matrix.shape[0], dtype=int)
    for source in ready:
        start, stop = outgoing.indptr[source], outgoing.indptr[source + 1]
        for target in outgoing.indices[start:stop]:
            depth[target] = max(depth[target], depth[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(int(target))
    return int(depth.max()) if len(ready) == matrix.shape[0] else None


def check_certificate(out: Path, certificate: dict) -> dict:
    """Compute exact fixed beliefs and integer similarity-transform the Jacobian."""
    root = Path(__file__).resolve().parents[4]
    family, seed = certificate["family"], certificate["seed"]
    method, step = certificate["method"], certificate["step"]
    part = "paper_confirmation" if family == "random_sparse" else "paper_validation"
    source_inputs = root / "results/aaai_derived_control_20260915"
    input_path = source_inputs.joinpath(part, "inputs", f"{family}_{seed}.npz")
    problem = load_problem(input_path, family, seed)
    key = f"{family}_{seed}_{method}_{step}"
    state = np.load(out / f"{key}.npz")
    saved = np.load(out / "basin_certificates" / f"{key}_fixed_point.npz")
    denominator = int(str(saved["denominator"]))
    numerator = saved["numerator"]
    fixed = np.array([int(value) for value in numerator.ravel()], dtype=object).reshape(
        numerator.shape
    )
    kernel = PairwiseKernel(problem, weights=state["weights"], damping=0.9)

    def exact(array: np.ndarray) -> np.ndarray:
        ratios = [float(value).as_integer_ratio() for value in array.ravel()]
        if any(denominator % den for _, den in ratios):
            raise AssertionError("stored exact denominator does not cover the input")
        return np.array(
            [num * (denominator // den) for num, den in ratios], dtype=object
        ).reshape(array.shape)

    tables, unary = exact(kernel.tables), exact(problem.unary)
    responses = np.empty_like(fixed)
    selectors = np.empty(fixed.shape, dtype=int)
    for recipient in (0, 1):
        scores = tables + (
            fixed[:, 1, None, :] if recipient == 0 else fixed[:, 0, :, None]
        )
        if recipient == 1:
            scores = scores.transpose(0, 2, 1)
        selectors[:, recipient] = scores.argmin(axis=2)
        responses[:, recipient] = scores.min(axis=2)
    responses -= responses[..., :1]
    np.testing.assert_array_equal(selectors, state["selectors"])
    beliefs = unary - unary[:, :1]
    np.add.at(beliefs, kernel.ends.ravel(), responses.reshape(-1, problem.d))
    fixed_residual = int(max(abs(fixed + responses - beliefs[kernel.ends]).ravel()))
    if fixed_residual:
        raise AssertionError("the saved point does not satisfy the exact nonlinear map")
    assignments = beliefs.argmin(axis=1)
    ordered = np.sort(beliefs, axis=1)
    decoding_gap = int(min(ordered[:, 1] - ordered[:, 0]))
    if decoding_gap <= 0:
        raise AssertionError("the fixed point has a nonunique decoded label")
    reference_alignment = []
    for clone, (first, second) in enumerate(kernel.ends):
        reference_alignment.extend(
            (
                selectors[clone, 0, assignments[first]] == assignments[second],
                selectors[clone, 1, assignments[second]] == assignments[first],
            )
        )
    if not all(reference_alignment):
        raise AssertionError(
            "a decoded reference does not select its neighbor's decoded label"
        )
    transform, inverse = coordinate_transforms(
        assignments[kernel.ends.ravel()], problem.d
    )
    original = sparse.load_npz(out / f"{key}_J.npz").tocsr()
    adapted = (transform @ original @ inverse).tocsr()
    adapted.eliminate_zeros()
    if np.any(adapted.data < 0):
        raise AssertionError("the decoded-reference Jacobian has negative entries")
    components, labels = connected_components(
        original, directed=True, connection="strong"
    )
    old_sizes = np.bincount(labels, minlength=components)
    index = dag_nilpotency_index(adapted)
    if index is None:
        raise AssertionError(
            "the nilpotent nonnegative Jacobian has a dependency cycle"
        )
    sparse.save_npz(out / "decoded_gauge" / f"{key}_J_decoded.npz", adapted)
    return {
        "family": family,
        "seed": seed,
        "method": method,
        "step": step,
        "dimension": original.shape[0],
        "exact_nonlinear_fixed_residual_numerator": fixed_residual,
        "denominator": str(denominator),
        "minimum_decoding_gap_numerator": str(decoding_gap),
        "minimum_decoding_gap": decoding_gap / denominator,
        "decoded_assignment": assignments.tolist(),
        "decoded_original_cost": problem.cost(assignments),
        "every_reference_response_selects_neighbor_decoded_label": True,
        "integer_similarity_inverse_verified": True,
        "original_negative_entries": int(np.sum(original.data < 0)),
        "original_cyclic_scc_sizes": sorted(
            old_sizes[old_sizes > 1].tolist(), reverse=True
        ),
        "adapted_negative_entries": int(np.sum(adapted.data < 0)),
        "adapted_maximum_entry": int(adapted.data.max(initial=0)),
        "adapted_nonzeros": adapted.nnz,
        "adapted_dependency_dag": True,
        "dag_nilpotency_index": index,
        "arithmetic": "exact integer gauge transform and exact dyadic fixed beliefs",
    }


def main() -> None:
    """Validate all six certificates and preserve their transformed matrices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folder = args.out / "decoded_gauge"
    folder.mkdir(exist_ok=False)
    certificate_path = args.out / "basin_certificates/certificates.json"
    records = []
    for certificate in json.loads(certificate_path.read_text()):
        record = check_certificate(args.out, certificate)
        records.append(record)
        print(json.dumps(record), flush=True)
    (folder / "records.json").write_text(json.dumps(records, indent=2) + "\n")
    provenance = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "certificate_sha256": hashlib.sha256(certificate_path.read_bytes()).hexdigest(),
        "arguments": {"out": str(args.out)},
    }
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
