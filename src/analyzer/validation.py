"""
Invariant checks and validation for belief propagation analysis.

Validates normalization, projector properties, matrix structures,
cycle alternation, and other mathematical invariants to catch bugs early.
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Any, Optional
from .snapshot import Snapshot
from .config import AnalysisConfig


def check_normalization_invariants(
    snap: Snapshot, config: Optional[AnalysisConfig] = None
) -> List[str]:
    """
    Check that Q messages are properly min-normalized.

    For each (u,f), min(Q[(u,f)]) should be 0 within tolerance.

    Args:
        snap: Snapshot containing Q messages
        config: Analysis configuration

    Returns:
        List of normalization violations
    """
    if config is None:
        config = AnalysisConfig()

    violations = []

    for (var_name, factor_name), q_msg in snap.Q.items():
        if len(q_msg) == 0:
            continue

        min_value = np.min(q_msg)
        if abs(min_value) > config.abs_tol:
            violations.append(
                f"Q[({var_name},{factor_name})] not min-normalized: min={min_value:.6e} > {config.abs_tol}"
            )

    return violations


def check_projector_invariants(
    P: csr_matrix,
    idxQ: Dict[Tuple[str, str, int], int],
    config: Optional[AnalysisConfig] = None,
) -> List[str]:
    """
    Check P projector properties.

    P should be idempotent (P^2 ≈ P) and each block should have rows summing to 0.

    Args:
        P: Projector matrix
        idxQ: Q slot indices for interpreting P structure
        config: Analysis configuration

    Returns:
        List of projector violations
    """
    if config is None:
        config = AnalysisConfig()

    violations = []

    if P.shape[0] == 0 or P.shape[1] == 0:
        return violations

    # Test idempotent property P^2 ≈ P on sample vectors
    max_test_size = min(P.shape[1], 50)  # Limit test size for performance

    for _ in range(min(5, max_test_size)):  # Test multiple random vectors
        test_vec = np.random.randn(P.shape[1])
        Px = P @ test_vec
        PPx = P @ Px

        diff = np.linalg.norm(PPx - Px)
        if diff > config.abs_tol * 10:  # More lenient for numerical errors
            violations.append(f"Projector P not idempotent: ||P^2x - Px|| = {diff:.6e}")
            break  # One failure is enough

    # Check row sum property for each block (should sum to 0)
    if config.validate_projector:
        # Group indices by edge (u,f)
        edge_slots = {}
        for var_name, factor_name, label_idx in idxQ:
            edge = (var_name, factor_name)
            if edge not in edge_slots:
                edge_slots[edge] = []
            edge_slots[edge].append(idxQ[(var_name, factor_name, label_idx)])

        for edge, slot_indices in edge_slots.items():
            if len(slot_indices) <= 1:
                continue  # Need at least 2 slots for meaningful check

            slot_indices.sort()

            # Check that rows in this block sum to 0
            for slot_idx in slot_indices:
                if slot_idx < P.shape[0]:
                    row_sum = P.getrow(slot_idx).sum()
                    if abs(row_sum) > config.abs_tol:
                        violations.append(
                            f"Projector P row {slot_idx} (edge {edge}) sum = {row_sum:.6e} ≠ 0"
                        )

    return violations


def check_matrix_A_structure(
    A: csr_matrix,
    snap: Snapshot,
    idxQ: Dict[Tuple[str, str, int], int],
    idxR: Dict[Tuple[str, str, int], int],
) -> List[str]:
    """
    Check matrix A structure invariants.

    A should only connect same labels from (g→u,a) to (u→f,a) with g≠f.

    Args:
        A: Variable aggregation matrix
        snap: Snapshot
        idxQ, idxR: Slot indices

    Returns:
        List of structure violations
    """
    violations = []

    if A.nnz == 0:
        return violations

    # Create reverse mappings
    reverse_idxQ = {idx: slot_key for slot_key, idx in idxQ.items()}
    reverse_idxR = {idx: slot_key for slot_key, idx in idxR.items()}

    # Check each nonzero entry
    A_coo = A.tocoo()
    for row, col, value in zip(A_coo.row, A_coo.col, A_coo.data):
        # Get slot keys
        if row not in reverse_idxQ or col not in reverse_idxR:
            violations.append(f"Matrix A has entry at invalid indices ({row},{col})")
            continue

        q_slot = reverse_idxQ[row]  # (u, f, a)
        r_slot = reverse_idxR[col]  # (g, u_r, a_r)

        if len(q_slot) != 3 or len(r_slot) != 3:
            continue

        u, f, a = q_slot
        g, u_r, a_r = r_slot

        # Check that variable names match
        if u != u_r:
            violations.append(f"Matrix A connects different variables: {u} != {u_r}")

        # Check that label indices match
        if a != a_r:
            violations.append(f"Matrix A connects different labels: {a} != {a_r}")

        # Check that factors are different (no self-loops)
        if g == f:
            violations.append(f"Matrix A has self-loop: variable {u} factor {f}")

        # Check that value is 1 (should only have 1s)
        if abs(value - 1.0) > 1e-10:
            violations.append(f"Matrix A has non-unit entry: A[{row},{col}] = {value}")

    return violations


def check_matrix_B_structure(
    B: csr_matrix,
    snap: Snapshot,
    idxQ: Dict[Tuple[str, str, int], int],
    idxR: Dict[Tuple[str, str, int], int],
) -> List[str]:
    """
    Check matrix B structure invariants.

    B should have exactly one nonzero per neighbor u≠v for each (f→v,a).

    Args:
        B: Factor selection matrix
        snap: Snapshot
        idxQ, idxR: Slot indices

    Returns:
        List of structure violations
    """
    violations = []

    if B.nnz == 0:
        return violations

    # Count nonzeros per row
    row_nonzero_counts = {}
    B_coo = B.tocoo()

    for row, col, value in zip(B_coo.row, B_coo.col, B_coo.data):
        if row not in row_nonzero_counts:
            row_nonzero_counts[row] = 0
        row_nonzero_counts[row] += 1

        # Check that value is 1
        if abs(value - 1.0) > 1e-10:
            violations.append(f"Matrix B has non-unit entry: B[{row},{col}] = {value}")

    # Check expected number of connections per R slot
    reverse_idxR = {idx: slot_key for slot_key, idx in idxR.items()}

    for row_idx, actual_connections in row_nonzero_counts.items():
        if row_idx not in reverse_idxR:
            continue

        r_slot = reverse_idxR[row_idx]  # (factor_name, var_name, label_idx)
        if len(r_slot) != 3:
            continue

        factor_name, var_name, label_idx = r_slot

        # Expected connections = |∂f| - 1 (all neighbors except target variable)
        connected_vars = snap.N_fac.get(factor_name, [])
        expected_connections = len(connected_vars) - 1

        if actual_connections != expected_connections:
            violations.append(
                f"Matrix B row {row_idx} (R[{factor_name},{var_name},{label_idx}]) "
                f"has {actual_connections} connections, expected {expected_connections}"
            )

    return violations


def check_cycle_alternation(
    cycles_detail: List[Dict[str, Any]], slot_graph
) -> List[str]:
    """
    Check that cycles properly alternate between R and Q slots.

    Args:
        cycles_detail: List of cycle information
        slot_graph: NetworkX slot graph

    Returns:
        List of alternation violations
    """
    violations = []

    # This is a simplified check - would need actual cycle node sequences
    # to properly validate alternation

    for i, cycle_info in enumerate(cycles_detail):
        cycle_length = cycle_info.get("length", 0)

        # Cycles should have even length (alternating R→Q→R→Q...)
        if cycle_length % 2 != 0:
            violations.append(
                f"Cycle {i} has odd length {cycle_length} - should alternate R↔Q"
            )

    return violations


def check_domain_consistency(snap: Snapshot) -> List[str]:
    """
    Check consistency of domain specifications across the snapshot.

    Args:
        snap: Snapshot to validate

    Returns:
        List of domain consistency violations
    """
    violations = []

    # Check that all referenced variables have domains
    all_vars = set()

    # Collect variables from N_var
    for var_name in snap.N_var:
        all_vars.add(var_name)

    # Collect variables from N_fac
    for factor_name, var_list in snap.N_fac.items():
        all_vars.update(var_list)

    # Collect variables from Q messages
    for var_name, factor_name in snap.Q:
        all_vars.add(var_name)

    # Collect variables from R messages
    for factor_name, var_name in snap.R:
        all_vars.add(var_name)

    # Check that all variables have domains
    for var_name in all_vars:
        if var_name not in snap.dom:
            violations.append(
                f"Variable {var_name} referenced but not in domain specification"
            )
        elif not snap.dom[var_name]:
            violations.append(f"Variable {var_name} has empty domain")

    # Check message dimensions match domains
    for (var_name, factor_name), q_msg in snap.Q.items():
        if var_name in snap.dom:
            expected_size = len(snap.dom[var_name])
            if len(q_msg) != expected_size:
                violations.append(
                    f"Q[({var_name},{factor_name})] has size {len(q_msg)}, "
                    f"expected {expected_size} (domain size)"
                )

    for (factor_name, var_name), r_msg in snap.R.items():
        if var_name in snap.dom:
            expected_size = len(snap.dom[var_name])
            if len(r_msg) != expected_size:
                violations.append(
                    f"R[({factor_name},{var_name})] has size {len(r_msg)}, "
                    f"expected {expected_size} (domain size)"
                )

    return violations


def check_graph_consistency(snap: Snapshot) -> List[str]:
    """
    Check consistency of graph structure (N_var, N_fac).

    Args:
        snap: Snapshot to validate

    Returns:
        List of graph consistency violations
    """
    violations = []

    # Check bidirectional consistency: if f ∈ N_var[u], then u ∈ N_fac[f]
    for var_name, factor_list in snap.N_var.items():
        for factor_name in factor_list:
            if factor_name not in snap.N_fac:
                violations.append(
                    f"Factor {factor_name} in N_var[{var_name}] but not in N_fac"
                )
            elif var_name not in snap.N_fac[factor_name]:
                violations.append(
                    f"Variable {var_name} connects to factor {factor_name} "
                    f"but {factor_name} doesn't connect back"
                )

    # Check reverse direction: if u ∈ N_fac[f], then f ∈ N_var[u]
    for factor_name, var_list in snap.N_fac.items():
        for var_name in var_list:
            if var_name not in snap.N_var:
                violations.append(
                    f"Variable {var_name} in N_fac[{factor_name}] but not in N_var"
                )
            elif factor_name not in snap.N_var[var_name]:
                violations.append(
                    f"Factor {factor_name} connects to variable {var_name} "
                    f"but {var_name} doesn't connect back"
                )

    return violations


def check_lambda_value(
    snap: Snapshot, config: Optional[AnalysisConfig] = None
) -> List[str]:
    """
    Check that lambda value is in valid range [0,1).

    Args:
        snap: Snapshot
        config: Analysis configuration

    Returns:
        List of lambda violations
    """
    violations = []

    lambda_val = snap.lambda_

    if not (0 <= lambda_val < 1):
        violations.append(f"Lambda value {lambda_val} not in range [0,1)")

    # Additional sanity checks
    if not isinstance(lambda_val, (int, float)):
        violations.append(f"Lambda value {lambda_val} is not numeric")

    if np.isnan(lambda_val) or np.isinf(lambda_val):
        violations.append(f"Lambda value {lambda_val} is NaN or infinite")

    return violations


def run_all_invariant_checks(
    snap: Snapshot,
    A: csr_matrix,
    P: csr_matrix,
    B: csr_matrix,
    idxQ: Dict[Tuple[str, str, int], int],
    idxR: Dict[Tuple[str, str, int], int],
    cycles_detail: List[Dict[str, Any]],
    config: Optional[AnalysisConfig] = None,
) -> Dict[str, List[str]]:
    """
    Run all invariant checks and return categorized violations.

    Args:
        snap: Snapshot
        A, P, B: System matrices
        idxQ, idxR: Slot indices
        cycles_detail: Cycle analysis results
        config: Analysis configuration

    Returns:
        Dictionary categorizing all violations by type
    """
    if config is None:
        config = AnalysisConfig()

    all_violations = {}

    if config.check_invariants:
        # Basic snapshot checks
        all_violations["lambda"] = check_lambda_value(snap, config)
        all_violations["domains"] = check_domain_consistency(snap)
        all_violations["graph"] = check_graph_consistency(snap)

        # Message normalization
        if config.validate_normalization:
            all_violations["normalization"] = check_normalization_invariants(
                snap, config
            )

        # Matrix structure checks
        if config.validate_matrix_structure:
            all_violations["matrix_A"] = check_matrix_A_structure(A, snap, idxQ, idxR)
            all_violations["matrix_B"] = check_matrix_B_structure(B, snap, idxQ, idxR)

        # Projector checks
        if config.validate_projector:
            all_violations["projector"] = check_projector_invariants(P, idxQ, config)

        # Cycle structure checks
        all_violations["cycles"] = check_cycle_alternation(cycles_detail, None)

    # Filter out empty violation lists
    all_violations = {k: v for k, v in all_violations.items() if v}

    return all_violations


def generate_invariant_report(violations: Dict[str, List[str]]) -> str:
    """
    Generate a formatted report of invariant violations.

    Args:
        violations: Dictionary of violations by category

    Returns:
        Formatted report string
    """
    if not violations:
        return "✓ All invariant checks passed - no violations found."

    report_lines = ["=== INVARIANT VIOLATIONS REPORT ==="]

    total_violations = sum(len(v) for v in violations.values())
    report_lines.append(f"Total violations: {total_violations}")
    report_lines.append("")

    for category, violation_list in violations.items():
        report_lines.append(f"{category.upper()} ({len(violation_list)} violations):")
        for i, violation in enumerate(violation_list, 1):
            report_lines.append(f"  {i}. {violation}")
        report_lines.append("")

    report_lines.append("=== END REPORT ===")

    return "\n".join(report_lines)


def get_violation_summary(violations: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Get summary statistics about violations.

    Args:
        violations: Dictionary of violations by category

    Returns:
        Summary statistics
    """
    total_violations = sum(len(v) for v in violations.values())

    summary = {
        "total_violations": total_violations,
        "categories_with_violations": len(violations),
        "violation_counts": {k: len(v) for k, v in violations.items()},
        "all_clear": total_violations == 0,
    }

    if violations:
        summary["most_violated_category"] = max(
            violations.keys(), key=lambda k: len(violations[k])
        )
        summary["severity"] = (
            "high"
            if total_violations > 10
            else ("medium" if total_violations > 3 else "low")
        )

    return summary


# Example usage and testing
if __name__ == "__main__":
    from .snapshot import SimpleSnapshot
    from .slot_indices import build_slot_indices
    from .winners import compute_winners, compute_min_idx
    from .matrices import build_A, build_P, build_B
    import numpy as np

    print("=== Invariant Validation Examples ===")

    # Create test snapshot (some violations intentional)
    test_snapshot = SimpleSnapshot(
        _lambda=0.5,  # Valid lambda
        _dom={"x1": ["0", "1"], "x2": ["0", "1"]},
        _N_var={"x1": ["f1"], "x2": ["f1"]},
        _N_fac={"f1": ["x1", "x2"]},
        _Q={
            ("x1", "f1"): np.array([0.1, 0.3]),  # NOT min-normalized (violation)
            ("x2", "f1"): np.array([0.0, 0.2]),  # Properly min-normalized
        },
        _R={("f1", "x1"): np.array([0.1, 0.0]), ("f1", "x2"): np.array([0.0, 0.1])},
        _unary={"x1": np.zeros(2), "x2": np.zeros(2)},
        _cost={"f1": lambda assign: 0.0},
        _split_map={},
    )

    # Build analysis components
    idxQ, idxR = build_slot_indices(test_snapshot)
    winners = compute_winners(test_snapshot)
    min_idx = compute_min_idx(test_snapshot)

    A = build_A(test_snapshot, idxQ, idxR)
    P = build_P(test_snapshot, min_idx, idxQ)
    B = build_B(test_snapshot, winners, idxQ, idxR)

    print(f"Built matrices: A({A.shape}), P({P.shape}), B({B.shape})")

    # Run invariant checks
    config = AnalysisConfig(check_invariants=True, validate_normalization=True)

    violations = run_all_invariant_checks(
        test_snapshot, A, P, B, idxQ, idxR, [], config
    )

    # Generate report
    report = generate_invariant_report(violations)
    print("\n" + report)

    # Get summary
    summary = get_violation_summary(violations)
    print(f"\nViolation summary: {summary}")

    # Test with valid snapshot
    valid_snapshot = SimpleSnapshot(
        _lambda=0.3,
        _dom={"x1": ["0", "1"], "x2": ["0", "1"]},
        _N_var={"x1": ["f1"], "x2": ["f1"]},
        _N_fac={"f1": ["x1", "x2"]},
        _Q={
            ("x1", "f1"): np.array([0.0, 0.3]),  # Properly min-normalized
            ("x2", "f1"): np.array([0.0, 0.2]),  # Properly min-normalized
        },
        _R={("f1", "x1"): np.array([0.1, 0.0]), ("f1", "x2"): np.array([0.0, 0.1])},
        _unary={"x1": np.zeros(2), "x2": np.zeros(2)},
        _cost={"f1": lambda assign: 0.0},
        _split_map={},
    )

    # Test valid snapshot
    idxQ_v, idxR_v = build_slot_indices(valid_snapshot)
    A_v = build_A(valid_snapshot, idxQ_v, idxR_v)
    P_v = build_P(valid_snapshot, compute_min_idx(valid_snapshot), idxQ_v)
    B_v = build_B(valid_snapshot, compute_winners(valid_snapshot), idxQ_v, idxR_v)

    valid_violations = run_all_invariant_checks(
        valid_snapshot, A_v, P_v, B_v, idxQ_v, idxR_v, [], config
    )

    valid_report = generate_invariant_report(valid_violations)
    print(f"\nValid snapshot check:\n{valid_report}")

    print("\n=== Ready for final orchestrator ===")
