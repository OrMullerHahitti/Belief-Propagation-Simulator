"""
Jacobian matrix construction for belief propagation analysis.

Builds the A (variable aggregation), P (min-normalization projector),
and B (factor selection) matrices for slot-cycle analysis.
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from typing import Dict, Tuple, List, Optional
from .snapshot import Snapshot
from .slot_indices import build_slot_indices, unique_edges_from_idxQ
from .config import AnalysisConfig


def build_A(
    snap: Snapshot,
    idxQ: Dict[Tuple[str, str, int], int],
    idxR: Dict[Tuple[str, str, int], int],
) -> csr_matrix:
    """
    Build variable aggregation matrix A (post-split).

    A has nonzeros: A[(u→f),a; (g→u),a] = 1 for all g ∈ N(u) ∖ {f}.

    This matrix represents how Q messages are aggregated from incoming R messages
    at variable nodes, excluding the message back to the sender.

    Args:
        snap: Snapshot containing graph structure
        idxQ: Q slot indices mapping (u,f,a) -> row_index
        idxR: R slot indices mapping (f,v,a) -> col_index

    Returns:
        Sparse matrix A of shape (nQ, nR)
    """
    rows, cols, data = [], [], []

    # For each Q message (u->f)
    for var_name, factor_name, label_idx in idxQ:
        row_idx = idxQ[(var_name, factor_name, label_idx)]

        # Find all factors connected to this variable (except the target factor)
        neighbor_factors = snap.N_var.get(var_name, [])

        for neighbor_factor in neighbor_factors:
            if neighbor_factor == factor_name:
                continue  # Skip message back to sender

            # Add connection from R message (neighbor_factor->var_name, same label)
            r_key = (neighbor_factor, var_name, label_idx)
            if r_key in idxR:
                col_idx = idxR[r_key]
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(1.0)

    n_rows = len(idxQ)
    n_cols = len(idxR)

    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


def build_P(
    snap: Snapshot,
    min_idx: Dict[Tuple[str, str], int],
    idxQ: Dict[Tuple[str, str, int], int],
) -> csr_matrix:
    """
    Build min-normalization projector matrix P.

    P is block-diagonal with blocks P_{u→f} = I - 1·e_{m(u→f)}^T
    where m(u→f) is the minimum index for Q message u→f.

    This projects Q messages to have minimum value 0 (min-normalization).

    Args:
        snap: Snapshot
        min_idx: Min indices for Q messages
        idxQ: Q slot indices mapping (u,f,a) -> index

    Returns:
        Sparse matrix P of shape (nQ, nQ)
    """
    nQ = len(idxQ)
    P = lil_matrix((nQ, nQ))

    # Group Q slots by edge (u,f)
    edge_slots = {}
    for var_name, factor_name, label_idx in idxQ:
        edge = (var_name, factor_name)
        if edge not in edge_slots:
            edge_slots[edge] = []
        edge_slots[edge].append((label_idx, idxQ[(var_name, factor_name, label_idx)]))

    # Build block for each edge
    for (var_name, factor_name), slot_list in edge_slots.items():
        # Sort by label index to ensure consistent ordering
        slot_list.sort()

        # Get minimum index for this edge
        min_label_idx = min_idx.get((var_name, factor_name), 0)

        # Find the slot index corresponding to the minimum
        min_slot_idx = None
        for label_idx, slot_idx in slot_list:
            if label_idx == min_label_idx:
                min_slot_idx = slot_idx
                break

        if min_slot_idx is None:
            # Fallback to first slot if min not found
            min_slot_idx = slot_list[0][1] if slot_list else 0

        # Fill block: I - 1·e_m^T
        for label_idx, slot_idx in slot_list:
            # Identity part: P[i,i] = 1
            P[slot_idx, slot_idx] = 1.0

            # Subtraction part: P[i,m] -= 1 for all i
            P[slot_idx, min_slot_idx] -= 1.0

    return P.tocsr()


def build_B(
    snap: Snapshot,
    winners: Dict[Tuple[str, str, str], Dict[str, str]],
    idxQ: Dict[Tuple[str, str, int], int],
    idxR: Dict[Tuple[str, str, int], int],
) -> csr_matrix:
    """
    Build factor selection Jacobian matrix B.

    B has nonzeros: B[(f→v),a; (u→f),x*_{u→f}(a)] = 1
    where x*_{u→f}(a) is the winning label for variable u when factor f
    sends message to variable v with target label a.

    Args:
        snap: Snapshot
        winners: Winner assignments
        idxQ: Q slot indices
        idxR: R slot indices

    Returns:
        Sparse matrix B of shape (nR, nQ)
    """
    rows, cols, data = [], [], []

    # For each R message (f->v)
    for factor_name, var_name, label_idx in idxR:
        row_idx = idxR[(factor_name, var_name, label_idx)]

        # Find connected variables (excluding target variable)
        connected_vars = snap.N_fac.get(factor_name, [])
        other_vars = [u for u in connected_vars if u != var_name]

        # Get target label string
        target_label = (
            snap.dom[var_name][label_idx]
            if label_idx < len(snap.dom[var_name])
            else "0"
        )

        # Look up winner assignment
        winner_key = (factor_name, var_name, target_label)
        winner_assignment = winners.get(winner_key, {})

        # For each other variable, add connection to its winning Q message
        for other_var in other_vars:
            if other_var in winner_assignment:
                winning_label = winner_assignment[other_var]

                # Convert winning label to index
                try:
                    winning_label_idx = snap.dom[other_var].index(winning_label)
                except (ValueError, KeyError):
                    winning_label_idx = 0  # Fallback to first label

                # Add connection to Q message (other_var -> factor_name, winning_label)
                q_key = (other_var, factor_name, winning_label_idx)
                if q_key in idxQ:
                    col_idx = idxQ[q_key]
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(1.0)

    n_rows = len(idxR)
    n_cols = len(idxQ)

    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


def compute_block_norms(
    A: csr_matrix, P: csr_matrix, B: csr_matrix, lambda_val: float
) -> Dict[str, Optional[float]]:
    """
    Compute block norms and crude global bound.

    Computes induced ∞-norm components and upper bound for ||M_Q^(min)||_∞:
    - ||BPA||_∞ (scaled by 1-λ)
    - ||B||_∞ (scaled by λ)
    - ||PA||_∞ (scaled by 1-λ)
    - ||M||_∞_upper (crude upper bound)

    Args:
        A: Variable aggregation matrix
        P: Min-normalization projector
        B: Factor selection matrix
        lambda_val: Damping parameter λ

    Returns:
        Dictionary with norm values
    """

    def row_sum_norm(M: csr_matrix) -> float:
        """Compute infinity norm (max row sum) of sparse matrix."""
        if M.nnz == 0:
            return 0.0
        M_abs = abs(M)
        row_sums = np.array(M_abs.sum(axis=1)).ravel()
        return float(np.max(row_sums))

    # Compute matrix products
    PA = P @ A
    BPA = B @ PA

    # Top-left block: (1-λ) * (B @ P @ A)
    tl_norm = row_sum_norm(BPA) if lambda_val < 1 else 0.0
    tl_scaled = tl_norm / (1 - lambda_val) if lambda_val < 1 else None

    # Top-right block: λ * B
    tr_norm = row_sum_norm(B) if lambda_val > 0 else 0.0
    tr_scaled = tr_norm / lambda_val if lambda_val > 0 else None

    # Bottom-left block: (1-λ) * (P @ A)
    bl_norm = row_sum_norm(PA) if lambda_val < 1 else 0.0
    bl_scaled = bl_norm / (1 - lambda_val) if lambda_val < 1 else None

    # Crude upper bound for ||M||_∞
    upper_bound = max(
        tl_norm + tr_norm,  # Top row blocks
        bl_norm + lambda_val,  # Bottom row blocks (bottom-right is λI)
    )

    return {
        "||BPA||_inf": tl_scaled,
        "||B||_inf": tr_scaled,
        "||PA||_inf": bl_scaled,
        "||M||_inf_upper": upper_bound,
    }


def validate_matrix_structure(
    snap: Snapshot,
    A: csr_matrix,
    P: csr_matrix,
    B: csr_matrix,
    idxQ: Dict[Tuple[str, str, int], int],
    idxR: Dict[Tuple[str, str, int], int],
) -> List[str]:
    """
    Validate matrix structures against expected properties.

    Args:
        snap: Snapshot
        A, P, B: Matrices to validate
        idxQ, idxR: Slot indices

    Returns:
        List of validation errors
    """
    errors = []

    # Validate dimensions
    nQ, nR = len(idxQ), len(idxR)

    if A.shape != (nQ, nR):
        errors.append(f"Matrix A has shape {A.shape}, expected ({nQ}, {nR})")

    if P.shape != (nQ, nQ):
        errors.append(f"Matrix P has shape {P.shape}, expected ({nQ}, {nQ})")

    if B.shape != (nR, nQ):
        errors.append(f"Matrix B has shape {B.shape}, expected ({nR}, {nQ})")

    # Validate A structure: should only connect same labels
    A_coo = A.tocoo()
    for i, (row, col) in enumerate(zip(A_coo.row, A_coo.col)):
        # Find corresponding slot keys
        row_slot = None
        col_slot = None

        for slot_key, idx in idxQ.items():
            if idx == row:
                row_slot = slot_key
                break

        for slot_key, idx in idxR.items():
            if idx == col:
                col_slot = slot_key
                break

        if row_slot and col_slot:
            # Row slot: (u, f, a), Col slot: (g, u, a)
            u, f, a = row_slot
            g, u_col, a_col = col_slot

            if u != u_col:
                errors.append(f"Matrix A connects different variables: {u} != {u_col}")
            if a != a_col:
                errors.append(f"Matrix A connects different labels: {a} != {a_col}")
            if g == f:
                errors.append(f"Matrix A has self-loop: variable {u} factor {f}")

    # Validate P projector properties (sample check)
    if P.shape[0] > 0 and P.shape[1] > 0:
        # Test on a random vector
        test_vec = np.random.randn(min(P.shape[1], 10))
        if len(test_vec) <= P.shape[1]:
            full_test_vec = np.zeros(P.shape[1])
            full_test_vec[: len(test_vec)] = test_vec

            Px = P @ full_test_vec
            PPx = P @ Px

            # Check P^2 ≈ P (projector property)
            diff_norm = np.linalg.norm(PPx - Px)
            if diff_norm > 1e-6:
                errors.append(
                    f"Matrix P projector property violated: ||P^2x - Px|| = {diff_norm:.2e}"
                )

    # Validate B structure: should have exactly one nonzero per factor neighborhood
    B_coo = B.tocoo()
    row_nonzero_counts = {}

    for row, col in zip(B_coo.row, B_coo.col):
        if row not in row_nonzero_counts:
            row_nonzero_counts[row] = 0
        row_nonzero_counts[row] += 1

    # Each R slot should connect to exactly |∂f| - 1 Q slots (one per neighbor except target)
    for (factor_name, var_name, label_idx), row_idx in idxR.items():
        connected_vars = snap.N_fac.get(factor_name, [])
        expected_connections = len(connected_vars) - 1  # Exclude target variable

        actual_connections = row_nonzero_counts.get(row_idx, 0)
        if actual_connections != expected_connections:
            errors.append(
                f"Matrix B row {row_idx} (R[{factor_name},{var_name},{label_idx}]) has {actual_connections} connections, expected {expected_connections}"
            )

    return errors


def get_matrix_statistics(
    A: csr_matrix, P: csr_matrix, B: csr_matrix
) -> Dict[str, any]:
    """
    Compute statistics about the matrices.

    Args:
        A, P, B: Matrices to analyze

    Returns:
        Dictionary with statistics
    """

    def matrix_stats(M: csr_matrix, name: str) -> Dict[str, any]:
        if M.nnz == 0:
            return {
                f"{name}_shape": M.shape,
                f"{name}_nnz": 0,
                f"{name}_density": 0.0,
                f"{name}_max_value": 0.0,
                f"{name}_min_value": 0.0,
            }

        return {
            f"{name}_shape": M.shape,
            f"{name}_nnz": M.nnz,
            f"{name}_density": M.nnz / (M.shape[0] * M.shape[1])
            if M.shape[0] * M.shape[1] > 0
            else 0.0,
            f"{name}_max_value": float(M.data.max()),
            f"{name}_min_value": float(M.data.min()),
        }

    stats = {}
    stats.update(matrix_stats(A, "A"))
    stats.update(matrix_stats(P, "P"))
    stats.update(matrix_stats(B, "B"))

    # Combined statistics
    total_nnz = A.nnz + P.nnz + B.nnz
    total_elements = (
        A.shape[0] * A.shape[1] + P.shape[0] * P.shape[1] + B.shape[0] * B.shape[1]
    )

    stats.update(
        {
            "total_nnz": total_nnz,
            "total_elements": total_elements,
            "overall_density": total_nnz / total_elements
            if total_elements > 0
            else 0.0,
        }
    )

    return stats


# Example usage and testing
if __name__ == "__main__":
    from .snapshot import SimpleSnapshot
    from .winners import compute_winners, compute_min_idx
    import numpy as np

    print("=== Matrix Construction Examples ===")

    # Create test snapshot
    test_snapshot = SimpleSnapshot(
        _lambda=0.6,
        _dom={"x1": ["0", "1"], "x2": ["0", "1"], "x3": ["0", "1"]},
        _N_var={"x1": ["f1", "f2"], "x2": ["f1"], "x3": ["f2"]},
        _N_fac={"f1": ["x1", "x2"], "f2": ["x1", "x3"]},
        _Q={
            ("x1", "f1"): np.array([0.0, 0.3]),
            ("x1", "f2"): np.array([0.1, 0.0]),
            ("x2", "f1"): np.array([0.0, 0.2]),
            ("x3", "f2"): np.array([0.2, 0.0]),
        },
        _R={
            ("f1", "x1"): np.array([0.1, 0.0]),
            ("f1", "x2"): np.array([0.0, 0.1]),
            ("f2", "x1"): np.array([0.2, 0.0]),
            ("f2", "x3"): np.array([0.0, 0.3]),
        },
        _unary={"x1": np.zeros(2), "x2": np.zeros(2), "x3": np.zeros(2)},
        _cost={
            "f1": lambda assign: 0.1 if assign.get("x1") == assign.get("x2") else 0.0,
            "f2": lambda assign: 0.2 if assign.get("x1") == assign.get("x3") else 0.0,
        },
        _split_map={},
    )

    # Build slot indices
    idxQ, idxR = build_slot_indices(test_snapshot)
    print(f"Built slot indices: {len(idxQ)} Q slots, {len(idxR)} R slots")

    # Compute winners and min indices
    winners = compute_winners(test_snapshot)
    min_idx = compute_min_idx(test_snapshot)
    print(f"Computed {len(winners)} winners, {len(min_idx)} min indices")

    # Build matrices
    A = build_A(test_snapshot, idxQ, idxR)
    P = build_P(test_snapshot, min_idx, idxQ)
    B = build_B(test_snapshot, winners, idxQ, idxR)

    print(f"Matrix A: {A.shape}, {A.nnz} nonzeros")
    print(f"Matrix P: {P.shape}, {P.nnz} nonzeros")
    print(f"Matrix B: {B.shape}, {B.nnz} nonzeros")

    # Compute norms
    norms = compute_block_norms(A, P, B, test_snapshot.lambda_)
    print(f"Block norms: {norms}")

    # Validate matrices
    errors = validate_matrix_structure(test_snapshot, A, P, B, idxQ, idxR)
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✓ Matrices are valid")

    # Get statistics
    stats = get_matrix_statistics(A, P, B)
    print(f"Matrix statistics: {stats}")

    print("\n=== Ready for cycle analysis ===")
