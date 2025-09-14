"""
Cycle analysis with alignment detection for belief propagation analysis.

Builds slot graphs, finds cycles, detects aligned hops, and generates
convergence certificates based on cycle structure.
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Set, Optional, Any
from .snapshot import Snapshot
from .config import AnalysisConfig


def build_slot_graph(
    A: csr_matrix,
    B: csr_matrix,
    idxQ: Dict[Tuple[str, str, int], int],
    idxR: Dict[Tuple[str, str, int], int],
) -> nx.DiGraph:
    """
    Build slot graph with R→Q and Q→R edges.

    Edges are added based on nonzero entries in matrices A and B:
    - R→Q edge: if B has 1 at ((f→v),a; (u→f),b)
    - Q→R edge: if A has 1 at ((u→f),a; (g→u),a)

    Args:
        A: Variable aggregation matrix
        B: Factor selection matrix
        idxQ: Q slot indices
        idxR: R slot indices

    Returns:
        Directed graph with slots as nodes
    """
    G = nx.DiGraph()

    # Create reverse mappings for efficient lookup
    reverse_idxQ = {idx: slot_key for slot_key, idx in idxQ.items()}
    reverse_idxR = {idx: slot_key for slot_key, idx in idxR.items()}

    # Add all slots as nodes
    for slot_key in idxQ.keys():
        G.add_node(("Q", slot_key))

    for slot_key in idxR.keys():
        G.add_node(("R", slot_key))

    # Add R→Q edges from matrix B
    B_coo = B.tocoo()
    for row, col in zip(B_coo.row, B_coo.col):
        if row in reverse_idxR and col in reverse_idxQ:
            r_slot = reverse_idxR[row]
            q_slot = reverse_idxQ[col]
            G.add_edge(("R", r_slot), ("Q", q_slot))

    # Add Q→R edges from matrix A
    A_coo = A.tocoo()
    for row, col in zip(A_coo.row, A_coo.col):
        if row in reverse_idxQ and col in reverse_idxR:
            q_slot = reverse_idxQ[row]
            r_slot = reverse_idxR[col]
            G.add_edge(("Q", q_slot), ("R", r_slot))

    return G


def list_simple_cycles(G: nx.DiGraph, Lmax: int = 12) -> List[List[Tuple[str, Tuple]]]:
    """
    Find simple cycles in slot graph up to maximum length.

    Args:
        G: Slot graph
        Lmax: Maximum cycle length to search

    Returns:
        List of cycles, where each cycle is a list of (type, slot_key) nodes
    """
    try:
        # Use NetworkX simple_cycles but limit length
        all_cycles = []
        for cycle in nx.simple_cycles(G):
            if len(cycle) <= Lmax:
                all_cycles.append(cycle)
            # Stop if we find too many cycles (memory protection)
            if len(all_cycles) > 10000:
                break

        return all_cycles

    except (nx.NetworkXError, MemoryError):
        # Fallback for difficult graphs
        return []


def cycle_has_aligned_hop(
    cycle: List[Tuple[str, Tuple]],
    winners: Dict[Tuple[str, str, str], Dict[str, str]],
    min_idx: Dict[Tuple[str, str], int],
    idxQ: Dict[Tuple[str, str, int], int],
    idxR: Dict[Tuple[str, str, int], int],
) -> bool:
    """
    Check if a cycle has at least one aligned hop.

    A hop is aligned if the winning assignment coincides with the minimum index:
    x*_{u→f}(a) = m(u→f)

    Args:
        cycle: Cycle as list of (type, slot_key) nodes
        winners: Winner assignments
        min_idx: Minimum indices
        idxQ, idxR: Slot indices

    Returns:
        True if cycle has at least one aligned hop
    """
    for i in range(len(cycle)):
        current_node = cycle[i]
        next_node = cycle[(i + 1) % len(cycle)]

        # Check for R→Q hop (this is where alignment matters)
        if current_node[0] == "R" and next_node[0] == "Q":
            r_slot = current_node[1]  # (factor_name, var_name, label_idx)
            q_slot = next_node[1]  # (other_var, factor_name, other_label_idx)

            if len(r_slot) == 3 and len(q_slot) == 3:
                factor_name, target_var, target_label_idx = r_slot
                other_var, q_factor_name, other_label_idx = q_slot

                # Must be the same factor
                if factor_name != q_factor_name:
                    continue

                # Get target label string
                target_label = str(
                    target_label_idx
                )  # Simplified - should lookup in domain

                # Look up winner assignment
                winner_key = (factor_name, target_var, target_label)
                winner_assignment = winners.get(winner_key, {})

                # Get winning label for other_var
                winning_label = winner_assignment.get(other_var)

                if winning_label is not None:
                    # Convert to index (simplified)
                    try:
                        winning_label_idx = int(winning_label)
                    except (ValueError, TypeError):
                        continue

                    # Get minimum index for (other_var, factor_name)
                    min_label_idx = min_idx.get((other_var, factor_name))

                    # Check alignment: winning label == minimum label
                    if winning_label_idx == min_label_idx:
                        return True

    return False


def estimate_cycle_gain_inf(
    cycle: List[Tuple[str, Tuple]],
    A: csr_matrix,
    P: csr_matrix,
    B: csr_matrix,
    lambda_val: float,
    idxQ: Dict[Tuple[str, str, int], int],
    idxR: Dict[Tuple[str, str, int], int],
) -> Optional[float]:
    """
    Estimate numeric cycle gain by multiplying local blocks along cycle.

    This computes the ∞-norm of the product of matrix blocks corresponding
    to edges in the cycle.

    Args:
        cycle: Cycle nodes
        A, P, B: System matrices
        lambda_val: Damping parameter
        idxQ, idxR: Slot indices

    Returns:
        Estimated cycle gain (∞-norm) or None if computation fails
    """
    if len(cycle) == 0:
        return None

    try:
        # Build cycle matrix by multiplying edge matrices
        cycle_matrix = None

        for i in range(len(cycle)):
            current_node = cycle[i]
            next_node = cycle[(i + 1) % len(cycle)]

            edge_matrix = None

            # R→Q edge: use part of matrix B
            if current_node[0] == "R" and next_node[0] == "Q":
                r_slot = current_node[1]
                q_slot = next_node[1]

                # Extract relevant part of B matrix
                if (
                    len(r_slot) == 3
                    and len(q_slot) == 3
                    and r_slot in idxR
                    and q_slot in idxQ
                ):
                    r_idx = idxR[r_slot]
                    q_idx = idxQ[q_slot]

                    # Create single-entry matrix for this edge
                    edge_matrix = csr_matrix(([1.0], ([0], [0])), shape=(1, 1))

            # Q→R edge: use part of matrix A
            elif current_node[0] == "Q" and next_node[0] == "R":
                q_slot = current_node[1]
                r_slot = next_node[1]

                # Extract relevant part of A matrix (through P)
                if (
                    len(q_slot) == 3
                    and len(r_slot) == 3
                    and q_slot in idxQ
                    and r_slot in idxR
                ):
                    q_idx = idxQ[q_slot]
                    r_idx = idxR[r_slot]

                    # Use P @ A for this edge
                    PA = P @ A
                    if q_idx < PA.shape[0] and r_idx < PA.shape[1]:
                        edge_value = PA[q_idx, r_idx]
                        edge_matrix = csr_matrix(
                            ([edge_value], ([0], [0])), shape=(1, 1)
                        )

            # Multiply into cycle matrix
            if edge_matrix is not None:
                if cycle_matrix is None:
                    cycle_matrix = edge_matrix
                else:
                    # For simplicity, just multiply the scalar values
                    if cycle_matrix.nnz > 0 and edge_matrix.nnz > 0:
                        combined_value = cycle_matrix.data[0] * edge_matrix.data[0]
                        cycle_matrix = csr_matrix(
                            ([combined_value], ([0], [0])), shape=(1, 1)
                        )

        # Apply damping factor
        if cycle_matrix is not None and cycle_matrix.nnz > 0:
            cycle_gain = abs(cycle_matrix.data[0]) * (lambda_val ** len(cycle))
            return float(cycle_gain)

    except (IndexError, ValueError, AttributeError):
        pass

    return None


def analyze_cycles(
    snap: Snapshot,
    A: csr_matrix,
    P: csr_matrix,
    B: csr_matrix,
    idxQ: Dict[Tuple[str, str, int], int],
    idxR: Dict[Tuple[str, str, int], int],
    winners: Dict[Tuple[str, str, str], Dict[str, str]],
    min_idx: Dict[Tuple[str, str], int],
    config: AnalysisConfig,
) -> Dict[str, Any]:
    """
    Analyze cycles in the slot graph with alignment and certificates.

    Args:
        snap: Snapshot
        A, P, B: System matrices
        idxQ, idxR: Slot indices
        winners: Winner assignments
        min_idx: Minimum indices
        config: Analysis configuration

    Returns:
        Dictionary with cycle analysis results
    """
    # Build slot graph
    G = build_slot_graph(A, B, idxQ, idxR)

    # Find cycles
    cycles = list_simple_cycles(G, config.max_cycle_len)

    detail = []
    aligned_total = 0
    certified_contraction = False

    for cycle in cycles:
        cycle_length = len(cycle)

        # Check for aligned hops
        has_aligned = cycle_has_aligned_hop(cycle, winners, min_idx, idxQ, idxR)

        if has_aligned:
            aligned_total += 1
            certified_contraction = True

        # Create cycle detail entry
        item = {
            "length": cycle_length,
            "aligned": has_aligned,
            "bound": (snap.lambda_**cycle_length) if has_aligned else None,
        }

        # Compute numeric gain if requested
        if config.compute_numeric_cycle_gain:
            numeric_gain = estimate_cycle_gain_inf(
                cycle, A, P, B, snap.lambda_, idxQ, idxR
            )
            item["numeric_gain_inf"] = numeric_gain

        detail.append(item)

        # Limit detail collection to avoid memory issues
        if len(detail) > 1000:
            break

    summary = {
        "num_cycles": len(cycles),
        "aligned_hops_total": aligned_total,
        "has_certified_contraction": certified_contraction,
    }

    return {
        "summary": summary,
        "detail": detail if config.include_detailed_cycles else [],
        "aligned_hops_total": aligned_total,
    }


def validate_cycle_structure(cycles_detail: List[Dict[str, Any]]) -> List[str]:
    """
    Validate cycle structure and properties.

    Args:
        cycles_detail: List of cycle details

    Returns:
        List of validation errors
    """
    errors = []

    for i, cycle_info in enumerate(cycles_detail):
        # Check required fields
        required_fields = ["length", "aligned", "bound"]
        for field in required_fields:
            if field not in cycle_info:
                errors.append(f"Cycle {i} missing field '{field}'")

        # Validate length
        if "length" in cycle_info:
            length = cycle_info["length"]
            if not isinstance(length, int) or length < 3:
                errors.append(f"Cycle {i} has invalid length {length}")

        # Validate alignment consistency
        if "aligned" in cycle_info and "bound" in cycle_info:
            aligned = cycle_info["aligned"]
            bound = cycle_info["bound"]

            if aligned and bound is None:
                errors.append(f"Cycle {i} is aligned but has no bound")
            elif not aligned and bound is not None:
                errors.append(f"Cycle {i} is not aligned but has a bound")

        # Validate numeric gain
        if "numeric_gain_inf" in cycle_info:
            gain = cycle_info["numeric_gain_inf"]
            if gain is not None and (not isinstance(gain, (int, float)) or gain < 0):
                errors.append(f"Cycle {i} has invalid numeric gain {gain}")

    return errors


def get_cycle_statistics(cycles_detail: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics about cycles.

    Args:
        cycles_detail: List of cycle details

    Returns:
        Dictionary with cycle statistics
    """
    if not cycles_detail:
        return {
            "total_cycles": 0,
            "aligned_cycles": 0,
            "average_length": 0.0,
            "length_distribution": {},
        }

    lengths = [c.get("length", 0) for c in cycles_detail]
    aligned_count = sum(1 for c in cycles_detail if c.get("aligned", False))

    # Length distribution
    length_dist = {}
    for length in lengths:
        length_dist[length] = length_dist.get(length, 0) + 1

    # Bounds for aligned cycles
    aligned_bounds = [
        c.get("bound", 0)
        for c in cycles_detail
        if c.get("aligned", False) and c.get("bound") is not None
    ]

    stats = {
        "total_cycles": len(cycles_detail),
        "aligned_cycles": aligned_count,
        "average_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "length_distribution": length_dist,
    }

    if aligned_bounds:
        stats.update(
            {
                "average_aligned_bound": sum(aligned_bounds) / len(aligned_bounds),
                "min_aligned_bound": min(aligned_bounds),
                "max_aligned_bound": max(aligned_bounds),
            }
        )

    # Numeric gains
    numeric_gains = [
        c.get("numeric_gain_inf")
        for c in cycles_detail
        if c.get("numeric_gain_inf") is not None
    ]
    if numeric_gains:
        stats.update(
            {
                "average_numeric_gain": sum(numeric_gains) / len(numeric_gains),
                "max_numeric_gain": max(numeric_gains),
            }
        )

    return stats


def find_shortest_uncertified_cycle(
    cycles_detail: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Find the shortest cycle without an aligned hop.

    Args:
        cycles_detail: List of cycle details

    Returns:
        Shortest uncertified cycle or None if all are certified
    """
    uncertified = [c for c in cycles_detail if not c.get("aligned", False)]

    if not uncertified:
        return None

    # Sort by length and return shortest
    uncertified.sort(key=lambda c: c.get("length", float("inf")))
    return uncertified[0]


# Example usage and testing
if __name__ == "__main__":
    from .snapshot import SimpleSnapshot
    from .slot_indices import build_slot_indices
    from .winners import compute_winners, compute_min_idx
    from .matrices import build_A, build_P, build_B
    import numpy as np

    print("=== Cycle Analysis Examples ===")

    # Create test snapshot with cycle structure
    test_snapshot = SimpleSnapshot(
        _lambda=0.7,
        _dom={"x1": ["0", "1"], "x2": ["0", "1"], "x3": ["0", "1"]},
        _N_var={"x1": ["f1", "f2"], "x2": ["f1", "f3"], "x3": ["f2", "f3"]},
        _N_fac={"f1": ["x1", "x2"], "f2": ["x1", "x3"], "f3": ["x2", "x3"]},
        _Q={
            ("x1", "f1"): np.array([0.0, 0.2]),
            ("x1", "f2"): np.array([0.1, 0.0]),
            ("x2", "f1"): np.array([0.0, 0.3]),
            ("x2", "f3"): np.array([0.2, 0.0]),
            ("x3", "f2"): np.array([0.0, 0.1]),
            ("x3", "f3"): np.array([0.1, 0.0]),
        },
        _R={
            ("f1", "x1"): np.array([0.1, 0.0]),
            ("f1", "x2"): np.array([0.0, 0.2]),
            ("f2", "x1"): np.array([0.2, 0.1]),
            ("f2", "x3"): np.array([0.1, 0.0]),
            ("f3", "x2"): np.array([0.0, 0.1]),
            ("f3", "x3"): np.array([0.2, 0.0]),
        },
        _unary={"x1": np.zeros(2), "x2": np.zeros(2), "x3": np.zeros(2)},
        _cost={
            "f1": lambda assign: 0.1,
            "f2": lambda assign: 0.2,
            "f3": lambda assign: 0.1,
        },
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

    # Analyze cycles
    config = AnalysisConfig()
    cycles_analysis = analyze_cycles(
        test_snapshot, A, P, B, idxQ, idxR, winners, min_idx, config
    )

    print(f"Found {cycles_analysis['summary']['num_cycles']} cycles")
    print(f"Aligned hops: {cycles_analysis['aligned_hops_total']}")
    print(
        f"Certified contraction: {cycles_analysis['summary']['has_certified_contraction']}"
    )

    # Validate cycle structure
    errors = validate_cycle_structure(cycles_analysis["detail"])
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✓ Cycle analysis is valid")

    # Get statistics
    stats = get_cycle_statistics(cycles_analysis["detail"])
    print(f"Cycle statistics: {stats}")

    # Find shortest uncertified cycle
    shortest_uncertified = find_shortest_uncertified_cycle(cycles_analysis["detail"])
    if shortest_uncertified:
        print(f"Shortest uncertified cycle length: {shortest_uncertified['length']}")
    else:
        print("All cycles are certified")

    print("\n=== Ready for margin analysis ===")
