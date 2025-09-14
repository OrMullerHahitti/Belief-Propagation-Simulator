"""
Complete Jacobian Analysis Module for Min-Sum/Max-Sum Message Passing
Supports both binary (K=2) and general (K>2) domains
Implements theory from "SCFG eliminates loop echo" paper
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import strongly_connected_components
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Core Data Structures
# ============================================================================


class MessageType(Enum):
    """Type of message in the dependency graph"""

    Q_MESSAGE = "Q"  # Variable to Factor
    R_MESSAGE = "R"  # Factor to Variable


@dataclass
class MessageCoordinate:
    """
    Represents a message difference coordinate in the Jacobian.

    Attributes:
        msg_type: Type of message (Q or R)
        sender: Name of sending agent
        recipient: Name of receiving agent
        label_from: Source label (for multi-label, None for aggregated)
        label_to: Target label (for multi-label, None for aggregated)
    """

    msg_type: MessageType
    sender: str
    recipient: str
    label_from: Optional[int] = None  # For K>2: source label
    label_to: Optional[int] = None  # For K>2: target label

    def __hash__(self):
        return hash(
            (self.msg_type, self.sender, self.recipient, self.label_from, self.label_to)
        )

    def __eq__(self, other):
        return (
            self.msg_type == other.msg_type
            and self.sender == other.sender
            and self.recipient == other.recipient
            and self.label_from == other.label_from
            and self.label_to == other.label_to
        )

    def __repr__(self):
        arrow = "→"
        prefix = "ΔQ" if self.msg_type == MessageType.Q_MESSAGE else "ΔR"
        base = f"{prefix}_{self.sender}{arrow}{self.recipient}"

        if self.label_from is not None and self.label_to is not None:
            return f"{base}({self.label_from},{self.label_to})"
        elif self.label_from is not None:
            return f"{base}({self.label_from})"
        return base


@dataclass
class FactorStepDerivative:
    """
    Derivative of a factor step.

    For binary: scalar in {-1, 0, +1}
    For K>2: matrix where entry (i,j) is ∂R_i/∂Q_j

    Attributes:
        factor: Factor name
        from_var: Variable sending Q message
        to_var: Variable receiving R message
        value: Derivative (scalar for binary, matrix for general)
        domain_size: Size of variable domain
        iteration: Iteration number
        is_binary: Whether this is binary domain
    """

    factor: str
    from_var: str
    to_var: str
    value: Union[int, np.ndarray]
    domain_size: int
    iteration: int
    is_binary: bool = True

    @property
    def is_neutral(self) -> bool:
        """Check if derivative represents neutral step(s)"""
        if self.is_binary:
            return self.value == 0
        else:
            # For multi-label: neutral if same column wins for all rows
            if isinstance(self.value, np.ndarray):
                # Check if all rows select same column
                selected_cols = np.argmax(self.value, axis=1)
                return len(np.unique(selected_cols)) == 1
            return False

    def get_derivative(self, i: Optional[int] = None, j: Optional[int] = None) -> float:
        """
        Get specific derivative entry.

        Args:
            i: Row index (for multi-label)
            j: Column index (for multi-label)

        Returns:
            Derivative value
        """
        if self.is_binary:
            return float(self.value)
        else:
            if i is not None and j is not None:
                return float(self.value[i, j])
            return 0.0


# ============================================================================
# Threshold Structures for Neutrality
# ============================================================================


@dataclass
class BinaryThresholds:
    """
    Thresholds for binary neutrality.

    Attributes:
        theta_0: Threshold to force x_j = 0
        theta_1: Threshold to force x_j = 1
    """

    theta_0: float  # Force x_j = 0
    theta_1: float  # Force x_j = 1

    def check_neutrality(self, delta_q: float) -> Tuple[bool, Optional[int]]:
        """
        Check if query difference causes neutrality.

        Args:
            delta_q: Query difference ΔQ

        Returns:
            (is_neutral, forced_value) where forced_value is 0, 1, or None
        """
        if delta_q >= self.theta_0:
            return True, 0
        elif delta_q <= -self.theta_1:
            return True, 1
        else:
            return False, None


@dataclass
class MultiLabelThresholds:
    """
    Thresholds for multi-label neutrality.

    Attributes:
        row_gaps: Minimum gap to next best for each label
        thresholds: Per-label thresholds for forcing selection
    """

    row_gaps: Dict[int, float]  # Label -> min gap to second best
    thresholds: Dict[int, np.ndarray]  # Label -> threshold vector
    domain_size: int

    def check_neutrality(self, query: np.ndarray) -> Tuple[bool, Optional[int]]:
        """
        Check if query causes neutrality in multi-label case.

        Args:
            query: Query vector Q

        Returns:
            (is_neutral, forced_label) where forced_label is the winning label or None
        """
        for label in range(self.domain_size):
            # Check if this label dominates
            gaps = query[label] - query  # How much label beats others
            gaps[label] = float("inf")  # Ignore self

            if np.all(gaps >= self.row_gaps[label]):
                return True, label

        return False, None


# ============================================================================
# Jacobian Matrix Structure
# ============================================================================


class JacobianBlock:
    """
    Represents a block in the Jacobian matrix.

    Used for organizing the Jacobian into Q→R and R→Q blocks.

    Attributes:
        block_type: Type identifier ('Q_to_R', 'R_to_Q', etc.)
        size: (rows, cols) dimensions
        data: The actual matrix data
        is_sparse: Whether using sparse representation
    """

    def __init__(self, block_type: str, size: Tuple[int, int], sparse: bool = False):
        self.type = block_type
        self.size = size
        self.is_sparse = sparse

        if sparse and size[0] * size[1] > 1000:
            self.data = lil_matrix(size)
        else:
            self.data = np.zeros(size)

    def set_entry(self, i: int, j: int, value: float):
        """Set entry at position (i, j)"""
        self.data[i, j] = value

    def get_entry(self, i: int, j: int) -> float:
        """Get entry at position (i, j)"""
        if self.is_sparse:
            return self.data[i, j]
        return float(self.data[i, j])

    def to_dense(self) -> np.ndarray:
        """Convert to dense numpy array"""
        if self.is_sparse:
            return self.data.toarray()
        return self.data


class Jacobian:
    """
    Complete Jacobian matrix for one iteration.

    Handles both binary and multi-label cases.

    Attributes:
        iteration: Iteration number
        message_coords: List of all message coordinates
        coord_to_idx: Mapping from coordinate to matrix index
        n: Matrix dimension
        domain_sizes: Domain size for each variable
        matrix: The Jacobian matrix (dense or sparse)
        factor_derivatives: Stored factor step derivatives
    """

    def __init__(
        self,
        message_coords: List[MessageCoordinate],
        iteration: int = 0,
        domain_sizes: Optional[Dict[str, int]] = None,
        factor_graph: Optional[Any] = None,
    ):
        """
        Initialize Jacobian.

        Args:
            message_coords: List of message coordinates
            iteration: Iteration number
            domain_sizes: Map of variable name to domain size
            factor_graph: Optional factor graph reference
        """
        self.iteration = iteration
        self.message_coords = message_coords
        self.coord_to_idx = {coord: i for i, coord in enumerate(message_coords)}
        self.n = len(message_coords)
        self.domain_sizes = domain_sizes or {}
        self.factor_graph = factor_graph

        # Determine if binary
        self.is_binary = (
            all(size == 2 for size in self.domain_sizes.values())
            if self.domain_sizes
            else True
        )

        # Dense for small, sparse for large
        if self.n < 100:
            self.matrix = np.zeros((self.n, self.n))
            self.is_sparse = False
        else:
            self.matrix = lil_matrix((self.n, self.n))
            self.is_sparse = True

        # Track factor derivatives
        self.factor_derivatives: Dict[str, FactorStepDerivative] = {}

        # Build initial structure
        self._build_structure()

    def _build_structure(self):
        """Build the Jacobian structure based on message dependencies"""
        for coord in self.message_coords:
            if coord.msg_type == MessageType.Q_MESSAGE:
                # Q messages depend linearly on R messages
                self._add_variable_dependencies(coord)
            elif coord.msg_type == MessageType.R_MESSAGE:
                # R messages depend nonlinearly on Q messages
                self._add_factor_dependencies(coord)

    def _add_variable_dependencies(self, q_coord: MessageCoordinate):
        """
        Add dependencies for variable update step.
        Variable updates are always linear (derivative = 1).
        """
        i = self.coord_to_idx[q_coord]
        var = q_coord.sender
        target_factor = q_coord.recipient

        # Q_{i->f} = Σ_{g≠f} R_{g->i}
        for coord in self.message_coords:
            if (
                coord.msg_type == MessageType.R_MESSAGE
                and coord.recipient == var
                and coord.sender != target_factor
            ):
                j = self.coord_to_idx[coord]

                if self.is_binary or (coord.label_from == q_coord.label_from):
                    # Binary: scalar addition
                    # Multi-label: only same-label components add
                    self.set_entry(i, j, 1.0)

    def _add_factor_dependencies(self, r_coord: MessageCoordinate):
        """
        Add dependencies for factor update step.
        These are filled in dynamically based on actual derivatives.
        """
        # Will be updated during runtime
        pass

    def set_entry(self, i: int, j: int, value: float):
        """Set Jacobian entry J[i,j]"""
        if abs(value) < 1e-10:
            return  # Skip zeros

        self.matrix[i, j] = value

    def get_entry(self, i: int, j: int) -> float:
        """Get Jacobian entry J[i,j]"""
        return float(self.matrix[i, j])

    def update_factor_derivative(self, factor_name: str, deriv: FactorStepDerivative):
        """
        Update Jacobian with computed factor derivative.

        Args:
            factor_name: Name of the factor
            deriv: Computed derivative object
        """
        self.factor_derivatives[factor_name] = deriv

        # Update matrix entries
        if deriv.is_binary:
            # Binary case: single scalar derivative
            r_coord = MessageCoordinate(
                MessageType.R_MESSAGE, factor_name, deriv.to_var
            )
            q_coord = MessageCoordinate(
                MessageType.Q_MESSAGE, deriv.from_var, factor_name
            )

            if r_coord in self.coord_to_idx and q_coord in self.coord_to_idx:
                i = self.coord_to_idx[r_coord]
                j = self.coord_to_idx[q_coord]
                self.set_entry(i, j, float(deriv.value))
        else:
            # Multi-label case: matrix of derivatives
            for label_i in range(deriv.domain_size):
                for label_j in range(deriv.domain_size):
                    r_coord = MessageCoordinate(
                        MessageType.R_MESSAGE,
                        factor_name,
                        deriv.to_var,
                        label_from=label_i,
                    )
                    q_coord = MessageCoordinate(
                        MessageType.Q_MESSAGE,
                        deriv.from_var,
                        factor_name,
                        label_from=label_j,
                    )

                    if r_coord in self.coord_to_idx and q_coord in self.coord_to_idx:
                        i = self.coord_to_idx[r_coord]
                        j = self.coord_to_idx[q_coord]
                        self.set_entry(i, j, deriv.value[label_i, label_j])

    def to_dense(self) -> np.ndarray:
        """Convert to dense matrix"""
        if self.is_sparse:
            return self.matrix.toarray()
        return self.matrix

    def to_sparse(self):
        """Convert to sparse CSR format"""
        if self.is_sparse:
            return self.matrix.tocsr()
        return csr_matrix(self.matrix)

    def compute_eigenvalues(self) -> np.ndarray:
        """Compute all eigenvalues"""
        return np.linalg.eigvals(self.to_dense())

    def spectral_radius(self) -> float:
        """Compute spectral radius ρ(J) = max|λ|"""
        eigenvals = self.compute_eigenvalues()
        return float(np.max(np.abs(eigenvals)))

    def is_nilpotent(self, tol: float = 1e-10) -> bool:
        """Check if Jacobian is nilpotent (all eigenvalues = 0)"""
        eigenvals = self.compute_eigenvalues()
        return bool(np.all(np.abs(eigenvals) < tol))

    def nilpotent_index(self, max_power: Optional[int] = None) -> Optional[int]:
        """
        Find nilpotent index L such that J^L = 0.

        Args:
            max_power: Maximum power to check (default: matrix dimension)

        Returns:
            L if nilpotent, None otherwise
        """
        if not self.is_nilpotent():
            return None

        J = self.to_dense()
        J_power = J.copy()
        max_power = max_power or self.n

        for L in range(1, max_power + 1):
            if np.allclose(J_power, 0, atol=1e-10):
                return L
            J_power = J_power @ J

        return None


# ============================================================================
# Message Dependency Graph
# ============================================================================


class MessageDependencyGraph:
    """
    Directed graph G(J) representing message dependencies.

    Vertices: Message coordinates
    Edges: Non-zero Jacobian entries

    Attributes:
        jacobian: Reference to Jacobian
        graph: NetworkX directed graph
        node_to_coord: Map from node ID to MessageCoordinate
    """

    def __init__(self, jacobian: Jacobian):
        """
        Build dependency graph from Jacobian.

        Args:
            jacobian: Jacobian matrix object
        """
        self.jacobian = jacobian
        self.graph = nx.DiGraph()
        self.node_to_coord = {}
        self._build_graph()

    def _build_graph(self):
        """Build directed graph from Jacobian matrix"""
        J = self.jacobian.to_dense()
        coords = self.jacobian.message_coords

        # Add nodes
        for i, coord in enumerate(coords):
            self.graph.add_node(i, coord=coord, label=str(coord))
            self.node_to_coord[i] = coord

        # Add edges for non-zero entries
        for i in range(len(coords)):
            for j in range(len(coords)):
                if abs(J[i, j]) > 1e-10:
                    self.graph.add_edge(j, i, weight=J[i, j])

    def find_all_cycles(self) -> List[List[int]]:
        """Find all simple cycles in the graph"""
        return list(nx.simple_cycles(self.graph))

    def compute_cycle_gain(self, cycle: List[int]) -> float:
        """
        Compute gain (product of weights) around a cycle.

        Args:
            cycle: List of node indices forming a cycle

        Returns:
            Product of edge weights around cycle
        """
        gain = 1.0
        for i in range(len(cycle)):
            j = (i + 1) % len(cycle)
            if self.graph.has_edge(cycle[i], cycle[j]):
                gain *= self.graph[cycle[i]][cycle[j]]["weight"]
            else:
                return 0.0  # Broken cycle
        return gain

    def compute_signed_cycle_gain(self, cycle: List[int]) -> int:
        """
        Compute signed cycle gain for binary case.

        Args:
            cycle: List of node indices

        Returns:
            -1, 0, or +1
        """
        gain = self.compute_cycle_gain(cycle)
        if abs(gain) < 1e-10:
            return 0
        return 1 if gain > 0 else -1

    def find_strongly_connected_components(self) -> List[Set[int]]:
        """Find all strongly connected components"""
        return list(nx.strongly_connected_components(self.graph))

    def is_dag(self) -> bool:
        """Check if the graph is a directed acyclic graph"""
        return nx.is_directed_acyclic_graph(self.graph)

    def longest_path_length(self) -> int:
        """
        Find longest path in the graph (for convergence time).

        Returns:
            Length of longest path, or -1 if graph has cycles
        """
        if not self.is_dag():
            return -1
        return nx.dag_longest_path_length(self.graph)

    def find_neutral_edges(self) -> Set[Tuple[int, int]]:
        """Find all edges with zero weight (neutral factor steps)"""
        neutral = set()
        for u, v, data in self.graph.edges(data=True):
            if abs(data.get("weight", 1)) < 1e-10:
                neutral.add((u, v))
        return neutral


# ============================================================================
# Factor Analyzers - Binary and Multi-Label
# ============================================================================


class BinaryFactorAnalyzer:
    """
    Analyzer for binary (K=2) factors.

    Computes thresholds and derivatives for binary factors.
    """

    @staticmethod
    def compute_thresholds(cost_table: np.ndarray) -> BinaryThresholds:
        """
        Compute neutrality thresholds for binary factor.

        Args:
            cost_table: 2x2 cost matrix [[C(0,0), C(0,1)],
                                         [C(1,0), C(1,1)]]

        Returns:
            BinaryThresholds object
        """
        # Row differences
        delta_0 = cost_table[0, 0] - cost_table[0, 1]  # Row x_i=0
        delta_1 = cost_table[1, 0] - cost_table[1, 1]  # Row x_i=1

        # Thresholds (Proposition from paper)
        theta_0 = max(delta_0, delta_1)
        theta_1 = max(-delta_0, -delta_1)

        return BinaryThresholds(theta_0, theta_1)

    @staticmethod
    def compute_derivative(
        cost_table: np.ndarray,
        delta_q: float,
        thresholds: Optional[BinaryThresholds] = None,
    ) -> int:
        """
        Compute binary factor step derivative.

        Args:
            cost_table: 2x2 cost matrix
            delta_q: Query difference ΔQ = Q(1) - Q(0)
            thresholds: Pre-computed thresholds (optional)

        Returns:
            Derivative in {-1, 0, +1}
        """
        if thresholds is None:
            thresholds = BinaryFactorAnalyzer.compute_thresholds(cost_table)

        is_neutral, forced_val = thresholds.check_neutrality(delta_q)

        if is_neutral:
            return 0

        # Determine column switch
        row_0_costs = cost_table[0, :] + np.array([0, delta_q])
        row_1_costs = cost_table[1, :] + np.array([0, delta_q])

        x_j_star_0 = np.argmin(row_0_costs)
        x_j_star_1 = np.argmin(row_1_costs)

        # Derivative = 𝟙(x*_j(1)=1) - 𝟙(x*_j(0)=1)
        return int(x_j_star_1 == 1) - int(x_j_star_0 == 1)

    @staticmethod
    def compute_margins(
        cost_table: np.ndarray,
        delta_q: float,
        thresholds: Optional[BinaryThresholds] = None,
    ) -> Dict[str, float]:
        """
        Compute neutrality margins.

        Args:
            cost_table: 2x2 cost matrix
            delta_q: Query difference
            thresholds: Pre-computed thresholds

        Returns:
            Dictionary with 'positive_margin' and 'negative_margin'
        """
        if thresholds is None:
            thresholds = BinaryFactorAnalyzer.compute_thresholds(cost_table)

        return {
            "positive_margin": delta_q - thresholds.theta_0,
            "negative_margin": -thresholds.theta_1 - delta_q,
        }


class MultiLabelFactorAnalyzer:
    """
    Analyzer for multi-label (K>2) factors.

    Handles general domain sizes with matrix derivatives.
    """

    @staticmethod
    def compute_thresholds(cost_table: np.ndarray) -> MultiLabelThresholds:
        """
        Compute neutrality thresholds for multi-label factor.

        Args:
            cost_table: KxK cost matrix

        Returns:
            MultiLabelThresholds object
        """
        K = cost_table.shape[1]
        row_gaps = {}
        thresholds = {}

        for label in range(K):
            # For each label, find minimum gap to force selection
            gaps_per_row = []

            for row in range(K):
                # Cost differences: C(row, other) - C(row, label)
                costs = cost_table[row, :]
                label_cost = costs[label]
                other_costs = np.delete(costs, label)

                if len(other_costs) > 0:
                    min_gap = np.min(other_costs) - label_cost
                    gaps_per_row.append(min_gap)

            row_gaps[label] = max(gaps_per_row) if gaps_per_row else float("inf")
            thresholds[label] = np.array([row_gaps[label]] * K)

        return MultiLabelThresholds(row_gaps, thresholds, K)

    @staticmethod
    def compute_derivative(
        cost_table: np.ndarray,
        query: np.ndarray,
        thresholds: Optional[MultiLabelThresholds] = None,
    ) -> np.ndarray:
        """
        Compute multi-label factor derivative matrix.

        Args:
            cost_table: KxK cost matrix
            query: Query vector Q of size K
            thresholds: Pre-computed thresholds

        Returns:
            KxK derivative matrix where entry (i,j) = ∂R_i/∂Q_j
        """
        K = cost_table.shape[0]
        derivative = np.zeros((K, K))

        # For each row, find winning column
        for row in range(K):
            costs_with_query = cost_table[row, :] + query
            winner = np.argmin(costs_with_query)

            # Derivative is 1 for winner, 0 elsewhere
            derivative[row, winner] = 1.0

        return derivative

    @staticmethod
    def is_neutral(derivative: np.ndarray) -> bool:
        """
        Check if multi-label derivative represents neutrality.

        Args:
            derivative: KxK derivative matrix

        Returns:
            True if same column selected for all rows
        """
        # Get selected column for each row
        selected = np.argmax(derivative, axis=1)
        return len(np.unique(selected)) == 1

    @staticmethod
    def compute_margins(
        cost_table: np.ndarray,
        query: np.ndarray,
        thresholds: Optional[MultiLabelThresholds] = None,
    ) -> np.ndarray:
        """
        Compute neutrality margins for each label.

        Args:
            cost_table: KxK cost matrix
            query: Query vector
            thresholds: Pre-computed thresholds

        Returns:
            Array of margins for each label
        """
        if thresholds is None:
            thresholds = MultiLabelFactorAnalyzer.compute_thresholds(cost_table)

        K = len(query)
        margins = np.zeros(K)

        for label in range(K):
            # How much does this label beat others?
            gaps = query[label] - query
            gaps[label] = float("inf")

            # Margin is minimum gap minus required threshold
            min_gap = np.min(gaps)
            margins[label] = min_gap - thresholds.row_gaps[label]

        return margins


# ============================================================================
# Cycle Gain and Neutral Cover Analysis
# ============================================================================


class CycleGainAnalyzer:
    """
    Analyzes cycle gains and finds neutral covers.

    Implements hitting set algorithms to break all cycles.
    """

    def __init__(self):
        self.cycle_gains: Dict[Tuple, float] = {}
        self.analysis_cache = {}

    def analyze_all_cycles(self, dep_graph: MessageDependencyGraph) -> Dict:
        """
        Comprehensive cycle analysis.

        Args:
            dep_graph: Message dependency graph

        Returns:
            Dictionary with cycle statistics and gains
        """
        cycles = dep_graph.find_all_cycles()

        results = {
            "num_cycles": len(cycles),
            "cycles": [],
            "has_zero_gain_cycle": False,
            "all_zero_gain": False,
            "max_gain": 0.0,
            "min_gain": float("inf") if cycles else 0.0,
            "cycle_length_distribution": {},
        }

        for cycle in cycles:
            gain = dep_graph.compute_cycle_gain(cycle)
            signed_gain = dep_graph.compute_signed_cycle_gain(cycle)

            cycle_info = {
                "nodes": cycle,
                "length": len(cycle),
                "gain": gain,
                "signed_gain": signed_gain,
                "is_zero": abs(gain) < 1e-10,
                "coords": [dep_graph.node_to_coord[n] for n in cycle],
            }
            results["cycles"].append(cycle_info)

            # Update statistics
            if cycle_info["is_zero"]:
                results["has_zero_gain_cycle"] = True

            results["max_gain"] = max(results["max_gain"], abs(gain))
            results["min_gain"] = min(results["min_gain"], abs(gain))

            # Track length distribution
            length = len(cycle)
            results["cycle_length_distribution"][length] = (
                results["cycle_length_distribution"].get(length, 0) + 1
            )

        # Check if all cycles have zero gain
        if cycles:
            results["all_zero_gain"] = all(c["is_zero"] for c in results["cycles"])

        return results

    def find_neutral_cover(
        self,
        jacobian: Jacobian,
        dep_graph: MessageDependencyGraph,
        strategy: str = "greedy",
    ) -> Set[MessageCoordinate]:
        """
        Find a neutral cover that hits all cycles.

        Args:
            jacobian: Jacobian matrix
            dep_graph: Dependency graph
            strategy: 'greedy', 'scc', or 'optimal'

        Returns:
            Set of MessageCoordinates forming neutral cover
        """
        if strategy == "greedy":
            return self._greedy_neutral_cover(jacobian, dep_graph)
        elif strategy == "scc":
            return self._scc_based_cover(jacobian, dep_graph)
        else:
            return self._greedy_neutral_cover(jacobian, dep_graph)

    def _greedy_neutral_cover(
        self, jacobian: Jacobian, dep_graph: MessageDependencyGraph
    ) -> Set[MessageCoordinate]:
        """
        Greedy hitting set algorithm for neutral cover.

        Iteratively selects neutral edges that cover most uncovered cycles.
        """
        cycles = dep_graph.find_all_cycles()
        if not cycles:
            return set()

        # Find all neutral edges
        neutral_edges = dep_graph.find_neutral_edges()

        # If no neutral edges available, find potential neutral factor steps
        if not neutral_edges:
            neutral_edges = self._find_neutralizable_edges(jacobian, dep_graph)

        # Greedy hitting set
        cover = set()
        cover_edges = set()
        uncovered_cycles = cycles.copy()

        while uncovered_cycles:
            best_edge = None
            best_coverage = 0

            # Find edge that covers most cycles
            for edge in neutral_edges - cover_edges:
                coverage = sum(
                    1 for cycle in uncovered_cycles if self._edge_in_cycle(edge, cycle)
                )
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_edge = edge

            if best_edge is None:
                logger.warning(
                    f"Cannot cover all cycles with neutral edges. "
                    f"{len(uncovered_cycles)} cycles remain."
                )
                break

            cover_edges.add(best_edge)
            # Convert edge to MessageCoordinate
            u, v = best_edge
            cover.add(dep_graph.node_to_coord[v])

            # Remove covered cycles
            uncovered_cycles = [
                c for c in uncovered_cycles if not self._edge_in_cycle(best_edge, c)
            ]

        return cover

    def _scc_based_cover(
        self, jacobian: Jacobian, dep_graph: MessageDependencyGraph
    ) -> Set[MessageCoordinate]:
        """
        SCC-based algorithm from the paper.

        Iteratively breaks SCCs by making edges neutral.
        """
        cover = set()
        graph = dep_graph.graph.copy()

        while True:
            # Find SCCs
            sccs = list(nx.strongly_connected_components(graph))

            # Find largest SCC with >1 node
            largest_scc = None
            for scc in sccs:
                if len(scc) > 1:
                    if largest_scc is None or len(scc) > len(largest_scc):
                        largest_scc = scc

            if largest_scc is None:
                break  # Graph is a DAG

            # Find best edge to neutralize in this SCC
            best_edge = self._find_best_edge_in_scc(
                graph, largest_scc, jacobian, dep_graph
            )

            if best_edge:
                u, v = best_edge
                cover.add(dep_graph.node_to_coord[v])
                graph.remove_edge(u, v)
            else:
                break

        return cover

    def _edge_in_cycle(self, edge: Tuple[int, int], cycle: List[int]) -> bool:
        """Check if edge appears in cycle"""
        u, v = edge
        for i in range(len(cycle)):
            j = (i + 1) % len(cycle)
            if cycle[i] == u and cycle[j] == v:
                return True
        return False

    def _find_neutralizable_edges(
        self, jacobian: Jacobian, dep_graph: MessageDependencyGraph
    ) -> Set[Tuple[int, int]]:
        """Find edges that could potentially be made neutral"""
        neutralizable = set()

        for u, v, data in dep_graph.graph.edges(data=True):
            coord = dep_graph.node_to_coord[v]

            # Check if this is a factor step
            if coord.msg_type == MessageType.R_MESSAGE:
                # Could potentially be made neutral through SCFG or other means
                neutralizable.add((u, v))

        return neutralizable

    def _find_best_edge_in_scc(
        self,
        graph: nx.DiGraph,
        scc: Set[int],
        jacobian: Jacobian,
        dep_graph: MessageDependencyGraph,
    ) -> Optional[Tuple[int, int]]:
        """Find best edge to remove in an SCC"""
        best_edge = None
        best_score = float("-inf")

        # Consider edges within the SCC
        for u in scc:
            for v in graph.successors(u):
                if v in scc:
                    # Score based on how many cycles this edge participates in
                    score = self._edge_cycle_score(
                        (u, v), scc, graph, jacobian, dep_graph
                    )
                    if score > best_score:
                        best_score = score
                        best_edge = (u, v)

        return best_edge

    def _edge_cycle_score(
        self,
        edge: Tuple[int, int],
        scc: Set[int],
        graph: nx.DiGraph,
        jacobian: Jacobian,
        dep_graph: MessageDependencyGraph,
    ) -> float:
        """Score an edge based on its importance in breaking cycles"""
        u, v = edge
        score = 0.0

        # Count cycles containing this edge
        subgraph = graph.subgraph(scc)
        cycles_in_scc = list(nx.simple_cycles(subgraph))

        for cycle in cycles_in_scc:
            if self._edge_in_cycle(edge, cycle):
                score += 1.0

        # Bonus if edge corresponds to a factor that's easy to neutralize
        coord = dep_graph.node_to_coord[v]
        if coord.msg_type == MessageType.R_MESSAGE:
            score += 0.5  # Prefer factor edges

        return score


# ============================================================================
# SCFG (Split Constraint-Function Graph) Support
# ============================================================================


@dataclass
class SplitFactorInfo:
    """
    Information about a split factor.

    Attributes:
        original_name: Name of original factor
        left_name: Name of left half
        right_name: Name of right half
        split_ratio: Split ratio q ∈ (0,1)
        left_threshold_scale: Scaling factor for left thresholds
        right_threshold_scale: Scaling factor for right thresholds
    """

    original_name: str
    left_name: str
    right_name: str
    split_ratio: float
    left_threshold_scale: float
    right_threshold_scale: float

    @property
    def left_weight(self) -> float:
        return self.split_ratio

    @property
    def right_weight(self) -> float:
        return 1.0 - self.split_ratio


class SCFGAnalyzer:
    """
    Analyzer for SCFG (Split Constraint-Function Graph) configurations.

    Handles splitting strategies and neutrality enforcement.
    """

    def __init__(self, split_ratio: float = 0.5):
        """
        Initialize SCFG analyzer.

        Args:
            split_ratio: Default split ratio q
        """
        self.split_ratio = split_ratio
        self.split_factors: Dict[str, SplitFactorInfo] = {}

    def split_factor(
        self,
        factor_name: str,
        cost_table: np.ndarray,
        split_ratio: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a factor into left and right halves.

        Args:
            factor_name: Name of factor to split
            cost_table: Original cost table
            split_ratio: Split ratio (uses default if None)

        Returns:
            (left_table, right_table) tuple
        """
        q = split_ratio or self.split_ratio

        # Split cost tables
        left_table = q * cost_table
        right_table = (1 - q) * cost_table

        # Store split information
        split_info = SplitFactorInfo(
            original_name=factor_name,
            left_name=f"{factor_name}_L",
            right_name=f"{factor_name}_R",
            split_ratio=q,
            left_threshold_scale=q,
            right_threshold_scale=1 - q,
        )

        self.split_factors[factor_name] = split_info

        return left_table, right_table

    def compute_split_thresholds_binary(
        self, factor_name: str, original_thresholds: BinaryThresholds
    ) -> Tuple[BinaryThresholds, BinaryThresholds]:
        """
        Compute scaled thresholds for split binary factor.

        Args:
            factor_name: Name of split factor
            original_thresholds: Original factor thresholds

        Returns:
            (left_thresholds, right_thresholds) tuple
        """
        if factor_name not in self.split_factors:
            raise ValueError(f"Factor {factor_name} not split")

        info = self.split_factors[factor_name]

        left_thresholds = BinaryThresholds(
            theta_0=info.left_threshold_scale * original_thresholds.theta_0,
            theta_1=info.left_threshold_scale * original_thresholds.theta_1,
        )

        right_thresholds = BinaryThresholds(
            theta_0=info.right_threshold_scale * original_thresholds.theta_0,
            theta_1=info.right_threshold_scale * original_thresholds.theta_1,
        )

        return left_thresholds, right_thresholds

    def compute_split_thresholds_multi(
        self, factor_name: str, original_thresholds: MultiLabelThresholds
    ) -> Tuple[MultiLabelThresholds, MultiLabelThresholds]:
        """
        Compute scaled thresholds for split multi-label factor.

        Args:
            factor_name: Name of split factor
            original_thresholds: Original factor thresholds

        Returns:
            (left_thresholds, right_thresholds) tuple
        """
        if factor_name not in self.split_factors:
            raise ValueError(f"Factor {factor_name} not split")

        info = self.split_factors[factor_name]
        K = original_thresholds.domain_size

        # Scale row gaps
        left_gaps = {
            k: info.left_threshold_scale * v
            for k, v in original_thresholds.row_gaps.items()
        }
        right_gaps = {
            k: info.right_threshold_scale * v
            for k, v in original_thresholds.row_gaps.items()
        }

        # Scale threshold vectors
        left_thresh = {
            k: info.left_threshold_scale * v
            for k, v in original_thresholds.thresholds.items()
        }
        right_thresh = {
            k: info.right_threshold_scale * v
            for k, v in original_thresholds.thresholds.items()
        }

        return (
            MultiLabelThresholds(left_gaps, left_thresh, K),
            MultiLabelThresholds(right_gaps, right_thresh, K),
        )

    def check_sibling_reinforcement(
        self, delta_r_sibling: float, threshold: float, kappa: float = 0.9
    ) -> bool:
        """
        Check if sibling reinforcement ensures neutrality.

        From Lemma in paper: (1-κ)|ΔR_sibling| ≥ threshold

        Args:
            delta_r_sibling: Sibling's response magnitude
            threshold: Required threshold for neutrality
            kappa: Antagonism bound parameter

        Returns:
            True if reinforcement sufficient for neutrality
        """
        return (1 - kappa) * abs(delta_r_sibling) >= threshold

    def optimize_split_ratio(
        self,
        cost_table: np.ndarray,
        sibling_response: float,
        target_margin: float = 0.1,
    ) -> float:
        """
        Optimize split ratio for maximum neutrality margin.

        Args:
            cost_table: Original cost table
            sibling_response: Expected sibling response magnitude
            target_margin: Desired neutrality margin

        Returns:
            Optimal split ratio q
        """
        if cost_table.shape == (2, 2):
            # Binary optimization
            thresholds = BinaryFactorAnalyzer.compute_thresholds(cost_table)

            # Find q that maximizes margin
            best_q = 0.5
            best_margin = 0.0

            for q in np.linspace(0.1, 0.9, 20):
                # Left half thresholds
                left_theta = q * max(thresholds.theta_0, thresholds.theta_1)

                # Expected margin with sibling reinforcement
                margin = abs(sibling_response) - left_theta

                if margin > best_margin:
                    best_margin = margin
                    best_q = q

            return best_q
        else:
            # Multi-label: use default for now
            return self.split_ratio


# ============================================================================
# Main Jacobian Tracker
# ============================================================================


class JacobianTracker:
    """
    Main class for tracking and analyzing Jacobians across iterations.

    This is the primary interface for Jacobian analysis.

    Attributes:
        factor_graph: Reference to factor graph
        jacobians: Dictionary of iteration -> Jacobian
        message_coords: List of all message coordinates
        cycle_analyzer: Cycle gain analyzer
        scfg_analyzer: Optional SCFG analyzer
        domain_sizes: Domain sizes for variables
        is_binary: Whether all variables are binary
    """

    def __init__(
        self, factor_graph: Any, domain_sizes: Optional[Dict[str, int]] = None
    ):
        """
        Initialize Jacobian tracker.

        Args:
            factor_graph: Factor graph object
            domain_sizes: Optional domain sizes for variables
        """
        self.factor_graph = factor_graph
        self.jacobians: Dict[int, Jacobian] = {}
        self.domain_sizes = domain_sizes or self._extract_domain_sizes()
        self.is_binary = all(size == 2 for size in self.domain_sizes.values())

        self.message_coords = self._extract_message_coordinates()
        self.cycle_analyzer = CycleGainAnalyzer()
        self.scfg_analyzer: Optional[SCFGAnalyzer] = None

        # Cache for analysis results
        self.analysis_cache = {}

        logger.info(
            f"Initialized JacobianTracker: "
            f"{'Binary' if self.is_binary else 'Multi-label'} domain, "
            f"{len(self.message_coords)} message coordinates"
        )

    def _extract_domain_sizes(self) -> Dict[str, int]:
        """Extract domain sizes from factor graph"""
        sizes = {}

        # Get from variables
        if hasattr(self.factor_graph, "variables"):
            for var_name, var in self.factor_graph.variables.items():
                if hasattr(var, "domain"):
                    sizes[var_name] = var.domain

        # Default to binary if not found
        if not sizes:
            for node in self.factor_graph.G.nodes():
                if self.factor_graph.G.nodes[node].get("bipartite") == 0:
                    sizes[node] = 2

        return sizes

    def _extract_message_coordinates(self) -> List[MessageCoordinate]:
        """Extract all message coordinates from factor graph"""
        coords = []

        for edge in self.factor_graph.G.edges():
            node1, node2 = edge
            node1_data = self.factor_graph.G.nodes[node1]
            node2_data = self.factor_graph.G.nodes[node2]

            # Determine types
            if node1_data.get("bipartite") == 0:  # node1 is variable
                var_name = node1
                factor_name = node2
            else:  # node1 is factor
                var_name = node2
                factor_name = node1

            domain_size = self.domain_sizes.get(var_name, 2)

            if self.is_binary or domain_size == 2:
                # Binary: single difference coordinate per message
                coords.append(
                    MessageCoordinate(
                        msg_type=MessageType.Q_MESSAGE,
                        sender=var_name,
                        recipient=factor_name,
                    )
                )
                coords.append(
                    MessageCoordinate(
                        msg_type=MessageType.R_MESSAGE,
                        sender=factor_name,
                        recipient=var_name,
                    )
                )
            else:
                # Multi-label: coordinate per label pair
                for label in range(domain_size):
                    coords.append(
                        MessageCoordinate(
                            msg_type=MessageType.Q_MESSAGE,
                            sender=var_name,
                            recipient=factor_name,
                            label_from=label,
                        )
                    )
                    coords.append(
                        MessageCoordinate(
                            msg_type=MessageType.R_MESSAGE,
                            sender=factor_name,
                            recipient=var_name,
                            label_from=label,
                        )
                    )

        return coords

    def create_jacobian(self, iteration: int) -> Jacobian:
        """
        Create a new Jacobian for the given iteration.

        Args:
            iteration: Iteration number

        Returns:
            Created Jacobian object
        """
        jac = Jacobian(
            self.message_coords, iteration, self.domain_sizes, self.factor_graph
        )
        self.jacobians[iteration] = jac

        # Clear cache for this iteration
        if iteration in self.analysis_cache:
            del self.analysis_cache[iteration]

        return jac

    def update_factor_derivative_binary(
        self,
        iteration: int,
        factor_name: str,
        from_var: str,
        to_var: str,
        derivative_value: int,
    ):
        """
        Update binary factor derivative.

        Args:
            iteration: Iteration number
            factor_name: Name of factor
            from_var: Variable sending Q message
            to_var: Variable receiving R message
            derivative_value: Derivative in {-1, 0, +1}
        """
        if iteration not in self.jacobians:
            self.create_jacobian(iteration)

        jac = self.jacobians[iteration]

        deriv = FactorStepDerivative(
            factor=factor_name,
            from_var=from_var,
            to_var=to_var,
            value=derivative_value,
            domain_size=2,
            iteration=iteration,
            is_binary=True,
        )

        jac.update_factor_derivative(factor_name, deriv)

    def update_factor_derivative_multi(
        self,
        iteration: int,
        factor_name: str,
        from_var: str,
        to_var: str,
        derivative_matrix: np.ndarray,
    ):
        """
        Update multi-label factor derivative.

        Args:
            iteration: Iteration number
            factor_name: Name of factor
            from_var: Variable sending Q message
            to_var: Variable receiving R message
            derivative_matrix: KxK derivative matrix
        """
        if iteration not in self.jacobians:
            self.create_jacobian(iteration)

        jac = self.jacobians[iteration]
        K = derivative_matrix.shape[0]

        deriv = FactorStepDerivative(
            factor=factor_name,
            from_var=from_var,
            to_var=to_var,
            value=derivative_matrix,
            domain_size=K,
            iteration=iteration,
            is_binary=False,
        )

        jac.update_factor_derivative(factor_name, deriv)

    def analyze_iteration(self, iteration: int) -> Dict:
        """
        Comprehensive analysis of a single iteration.

        Args:
            iteration: Iteration number

        Returns:
            Dictionary with analysis results
        """
        # Check cache
        if iteration in self.analysis_cache:
            return self.analysis_cache[iteration]

        if iteration not in self.jacobians:
            return {"error": f"No Jacobian for iteration {iteration}"}

        jac = self.jacobians[iteration]
        dep_graph = MessageDependencyGraph(jac)

        results = {
            "iteration": iteration,
            "is_binary": jac.is_binary,
            "matrix_size": jac.n,
            "spectral_radius": jac.spectral_radius(),
            "is_nilpotent": jac.is_nilpotent(),
            "nilpotent_index": jac.nilpotent_index(),
            "is_dag": dep_graph.is_dag(),
            "longest_path": dep_graph.longest_path_length()
            if dep_graph.is_dag()
            else -1,
            "cycle_analysis": self.cycle_analyzer.analyze_all_cycles(dep_graph),
            "num_sccs": len(dep_graph.find_strongly_connected_components()),
            "neutral_cover": list(
                self.cycle_analyzer.find_neutral_cover(jac, dep_graph)
            ),
            "num_neutral_edges": len(dep_graph.find_neutral_edges()),
        }

        # Cache results
        self.analysis_cache[iteration] = results

        return results

    def analyze_convergence(
        self, start_iter: int = 0, end_iter: Optional[int] = None
    ) -> Dict:
        """
        Analyze convergence properties across iterations.

        Args:
            start_iter: Starting iteration
            end_iter: Ending iteration (uses max available if None)

        Returns:
            Dictionary with convergence analysis
        """
        if end_iter is None:
            end_iter = max(self.jacobians.keys()) if self.jacobians else 0

        results = {
            "iterations_analyzed": list(range(start_iter, end_iter + 1)),
            "spectral_radii": [],
            "nilpotent_iterations": [],
            "dag_iterations": [],
            "convergence_detected": False,
            "convergence_iteration": None,
            "max_convergence_time": 0,
            "eigenvalue_trajectory": [],
        }

        for i in range(start_iter, end_iter + 1):
            if i in self.jacobians:
                analysis = self.analyze_iteration(i)

                # Track spectral radius
                results["spectral_radii"].append((i, analysis["spectral_radius"]))

                # Track eigenvalues
                eigenvals = self.jacobians[i].compute_eigenvalues()
                results["eigenvalue_trajectory"].append(
                    {
                        "iteration": i,
                        "eigenvalues": eigenvals.tolist(),
                        "max_real": float(np.max(np.real(eigenvals))),
                        "max_imag": float(np.max(np.abs(np.imag(eigenvals)))),
                    }
                )

                # Check nilpotency
                if analysis["is_nilpotent"]:
                    results["nilpotent_iterations"].append(i)

                # Check DAG
                if analysis["is_dag"]:
                    results["dag_iterations"].append(i)
                    results["max_convergence_time"] = max(
                        results["max_convergence_time"], analysis["longest_path"]
                    )

                    if not results["convergence_detected"]:
                        results["convergence_detected"] = True
                        results["convergence_iteration"] = i

        return results

    def get_convergence_summary(self) -> str:
        """
        Get a human-readable convergence summary.

        Returns:
            Summary string
        """
        analysis = self.analyze_convergence()

        summary = []
        summary.append("=" * 60)
        summary.append("CONVERGENCE ANALYSIS SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Domain type: {'Binary' if self.is_binary else 'Multi-label'}")
        summary.append(f"Message coordinates: {len(self.message_coords)}")
        summary.append(f"Iterations analyzed: {len(analysis['iterations_analyzed'])}")

        if analysis["convergence_detected"]:
            summary.append(
                f"✓ Convergence detected at iteration {analysis['convergence_iteration']}"
            )
            summary.append(
                f"  Maximum convergence time: {analysis['max_convergence_time']} sweeps"
            )
        else:
            summary.append("✗ No convergence detected")

        summary.append(f"Nilpotent iterations: {analysis['nilpotent_iterations']}")
        summary.append(f"DAG iterations: {analysis['dag_iterations']}")

        if analysis["spectral_radii"]:
            radii = [r[1] for r in analysis["spectral_radii"]]
            summary.append(
                f"Spectral radius range: [{min(radii):.4f}, {max(radii):.4f}]"
            )

        summary.append("=" * 60)

        return "\n".join(summary)
