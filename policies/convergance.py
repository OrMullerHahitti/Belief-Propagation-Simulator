import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection."""

    belief_threshold: float = 1e-6
    assignment_threshold: int = 0  # For discrete assignments
    min_iterations: int = 10
    patience: int = 5  # How many iterations to wait after apparent convergence
    use_relative_change: bool = False  # Use relative vs absolute change


class ConvergenceMonitor:
    """Monitor and detect convergence in belief propagation."""

    def __init__(self, config: Optional[ConvergenceConfig] = None):
        self.config = config or ConvergenceConfig()
        self.prev_beliefs: Optional[Dict[str, np.ndarray]] = None
        self.prev_assignments: Optional[Dict[str, int]] = None
        self.stable_count = 0
        self.iteration = 0
        self.convergence_history = []

    def check_convergence(
        self, beliefs: Dict[str, np.ndarray], assignments: Dict[str, int]
    ) -> bool:
        """Check if algorithm has converged based on beliefs and assignments."""
        self.iteration += 1

        # Don't converge too early
        if self.iteration < self.config.min_iterations:
            self._update_state(beliefs, assignments)
            return False

        # First iteration after min_iterations
        if self.prev_beliefs is None:
            self._update_state(beliefs, assignments)
            return False

        # Check belief convergence
        belief_changes = []
        for var in beliefs:
            if var in self.prev_beliefs:
                if self.config.use_relative_change:
                    # Relative change
                    prev_norm = np.linalg.norm(self.prev_beliefs[var])
                    if prev_norm > 0:
                        change = (
                            np.linalg.norm(beliefs[var] - self.prev_beliefs[var])
                            / prev_norm
                        )
                    else:
                        change = np.linalg.norm(beliefs[var])
                else:
                    # Absolute change
                    change = np.linalg.norm(beliefs[var] - self.prev_beliefs[var])
                belief_changes.append(change)

        max_belief_change = max(belief_changes) if belief_changes else 0
        belief_converged = max_belief_change < self.config.belief_threshold

        # Check assignment convergence
        assignment_converged = all(
            assignments.get(var) == self.prev_assignments.get(var)
            for var in assignments
        )

        # Log convergence status
        logger.debug(
            f"Iteration {self.iteration}: max_belief_change={max_belief_change:.6f}, "
            f"belief_converged={belief_converged}, assignment_converged={assignment_converged}"
        )

        # Store convergence info
        self.convergence_history.append(
            {
                "iteration": self.iteration,
                "max_belief_change": max_belief_change,
                "belief_converged": belief_converged,
                "assignment_converged": assignment_converged,
            }
        )

        # Update state
        self._update_state(beliefs, assignments)

        # Check if both converged
        if belief_converged and assignment_converged:
            self.stable_count += 1
            converged = self.stable_count >= self.config.patience
            if converged:
                logger.info(f"Converged after {self.iteration} iterations")
            return converged
        else:
            self.stable_count = 0
            return False

    def _update_state(
        self, beliefs: Dict[str, np.ndarray], assignments: Dict[str, int]
    ):
        """Update internal state with current beliefs and assignments."""
        self.prev_beliefs = {k: v.copy() for k, v in beliefs.items()}
        self.prev_assignments = assignments.copy()

    def reset(self):
        """Reset monitor for new run."""
        self.prev_beliefs = None
        self.prev_assignments = None
        self.stable_count = 0
        self.iteration = 0
        self.convergence_history.clear()
        logger.debug("Convergence monitor reset")

    def get_convergence_summary(self) -> Dict:
        """Get summary of convergence history."""
        if not self.convergence_history:
            return {}

        return {
            "total_iterations": self.iteration,
            "converged": self.stable_count >= self.config.patience,
            "final_max_belief_change": self.convergence_history[-1][
                "max_belief_change"
            ],
            "history": self.convergence_history,
        }
