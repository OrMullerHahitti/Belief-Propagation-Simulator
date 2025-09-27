"""Convergence Detection for Belief Propagation Algorithms.

This module provides tools to monitor the state of a belief propagation
simulation and determine if it has converged. Convergence is typically
assessed by checking if the beliefs and variable assignments have stabilized
over a certain number of iterations.
"""
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

from ..configs.global_config_mapping import CONVERGENCE_DEFAULTS

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceConfig:
    """Configuration settings for the convergence detection logic.

    Attributes:
        belief_threshold: The maximum change in the norm of a belief vector
            for it to be considered stable.
        assignment_threshold: The maximum number of assignment changes allowed
            for the assignments to be considered stable.
        min_iterations: The minimum number of iterations to run before
            checking for convergence.
        patience: The number of consecutive iterations for which the beliefs
            and assignments must remain stable before declaring convergence.
        use_relative_change: If True, uses the relative change in belief norm
            for the threshold check; otherwise, uses the absolute change.
    """
    belief_threshold: float = CONVERGENCE_DEFAULTS["belief_threshold"]
    assignment_threshold: int = CONVERGENCE_DEFAULTS["assignment_threshold"]
    min_iterations: int = CONVERGENCE_DEFAULTS["min_iterations"]
    patience: int = CONVERGENCE_DEFAULTS["patience"]
    use_relative_change: bool = CONVERGENCE_DEFAULTS["use_relative_change"]


class ConvergenceMonitor:
    """Monitors and detects convergence in a belief propagation simulation.

    This class tracks the history of beliefs and assignments to determine if
    the algorithm has reached a stable state according to the provided
    configuration.
    """

    def __init__(self, config: Optional[ConvergenceConfig] = None):
        """Initializes the ConvergenceMonitor.

        Args:
            config: A `ConvergenceConfig` object containing the parameters
                for convergence detection. If None, default settings are used.
        """
        self.config = config or ConvergenceConfig()
        self.prev_beliefs: Optional[Dict[str, np.ndarray]] = None
        self.prev_assignments: Optional[Dict[str, int]] = None
        self.stable_count = 0
        self.iteration = 0
        self.convergence_history = []

    def check_convergence(
        self, beliefs: Dict[str, np.ndarray], assignments: Dict[str, int]
    ) -> bool:
        """Checks if the algorithm has converged.

        This method compares the current beliefs and assignments with the
        previous state to determine if they have stabilized.

        Args:
            beliefs: A dictionary mapping variable names to their current belief vectors.
            assignments: A dictionary mapping variable names to their current assignments.

        Returns:
            True if the algorithm is considered to have converged, False otherwise.
        """
        self.iteration += 1
        if self.iteration < self.config.min_iterations:
            self._update_state(beliefs, assignments)
            return False

        if self.prev_beliefs is None:
            self._update_state(beliefs, assignments)
            return False

        belief_changes = []
        for var in beliefs:
            if var in self.prev_beliefs:
                if self.config.use_relative_change:
                    prev_norm = np.linalg.norm(self.prev_beliefs[var])
                    change = (
                        np.linalg.norm(beliefs[var] - self.prev_beliefs[var]) / prev_norm
                        if prev_norm > 0 else np.linalg.norm(beliefs[var])
                    )
                else:
                    change = np.linalg.norm(beliefs[var] - self.prev_beliefs[var])
                belief_changes.append(change)

        max_belief_change = max(belief_changes) if belief_changes else 0
        belief_converged = max_belief_change < self.config.belief_threshold
        assignment_converged = all(
            assignments.get(var) == self.prev_assignments.get(var) for var in assignments
        )

        logger.debug(
            f"Iteration {self.iteration}: max_belief_change={max_belief_change:.6f}, "
            f"belief_converged={belief_converged}, assignment_converged={assignment_converged}"
        )
        self.convergence_history.append({
            "iteration": self.iteration,
            "max_belief_change": max_belief_change,
            "belief_converged": belief_converged,
            "assignment_converged": assignment_converged,
        })

        self._update_state(beliefs, assignments)

        if belief_converged and assignment_converged:
            self.stable_count += 1
            if self.stable_count >= self.config.patience:
                logger.info(f"Converged after {self.iteration} iterations")
                return True
        else:
            self.stable_count = 0

        return False

    def _update_state(
        self, beliefs: Dict[str, np.ndarray], assignments: Dict[str, int]
    ) -> None:
        """Updates the internal state with the current beliefs and assignments."""
        self.prev_beliefs = {k: v.copy() for k, v in beliefs.items()}
        self.prev_assignments = assignments.copy()

    def reset(self) -> None:
        """Resets the monitor to its initial state for a new simulation run."""
        self.prev_beliefs = None
        self.prev_assignments = None
        self.stable_count = 0
        self.iteration = 0
        self.convergence_history.clear()
        logger.debug("Convergence monitor reset")

    def get_convergence_summary(self) -> Dict:
        """Returns a summary of the convergence history.

        Returns:
            A dictionary containing the total iterations, convergence status,
            and the history of belief changes.
        """
        if not self.convergence_history:
            return {}

        return {
            "total_iterations": self.iteration,
            "converged": self.stable_count >= self.config.patience,
            "final_max_belief_change": self.convergence_history[-1]["max_belief_change"],
            "history": self.convergence_history,
        }
