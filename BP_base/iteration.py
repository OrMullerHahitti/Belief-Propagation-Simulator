# iterations.py
from pydantic import BaseModel, Field
from typing import Dict, Optional
import numpy as np
from datetime import datetime


class Iteration:
    """
    Represents a single iteration in the Belief Propagation algorithm.
    Contains all relevant information about the iteration state.
    """
    def __init__(self):
        number: int = Field(ge=0)  # Current iteration number (0-based)
        max_iterations: int = Field(ge=1)  # Maximum number of iterations

        # Messages state
        Q_messages: Optional[Dict] = None  # Current Q messages
        R_messages: Optional[Dict] = None  # Current R messages
        Q_previous: Optional[Dict] = None  # Previous Q messages
        R_previous: Optional[Dict] = None  # Previous R messages

        # Convergence metrics
        message_residual: Optional[float] = None  # Difference from previous iteration

        # Timing information
        start_time: datetime = Field(default_factory=datetime.utcnow)
        end_time: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def is_first_iteration(self) -> bool:
        """Check if this is the first iteration"""
        return self.number == 0

    @property
    def is_last_iteration(self) -> bool:
        """Check if this is the last allowed iteration"""
        return self.number >= self.max_iterations - 1

    @property
    def duration(self) -> float:
        """Get the duration of this iteration in seconds"""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    def update_messages(self, Q: Dict, R: Dict) -> None:
        """Update the current messages and store previous ones"""
        self.Q_previous = self.Q_messages
        self.R_previous = self.R_messages
        self.Q_messages = Q
        self.R_messages = R

    def calculate_residual(self) -> float:
        """Calculate the message residual from previous iteration"""
        if self.Q_previous is None or self.R_previous is None:
            return float('inf')

        # Calculate maximum difference in messages
        q_diff = max(np.max(np.abs(self.Q_messages[k] - self.Q_previous[k]))
                     for k in self.Q_messages.keys())
        r_diff = max(np.max(np.abs(self.R_messages[k] - self.R_previous[k]))
                     for k in self.R_messages.keys())

        self.message_residual = max(q_diff, r_diff)
        return self.message_residual

    def complete(self) -> None:
        """Mark the iteration as complete"""
        self.end_time = datetime.utcnow()