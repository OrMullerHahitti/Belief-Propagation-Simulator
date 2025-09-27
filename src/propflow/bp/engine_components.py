import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np
from ..core.dcop_base import Agent
from ..core.components import Message


@dataclass
class Step:
    """Represents a single step in the simulation.

    A step typically involves one round of message computation and exchange.

    Attributes:
        num (int): The step number.
        messages (dict): A dictionary mapping agent names to a list of messages
            they received in this step.
        q_messages (dict): Messages from variable nodes to factor nodes.
        r_messages (dict): Messages from factor nodes to variable nodes.
    """

    num: int = 0
    messages: Dict[str, List[Message]] = field(default_factory=dict)
    q_messages: Dict[str, list] = field(default_factory=dict)  # variable-to-factor
    r_messages: Dict[str, list] = field(default_factory=dict)  # factor-to-variable

    def add(self, agent: Agent, message: Message):
        """Adds a received message to an agent for this step.

        Args:
            agent: The agent who will receive the message.
            message: The message to be added.
        """
        if agent.name not in self.messages:
            self.messages[agent.name] = []
        self.messages[agent.name].append(message)

    def add_q(self, var_name: str, messages: list):
        """Adds outgoing Q messages from a variable node.

        Args:
            var_name: The name of the sending variable node.
            messages: A list of Q messages.
        """
        self.q_messages[var_name] = messages

    def add_r(self, factor_name: str, messages: list):
        """Adds outgoing R messages from a factor node.

        Args:
            factor_name: The name of the sending factor node.
            messages: A list of R messages.
        """
        self.r_messages[factor_name] = messages


@dataclass
class Cycle:
    """Represents a full message-passing cycle in the simulation.

    A cycle consists of a sequence of steps, typically corresponding to the
    diameter of the graph.

    Attributes:
        number (int): The cycle number.
        steps (List[Step]): A list of steps that make up the cycle.
    """

    number: int
    steps: List[Step] = field(default_factory=list)

    def add(self, step: Step):
        """Adds a step to the cycle.

        Args:
            step: The `Step` object to add.
        """
        self.steps.append(step)

    def __eq__(self, other: "Cycle") -> bool:
        """Checks if two cycles are equal by comparing their steps' messages.

        Args:
            other: The other `Cycle` object to compare against.

        Returns:
            True if the cycles are equal, False otherwise.
        """
        if len(self.steps) != len(other.steps):
            return False
        for step1, step2 in zip(self.steps, other.steps):
            if step1.messages != step2.messages:
                return False
        return True


@dataclass
class MessageData:
    """A data structure for storing simplified message information for BCT analysis.

    Attributes:
        sender (str): The name of the message sender.
        recipient (str): The name of the message recipient.
        data (List[float]): The numerical content of the message.
        step (int): The simulation step in which the message was sent.
    """

    sender: str
    recipient: str
    data: List[float]  # Message values
    step: int


class History:
    """Tracks the state of a simulation over time.

    This class can operate in two modes:
    1.  **Legacy Mode**: Tracks data on a cycle-by-cycle basis, storing beliefs,
        assignments, and costs at the end of each message passing cycle.
    2.  **BCT Mode** (`use_bct_history=True`): Tracks data at every single step,
        providing a much more granular history suitable for detailed analysis
        with tools like the Belief Propagation Analysis Toolkit (BCT).

    Attributes:
        config (dict): The configuration parameters of the simulation.
        cycles (dict): A dictionary of `Cycle` objects, keyed by cycle number.
        beliefs (dict): A dictionary of belief states, keyed by cycle/step number.
        assignments (dict): A dictionary of variable assignments, keyed by cycle/step number.
        costs (list): A list of global costs over cycles/steps.
        engine_type (str): The type of engine used for the simulation.
        use_bct_history (bool): A flag indicating whether to use step-by-step tracking.
    """

    def __init__(
        self, engine_type: str = "Engine", use_bct_history: bool = False, **kwargs
    ):
        """Initializes the History tracker.

        Args:
            engine_type: The name of the engine class running the simulation.
            use_bct_history: If True, enables detailed step-by-step data collection.
            **kwargs: Additional configuration parameters to be stored.
        """
        self.config = dict(kwargs)
        self.cycles: Dict[int, "Cycle"] = {}
        self.beliefs: Dict[int, Dict[str, np.ndarray]] = {}
        self.assignments: Dict[int, Dict[str, Union[int, float]]] = {}
        self.costs: List[Union[int, float]] = []
        self.engine_type = engine_type
        self.use_bct_history = use_bct_history

        if self.use_bct_history:
            # BCT-specific attributes for step-by-step tracking
            self.step_beliefs: Dict[int, Dict[str, float]] = {}
            self.step_assignments: Dict[int, Dict[str, int]] = {}
            self.step_messages: Dict[int, List[MessageData]] = {}
            self.step_costs: List[float] = []
            self.current_step = 0

    def __setitem__(self, key: int, value: Cycle):
        """Allows setting a cycle using dictionary-like syntax."""
        self.cycles[key] = value

    def __getitem__(self, key: int) -> Cycle:
        """Allows retrieving a cycle using dictionary-like syntax."""
        return self.cycles[key]

    def initialize_cost(self, x: Union[int, float]) -> None:
        """Initializes the cost history with a baseline value.

        Args:
            x: The initial cost value.
        """
        for _ in range(5):  # Pad with initial cost for stability
            self.costs.append(x)
            if self.use_bct_history:
                self.step_costs.append(float(x))

    def compare_last_two_cycles(self) -> bool:
        """Compares the assignments of the last two recorded cycles.

        Returns:
            True if the assignments are identical, False otherwise.
        """
        if len(self.cycles) < 2:
            return False
        last_iteration = list(self.cycles)[-1]
        last_cycle = list(self.assignments[last_iteration].values())
        second_last_cycle = list(self.assignments[last_iteration - 1].values())
        return last_cycle == second_last_cycle

    @property
    def name(self) -> str:
        """str: A configurable name for the history record."""
        return "test_1"  # TODO: make this configurable

    def track_step_data(self, step_num: int, step_result: Step, engine) -> None:
        """Tracks detailed data for a single step if BCT mode is enabled.

        Args:
            step_num: The current step number.
            step_result: The `Step` object containing message data for the step.
            engine: The engine instance, used to query current state.
        """
        if not self.use_bct_history:
            return

        self.current_step = step_num

        if hasattr(engine, "get_beliefs"):
            current_beliefs = engine.get_beliefs()
            step_beliefs = {}
            for var_name, belief_array in current_beliefs.items():
                if isinstance(belief_array, np.ndarray):
                    belief_value = float(np.min(belief_array))
                else:
                    belief_value = float(belief_array) if belief_array is not None else 0.0
                step_beliefs[var_name] = belief_value
            self.step_beliefs[step_num] = step_beliefs

        if hasattr(engine, "assignments"):
            current_assignments = engine.assignments
            self.step_assignments[step_num] = {k: int(v) for k, v in current_assignments.items()}

        if hasattr(step_result, "messages"):
            step_messages = []
            for agent_name, agent_messages in step_result.messages.items():
                for message in agent_messages:
                    if hasattr(message, "sender") and hasattr(message, "recipient"):
                        data_list = (
                            message.data.tolist()
                            if isinstance(message.data, np.ndarray)
                            else [float(message.data)]
                        )
                        step_messages.append(
                            MessageData(
                                sender=getattr(message.sender, "name", "unknown"),
                                recipient=getattr(message.recipient, "name", "unknown"),
                                data=data_list,
                                step=step_num,
                            )
                        )
            self.step_messages[step_num] = step_messages

        if hasattr(engine, "calculate_global_cost"):
            try:
                self.step_costs.append(float(engine.calculate_global_cost()))
            except Exception:
                fallback_cost = self.step_costs[-1] if self.step_costs else 0.0
                self.step_costs.append(fallback_cost)

    def get_bct_data(self) -> Dict:
        """Returns the history formatted for the BCT analysis tool.

        If `use_bct_history` is True, it returns fine-grained step-by-step data.
        Otherwise, it converts the legacy cycle-based data into the BCT format.

        Returns:
            A dictionary containing beliefs, messages, assignments, costs, and metadata.
        """
        if not self.use_bct_history:
            return self._convert_legacy_to_bct_format()

        return {
            "beliefs": self._format_step_beliefs(),
            "messages": self._format_step_messages(),
            "assignments": self._format_step_assignments(),
            "costs": self.step_costs.copy(),
            "metadata": {
                "engine_type": self.engine_type,
                "use_bct_history": self.use_bct_history,
                "total_steps": len(self.step_beliefs),
                "has_step_data": True,
            },
        }

    def _format_step_beliefs(self) -> Dict[str, List[float]]:
        """Converts step-by-step beliefs into a BCT-compatible time series format."""
        beliefs_by_var = {}
        for step_num in sorted(self.step_beliefs.keys()):
            step_data = self.step_beliefs[step_num]
            for var_name, belief_value in step_data.items():
                beliefs_by_var.setdefault(var_name, []).append(belief_value)
        return beliefs_by_var

    def _format_step_assignments(self) -> Dict[str, List[int]]:
        """Converts step-by-step assignments into a BCT-compatible time series format."""
        assignments_by_var = {}
        for step_num in sorted(self.step_assignments.keys()):
            step_data = self.step_assignments[step_num]
            for var_name, assignment in step_data.items():
                assignments_by_var.setdefault(var_name, []).append(assignment)
        return assignments_by_var

    def _format_step_messages(self) -> Dict[str, List[float]]:
        """Converts step-by-step messages into a BCT-compatible time series format."""
        messages_by_flow = {}
        for step_num in sorted(self.step_messages.keys()):
            for msg_data in self.step_messages[step_num]:
                key = f"{msg_data.sender}->{msg_data.recipient}"
                value = msg_data.data[0] if msg_data.data else 0.0
                messages_by_flow.setdefault(key, []).append(value)
        return messages_by_flow

    def _convert_legacy_to_bct_format(self) -> Dict:
        """Converts legacy cycle-based data to a BCT-compatible format."""
        beliefs_by_var = {}
        for cycle_num in sorted(self.beliefs.keys()):
            for var_name, belief_array in self.beliefs[cycle_num].items():
                value = (
                    float(np.min(belief_array))
                    if isinstance(belief_array, np.ndarray)
                    else float(belief_array) if belief_array is not None else 0.0
                )
                beliefs_by_var.setdefault(var_name, []).append(value)

        assignments_by_var = {}
        for cycle_num in sorted(self.assignments.keys()):
            for var_name, assignment in self.assignments[cycle_num].items():
                assignments_by_var.setdefault(var_name, []).append(int(assignment))

        return {
            "beliefs": beliefs_by_var,
            "messages": {},  # No step-by-step messages in legacy mode
            "assignments": assignments_by_var,
            "costs": [float(cost) for cost in self.costs],
            "metadata": {
                "engine_type": self.engine_type,
                "use_bct_history": self.use_bct_history,
                "total_steps": len(self.beliefs),
                "has_step_data": False,
            },
        }

    def to_json(self, filepath: str) -> str:
        """Saves the history data to a JSON file.

        The format of the JSON file depends on whether BCT mode is enabled.

        Args:
            filepath: The path to the output JSON file.

        Returns:
            The filepath where the history was saved.
        """
        data = self.get_bct_data() if self.use_bct_history else self._get_legacy_json_data()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, dict): return {str(k): convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [convert_numpy(item) for item in obj]
            return obj

        with open(filepath, "w") as f:
            json.dump(convert_numpy(data), f, indent=2)

        print(f"History saved to: {filepath}")
        return filepath

    def _get_legacy_json_data(self) -> Dict:
        """Returns the history data in the original, legacy JSON format."""
        return {
            "config": self.config,
            "engine_type": self.engine_type,
            "cycles": self._serialize_cycles(),
            "beliefs": self._serialize_beliefs(),
            "assignments": self._serialize_assignments(),
            "costs": [float(cost) for cost in self.costs],
        }

    def _serialize_cycles(self) -> Dict:
        """Serializes `Cycle` objects for JSON output."""
        serialized = {}
        for cycle_num, cycle in self.cycles.items():
            serialized[str(cycle_num)] = {
                "number": cycle.number,
                "steps": self._serialize_steps(cycle.steps),
            }
        return serialized

    def _serialize_steps(self, steps: List[Step]) -> List[Dict]:
        """Serializes `Step` objects for JSON output."""
        return [
            {"num": step.num, "messages": self._serialize_step_messages(step.messages)}
            for step in steps
        ]

    def _serialize_step_messages(self, messages: Dict) -> Dict:
        """Serializes message objects within a step for JSON output."""
        serialized = {}
        for agent_name, agent_messages in messages.items():
            serialized[str(agent_name)] = [
                {
                    "sender": getattr(msg.sender, "name", "unknown"),
                    "recipient": getattr(msg.recipient, "name", "unknown"),
                    "data": msg.data.tolist() if isinstance(msg.data, np.ndarray) else [float(msg.data)],
                }
                for msg in agent_messages
            ]
        return serialized

    def _serialize_beliefs(self) -> Dict:
        """Serializes belief dictionaries for JSON output."""
        serialized = {}
        for cycle_num, beliefs in self.beliefs.items():
            serialized[str(cycle_num)] = {
                var_name: belief_array.tolist() if isinstance(belief_array, np.ndarray) else belief_array
                for var_name, belief_array in beliefs.items()
            }
        return serialized

    def _serialize_assignments(self) -> Dict:
        """Serializes assignment dictionaries for JSON output."""
        return {
            str(cycle_num): {var_name: int(assignment) for var_name, assignment in assignments.items()}
            for cycle_num, assignments in self.assignments.items()
        }

    def save_results(self, filename: str = None) -> str:
        """Saves the history to a JSON file (backward-compatible wrapper for `to_json`).

        Args:
            filename: The name for the output file.

        Returns:
            The filepath where the results were saved.
        """
        if filename is None:
            filename = f"{self.name}_results.json"
        return self.to_json(filename)

    def save_csv(self, config_name: Optional[str] = None) -> str:
        """Placeholder for saving results in CSV format.

        Args:
            config_name: The name of the configuration.

        Returns:
            An empty string (not implemented).
        """
        return ""
