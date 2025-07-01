import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from base_models.dcop_base import Agent
from base_models.components import Message


@dataclass
class Step:
    """
    A class to represent a step in the factor graph.
    """

    num: int = 0
    messages: Dict[str, List[Message]] = field(default_factory=dict)
    q_messages: Dict[str, list] = field(default_factory=dict)  # variable-to-factor
    r_messages: Dict[str, list] = field(default_factory=dict)  # factor-to-variable

    def add(self, agent: Agent, message: Message):
        """
        Add a List of messages for each agent per step.
        :param agent: Agent who will send the messages next step
        :param message: the messages to be sent
        :return:
        """
        if agent.name not in self.messages:
            self.messages[agent.name] = []
        # Ensure messages is a list, even if None
        self.messages[agent.name].append(message)

    def add_q(self, var_name: str, messages: list):
        self.q_messages[var_name] = messages

    def add_r(self, factor_name: str, messages: list):
        self.r_messages[factor_name] = messages


@dataclass
class Cycle:
    """
    A class to represent a cycle in the factor graph.
    """

    number: int
    steps: List[Step] = field(default_factory=list)

    def add(self, step: Step):
        """
        Add a step to the cycle.
        """
        self.steps.append(step)

    def __eq__(self, other: "Cycle"):
        """
        Check if two cycles are equal.
        """
        if len(self.steps) != len(other.steps):
            return False
        for step1, step2 in zip(self.steps, other.steps):
            if step1.messages != step2.messages:
                return False
        return True


@dataclass
class MessageData:
    """Simple structure to store message data for BCT"""

    sender: str
    recipient: str
    data: List[float]  # Message values
    step: int


class History:
    """Enhanced History class with optional BCT data collection"""

    def __init__(
        self, engine_type: str = "Engine", use_bct_history: bool = False, **kwargs
    ):
        # Original History attributes
        self.config = dict(kwargs)
        self.cycles: Dict[int, "Cycle"] = {}
        self.beliefs: Dict[int, Dict[str, np.ndarray]] = {}
        self.assignments: Dict[int, Dict[str, Union[int, float]]] = {}
        self.costs: List[Union[int, float]] = []
        self.engine_type = engine_type

        # BCT-specific attributes
        self.use_bct_history = use_bct_history

        if self.use_bct_history:
            # Step-by-step tracking for BCT
            self.step_beliefs: Dict[int, Dict[str, float]] = (
                {}
            )  # step -> var -> belief_value
            self.step_assignments: Dict[int, Dict[str, int]] = (
                {}
            )  # step -> var -> assignment
            self.step_messages: Dict[int, List[MessageData]] = (
                {}
            )  # step -> list of messages
            self.step_costs: List[float] = []  # cost per step
            self.current_step = 0

    def __setitem__(self, key: int, value):
        self.cycles[key] = value

    def __getitem__(self, key: int):
        return self.cycles[key]

    def initialize_cost(self, x: Union[int, float]) -> None:
        """Initialize cost baseline"""
        for _ in range(5):
            self.costs.append(x)
            if self.use_bct_history:
                self.step_costs.append(float(x))

    def compare_last_two_cycles(self):
        """Compare last two cycles for convergence"""
        if len(self.cycles) < 2:
            return False
        last_iteration = list(self.cycles)[-1]
        last_cycle = list(self.assignments[last_iteration].values())
        second_last_cycle = list(self.assignments[last_iteration - 1].values())
        return last_cycle == second_last_cycle

    @property
    def name(self):
        return f"test_1"  # TODO: make this configurable

    # BCT-specific methods
    def track_step_data(self, step_num: int, step_result, engine) -> None:
        """Track detailed step data for BCT (only if BCT mode is enabled)"""
        if not self.use_bct_history:
            return

        self.current_step = step_num

        # Track beliefs at this step
        if hasattr(engine, "get_beliefs"):
            current_beliefs = engine.get_beliefs()
            step_beliefs = {}
            for var_name, belief_array in current_beliefs.items():
                # Extract single value from belief array
                if isinstance(belief_array, np.ndarray):
                    belief_value = float(np.min(belief_array))  # Use min for min-sum
                elif belief_array is not None:
                    belief_value = float(belief_array)
                else:
                    belief_value = 0.0
                step_beliefs[var_name] = belief_value

            self.step_beliefs[step_num] = step_beliefs

        # Track assignments at this step
        if hasattr(engine, "assignments"):
            current_assignments = engine.assignments
            step_assignments = {}
            for var_name, assignment in current_assignments.items():
                step_assignments[var_name] = int(assignment)

            self.step_assignments[step_num] = step_assignments

        # Track messages from this step
        if hasattr(step_result, "messages"):
            step_messages = []
            for agent_name, agent_messages in step_result.messages.items():
                for message in agent_messages:
                    if hasattr(message, "sender") and hasattr(message, "recipient"):
                        sender_name = getattr(
                            message.sender, "name", str(message.sender)
                        )
                        recipient_name = getattr(
                            message.recipient, "name", str(message.recipient)
                        )

                        # Extract message data
                        if hasattr(message, "data"):
                            if isinstance(message.data, np.ndarray):
                                data_list = message.data.tolist()
                            else:
                                data_list = [float(message.data)]
                        else:
                            data_list = [0.0]

                        msg_data = MessageData(
                            sender=sender_name,
                            recipient=recipient_name,
                            data=data_list,
                            step=step_num,
                        )
                        step_messages.append(msg_data)

            self.step_messages[step_num] = step_messages

        # Track cost at this step
        if hasattr(engine, "calculate_global_cost"):
            try:
                current_cost = engine.calculate_global_cost()
                self.step_costs.append(float(current_cost))
            except:
                # Fallback if cost calculation fails
                if self.step_costs:
                    self.step_costs.append(self.step_costs[-1])
                else:
                    self.step_costs.append(0.0)

    def get_bct_data(self) -> Dict:
        """Get BCT-compatible data structure"""
        if not self.use_bct_history:
            # Return legacy format from cycle data
            return self._convert_legacy_to_bct_format()

        # Return step-by-step BCT data
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
        """Convert step beliefs to BCT format"""
        beliefs_by_var = {}
        for step_num in sorted(self.step_beliefs.keys()):
            step_data = self.step_beliefs[step_num]
            for var_name, belief_value in step_data.items():
                if var_name not in beliefs_by_var:
                    beliefs_by_var[var_name] = []
                beliefs_by_var[var_name].append(belief_value)
        return beliefs_by_var

    def _format_step_assignments(self) -> Dict[str, List[int]]:
        """Convert step assignments to BCT format"""
        assignments_by_var = {}
        for step_num in sorted(self.step_assignments.keys()):
            step_data = self.step_assignments[step_num]
            for var_name, assignment in step_data.items():
                if var_name not in assignments_by_var:
                    assignments_by_var[var_name] = []
                assignments_by_var[var_name].append(assignment)
        return assignments_by_var

    def _format_step_messages(self) -> Dict[str, List[float]]:
        """Convert step messages to BCT format"""
        messages_by_flow = {}
        for step_num in sorted(self.step_messages.keys()):
            step_data = self.step_messages[step_num]
            for msg_data in step_data:
                key = f"{msg_data.sender}->{msg_data.recipient}"
                if key not in messages_by_flow:
                    messages_by_flow[key] = []

                # Use first element of message data or min/max as appropriate
                if msg_data.data:
                    value = msg_data.data[0]  # Can be changed to min/max if needed
                else:
                    value = 0.0
                messages_by_flow[key].append(value)
        return messages_by_flow

    def _convert_legacy_to_bct_format(self) -> Dict:
        """Convert legacy cycle-based data to BCT format"""
        beliefs_by_var = {}
        assignments_by_var = {}

        # Convert beliefs
        for cycle_num in sorted(self.beliefs.keys()):
            cycle_beliefs = self.beliefs[cycle_num]
            for var_name, belief_array in cycle_beliefs.items():
                if var_name not in beliefs_by_var:
                    beliefs_by_var[var_name] = []

                if isinstance(belief_array, np.ndarray):
                    belief_value = float(np.min(belief_array))
                else:
                    belief_value = (
                        float(belief_array) if belief_array is not None else 0.0
                    )
                beliefs_by_var[var_name].append(belief_value)

        # Convert assignments
        for cycle_num in sorted(self.assignments.keys()):
            cycle_assignments = self.assignments[cycle_num]
            for var_name, assignment in cycle_assignments.items():
                if var_name not in assignments_by_var:
                    assignments_by_var[var_name] = []
                assignments_by_var[var_name].append(int(assignment))

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
        """Save history data to JSON file"""
        # Get the appropriate data format
        if self.use_bct_history:
            data = self.get_bct_data()
        else:
            data = self._get_legacy_json_data()

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            return obj

        clean_data = convert_numpy(data)

        with open(filepath, "w") as f:
            json.dump(clean_data, f, indent=2)

        print(f"History saved to: {filepath}")
        return filepath

    def _get_legacy_json_data(self) -> Dict:
        """Get legacy JSON data format (original save_results format)"""
        return {
            "config": self.config,
            "engine_type": self.engine_type,
            "cycles": self._serialize_cycles(),
            "beliefs": self._serialize_beliefs(),
            "assignments": self._serialize_assignments(),
            "costs": [float(cost) for cost in self.costs],
        }

    def _serialize_cycles(self) -> Dict:
        """Serialize cycles for JSON"""
        serialized = {}
        for cycle_num, cycle in self.cycles.items():
            serialized[str(cycle_num)] = {
                "number": cycle.number if hasattr(cycle, "number") else cycle_num,
                "steps": self._serialize_steps(
                    cycle.steps if hasattr(cycle, "steps") else []
                ),
            }
        return serialized

    def _serialize_steps(self, steps) -> List[Dict]:
        """Serialize steps for JSON"""
        serialized_steps = []
        for step in steps:
            step_data = {
                "num": getattr(step, "num", 0),
                "messages": self._serialize_step_messages(
                    getattr(step, "messages", {})
                ),
            }
            serialized_steps.append(step_data)
        return serialized_steps

    def _serialize_step_messages(self, messages) -> Dict:
        """Serialize step messages for JSON"""
        serialized = {}
        for agent_name, agent_messages in messages.items():
            serialized[str(agent_name)] = []
            for msg in agent_messages:
                msg_data = {
                    "sender": (
                        getattr(msg.sender, "name", str(msg.sender))
                        if hasattr(msg, "sender")
                        else "unknown"
                    ),
                    "recipient": (
                        getattr(msg.recipient, "name", str(msg.recipient))
                        if hasattr(msg, "recipient")
                        else "unknown"
                    ),
                    "data": (
                        msg.data.tolist()
                        if isinstance(msg.data, np.ndarray)
                        else [float(msg.data)] if hasattr(msg, "data") else []
                    ),
                }
                serialized[str(agent_name)].append(msg_data)
        return serialized

    def _serialize_beliefs(self) -> Dict:
        """Serialize beliefs for JSON"""
        serialized = {}
        for cycle_num, beliefs in self.beliefs.items():
            serialized[str(cycle_num)] = {}
            for var_name, belief_array in beliefs.items():
                if isinstance(belief_array, np.ndarray):
                    serialized[str(cycle_num)][var_name] = belief_array.tolist()
                else:
                    serialized[str(cycle_num)][var_name] = belief_array
        return serialized

    def _serialize_assignments(self) -> Dict:
        """Serialize assignments for JSON"""
        serialized = {}
        for cycle_num, assignments in self.assignments.items():
            serialized[str(cycle_num)] = {}
            for var_name, assignment in assignments.items():
                serialized[str(cycle_num)][var_name] = int(assignment)
        return serialized

    # Keep original methods for backward compatibility
    def save_results(self, filename: str = None) -> str:
        """Original save_results method for backward compatibility"""
        if filename is None:
            filename = f"{self.name}_results.json"
        return self.to_json(filename)

    # TODO: Implement save_csv if needed
    def save_csv(self, config_name: Optional[str] = None) -> str:
        """Save CSV format (placeholder - implement if needed)"""
        # This would implement the original CSV saving logic
        # For now, just return empty string to maintain compatibility
        return ""
