import json
import os
import numpy as np
from typing import Dict, List, Union, Optional

def save_simulation_data(engine, filepath: str) -> str:
    """
    Save simulation data from a BPEngine instance to a JSON file.
    Includes information about agents, factors, convergence metrics, and history.

    Args:
        engine: BPEngine instance that has completed a run
        filepath: Path to save the JSON file

    Returns:
        Path to the saved JSON file
    """
    # Extract agent information
    agents_data = []
    for idx, agent in enumerate(engine.graph.variables):
        # Convert domain from int to list of string values
        domain_values = []
        if isinstance(agent.domain, int):
            # If domain is an integer, create list of strings ["0", "1", ..., "n-1"]
            domain_values = [str(i) for i in range(agent.domain)]
        else:
            # If domain is already a list, convert to strings
            domain_values = [str(val) for val in agent.domain]

        agents_data.append({
            "id": f"agent{idx+1}",  # Create a consistent ID
            "name": agent.name,
            "domain": domain_values
        })

    # Extract factor information
    factors_data = []
    for idx, factor in enumerate(engine.graph.factors):
        # Extract connected agents
        connected_agents = []
        if hasattr(factor, 'connection_number'):
            # Map variable names to agent IDs
            name_to_id = {agent["name"]: agent["id"] for agent in agents_data}
            connected_agents = [name_to_id.get(name, name) for name in factor.connection_number.keys()]

        # Determine factor type based on number of connected agents
        factor_type = "binary" if len(connected_agents) > 1 else "unary"

        # Extract default cost (use mean if no specific default)
        default_cost = -1
        if hasattr(factor, 'cost_table') and factor.cost_table is not None:
            default_cost = float(factor.cost_table.mean())

        factors_data.append({
            "id": f"factor{idx+1}",
            "name": factor.name,
            "connectedAgents": connected_agents,
            "type": factor_type,
            "defaultCost": default_cost
        })

    # Create the base data dictionary
    data = {
        "agents": agents_data,
        "factors": factors_data
    }

    # Add convergence metrics and history if available
    if hasattr(engine, 'convergence_monitor') and engine.convergence_monitor:
        convergence_data = {}
        if hasattr(engine.convergence_monitor, 'convergence_history'):
            convergence_data["convergence_history"] = engine.convergence_monitor.convergence_history
        data["convergence"] = convergence_data

    # Add run history if available
    if hasattr(engine, 'history') and engine.history:
        history_data = {}

        # Add cost history
        if hasattr(engine.history, 'costs') and engine.history.costs:
            # Convert numpy values to Python native types
            history_data["costs"] = [float(cost) for cost in engine.history.costs]

        # Add belief history (simplified to avoid large data)
        if hasattr(engine.history, 'beliefs') and engine.history.beliefs:
            beliefs_summary = {}
            for cycle, beliefs in engine.history.beliefs.items():
                beliefs_summary[str(cycle)] = {
                    agent_name: _serialize_numpy_array(belief)
                    for agent_name, belief in beliefs.items()
                }
            history_data["beliefs_summary"] = beliefs_summary

        # Add assignment history
        if hasattr(engine.history, 'assignments') and engine.history.assignments:
            assignments_data = {}
            for cycle, assignments in engine.history.assignments.items():
                assignments_data[str(cycle)] = {
                    agent_name: int(assignment)
                    for agent_name, assignment in assignments.items()
                }
            history_data["assignments"] = assignments_data

        data["history"] = history_data

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    # Save to file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return filepath

def _serialize_numpy_array(arr):
    """Helper function to convert numpy arrays and other non-JSON serializable types to Python types."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, np.integer):
        return int(arr)
    elif isinstance(arr, np.floating):
        return float(arr)
    elif isinstance(arr, np.bool_):
        return bool(arr)
    elif isinstance(arr, (list, tuple)):
        return [_serialize_numpy_array(item) for item in arr]
    elif isinstance(arr, dict):
        return {k: _serialize_numpy_array(v) for k, v in arr.items()}
    return arr

def save_simulation_result(engine, filepath: str) -> str:
    """
    filepath: Path to save the JSON file example: "data/simulation_result.json"
    Save simulation result in the required JSON structure for frontend consumption.
    """
    import time
    def _to_py(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (list, tuple)):
            return [_to_py(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _to_py(v) for k, v in obj.items()}
        if isinstance(obj, bool):
            return bool(obj)
        return obj

    history = getattr(engine, 'history', None)
    convergence_monitor = getattr(engine, 'convergence_monitor', None)
    steps = []
    # Compose steps (minimal example, expand as needed)
    for i, cost in enumerate(getattr(history, 'costs', [])):
        step = {
            "iteration": i,
            "timestamp": int(time.time() * 1000),
            "messages": [],  # Could be filled with message info if needed
            "agentBeliefs": _to_py(history.beliefs.get(i, {})),
            "selectedConstraints": [],  # Could be filled with factor names if needed
            "globalCost": float(cost),
            "convergenceMetric": 0.0  # Placeholder, can be filled with real metric
        }
        steps.append(step)

    # Final beliefs (convert to readable form if needed)
    final_beliefs = {}
    if history and history.beliefs:
        last_beliefs = history.beliefs[min(history.beliefs.keys())]
        for agent, belief in last_beliefs.items():
            # Example: pick argmin/argmax or string label
            if hasattr(belief, 'tolist'):
                arr = belief.tolist()
                final_beliefs[agent] = str(arr.index(min(arr))) if arr else "unknown"
            else:
                final_beliefs[agent] = str(belief)

    total_iterations = len(history.costs) if history and hasattr(history, 'costs') else 0
    convergence_achieved = False
    if convergence_monitor and hasattr(convergence_monitor, 'convergence_history'):
        convergence_achieved = any(
            h.get('belief_converged', False) and h.get('assignment_converged', False)
            for h in convergence_monitor.convergence_history
        )
    execution_time = 0  # Could be filled with timing info if available
    message_count = 0   # Could be filled with message count if available

    simulation_result = {
        "steps": steps,
        "finalBeliefs": final_beliefs,
        "totalIterations": total_iterations,
        "convergenceAchieved": bool(convergence_achieved),
        "executionTime": execution_time,
        "messageCount": message_count
    }

    # Example metrics (fill with real values if available)
    metrics = {
        "convergenceRate": 0.85,
        "messageEfficiency": 0.72,
        "beliefStability": 0.91,
        "constraintSatisfaction": 0.88,
        "communicationOverhead": 0.65
    }

    data = {
        "simulationResult": simulation_result,
        "metrics": metrics
    }

    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    import json
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    return filepath

