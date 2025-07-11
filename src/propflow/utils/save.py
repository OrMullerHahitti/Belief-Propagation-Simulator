import json
import os
import csv
import numpy as np
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
import time
from dataclasses import dataclass


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

        agents_data.append(
            {
                "id": f"agent{ idx + 1 }",  # Create a consistent ID
                "name": agent.name,
                "domain": domain_values,
            }
        )

    # Extract factor information
    factors_data = []
    for idx, factor in enumerate(engine.graph.factors):
        # Extract connected agents
        connected_agents = []
        if hasattr(factor, "connection_number"):
            # Map variable names to agent IDs
            name_to_id = {agent["name"]: agent["id"] for agent in agents_data}
            connected_agents = [
                name_to_id.get(name, name) for name in factor.connection_number.keys()
            ]

        # Determine factor type based on number of connected agents
        factor_type = "binary" if len(connected_agents) > 1 else "unary"

        # Extract default cost (use mean if no specific default)
        default_cost = -1
        if hasattr(factor, "cost_table") and factor.cost_table is not None:
            default_cost = float(factor.cost_table.mean())

        factors_data.append(
            {
                "id": f"factor{idx+1}",
                "name": factor.name,
                "connectedAgents": connected_agents,
                "type": factor_type,
                "defaultCost": default_cost,
            }
        )

    # Create the base data dictionary
    data = {"agents": agents_data, "factors": factors_data}

    # Add convergence metrics and history if available
    if hasattr(engine, "convergence_monitor") and engine.convergence_monitor:
        convergence_data = {}
        if hasattr(engine.convergence_monitor, "convergence_history"):
            convergence_data[
                "convergence_history"
            ] = engine.convergence_monitor.convergence_history
        data["convergence"] = convergence_data

    # Add run history if available
    if hasattr(engine, "history") and engine.history:
        history_data = {}

        # Add cost history
        if hasattr(engine.history, "costs") and engine.history.costs:
            # Convert numpy values to Python native types
            history_data["costs"] = [float(cost) for cost in engine.history.costs]

        # Add belief history (simplified to avoid large data)
        if hasattr(engine.history, "beliefs") and engine.history.beliefs:
            beliefs_summary = {}
            for cycle, beliefs in engine.history.beliefs.items():
                beliefs_summary[str(cycle)] = {
                    agent_name: _serialize_numpy_array(belief)
                    for agent_name, belief in beliefs.items()
                }
            history_data["beliefs_summary"] = beliefs_summary

        # Add assignment history
        if hasattr(engine.history, "assignments") and engine.history.assignments:
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
    with open(filepath, "w") as f:
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

    history = getattr(engine, "history", None)
    convergence_monitor = getattr(engine, "convergence_monitor", None)
    steps = []
    # Compose steps (minimal example, expand as needed)
    for i, cost in enumerate(getattr(history, "costs", [])):
        step = {
            "iteration": i,
            "timestamp": int(time.time() * 1000),
            "messages": [],  # Could be filled with message info if needed
            "agentBeliefs": _to_py(history.beliefs.get(i, {})),
            "selectedConstraints": [],  # Could be filled with factor names if needed
            "globalCost": float(cost),
            "convergenceMetric": 0.0,  # Placeholder, can be filled with real metric
        }
        steps.append(step)

    # Final beliefs (convert to readable form if needed)
    final_beliefs = {}
    if history and history.beliefs:
        last_beliefs = history.beliefs[min(history.beliefs.keys())]
        for agent, belief in last_beliefs.items():
            # Example: pick argmin/argmax or string label
            if hasattr(belief, "tolist"):
                arr = belief.tolist()
                final_beliefs[agent] = str(arr.index(min(arr))) if arr else "unknown"
            else:
                final_beliefs[agent] = str(belief)

    total_iterations = (
        len(history.costs) if history and hasattr(history, "costs") else 0
    )
    convergence_achieved = False
    if convergence_monitor and hasattr(convergence_monitor, "convergence_history"):
        convergence_achieved = any(
            h.get("belief_converged", False) and h.get("assignment_converged", False)
            for h in convergence_monitor.convergence_history
        )
    execution_time = 0  # Could be filled with timing info if available
    message_count = 0  # Could be filled with message count if available

    simulation_result = {
        "steps": steps,
        "finalBeliefs": final_beliefs,
        "totalIterations": total_iterations,
        "convergenceAchieved": bool(convergence_achieved),
        "executionTime": execution_time,
        "messageCount": message_count,
    }

    # Example metrics (fill with real values if available)
    metrics = {
        "convergenceRate": 0.85,
        "messageEfficiency": 0.72,
        "beliefStability": 0.91,
        "constraintSatisfaction": 0.88,
        "communicationOverhead": 0.65,
    }

    data = {"simulationResult": simulation_result, "metrics": metrics}

    # Ensure directory exists
    import os

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    import json

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    return filepath


@dataclass
class SimulatorAnalysisData:
    """Data structure for comprehensive simulator analysis"""

    # Engine comparison data
    engine_stats: Dict[str, Dict[str, Any]]
    convergence_analysis: Dict[str, Dict[str, Any]]
    performance_comparison: Dict[str, Dict[str, Any]]

    # Cross-engine comparisons
    cost_convergence_comparison: Dict[str, List[float]]
    final_cost_distributions: Dict[str, List[float]]

    # Metadata
    total_runs: int
    graph_count: int
    timestamp: str
    config_summary: Dict[str, Any]


class EnhancedSaveModule:
    """Enhanced save module with two main functionalities:
    1. Simulator accumulated data analysis
    2. Enhanced single engine analysis
    """

    def __init__(self):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

    def save_simulator_analysis(
        self, simulator, filepath: Optional[str] = None, save_csv: bool = True
    ) -> str:
        """
        Save comprehensive analysis of simulator data across multiple engines and runs.

        Args:
            simulator: Simulator instance with accumulated results
            filepath: Optional custom filepath (defaults to timestamped file)
            save_csv: Whether to also save CSV summary

        Returns:
            Path to saved JSON file
        """
        if filepath is None:
            filepath = f"simulator_analysis_{self.timestamp}.json"

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        # Extract comprehensive data
        analysis_data = self._extract_simulator_data(simulator)

        # Convert to serializable format
        serializable_data = self._make_json_serializable(analysis_data.__dict__)

        # Save JSON
        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=2)

        # Save CSV summary if requested
        if save_csv:
            csv_path = filepath.replace(".json", "_summary.csv")
            self._save_simulator_csv(analysis_data, csv_path)

        print(f"Simulator analysis saved to: {filepath}")
        if save_csv:
            print(f"CSV summary saved to: {csv_path}")

        return filepath

    def save_enhanced_engine_data(
        self,
        engine,
        filepath: Optional[str] = None,
        include_performance: bool = True,
        include_convergence_detail: bool = True,
    ) -> str:
        """
        Save enhanced analysis of a single engine run with detailed metrics.

        Args:
            engine: BPEngine instance that has completed a run
            filepath: Optional custom filepath
            include_performance: Include performance monitor data if available
            include_convergence_detail: Include detailed convergence analysis

        Returns:
            Path to saved JSON file
        """
        if filepath is None:
            engine_name = getattr(engine, "name", engine.__class__.__name__)
            filepath = f"engine_analysis_{engine_name}_{self.timestamp}.json"

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        # Start with existing save_simulation_data functionality
        base_data = self._get_base_engine_data(engine)

        # Add enhanced analysis
        enhanced_data = self._extract_enhanced_engine_data(
            engine, include_performance, include_convergence_detail
        )

        # Merge data
        combined_data = {**base_data, **enhanced_data}

        # Convert to serializable format
        serializable_data = self._make_json_serializable(combined_data)

        # Save JSON
        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=2)

        print(f"Enhanced engine analysis saved to: {filepath}")
        return filepath

    def _extract_simulator_data(self, simulator) -> SimulatorAnalysisData:
        """Extract comprehensive data from simulator results"""
        engine_stats = {}
        convergence_analysis = {}
        performance_comparison = {}
        cost_convergence_comparison = {}
        final_cost_distributions = {}

        for engine_name, costs_list in simulator.results.items():
            if not costs_list:
                continue

            # Filter out empty cost lists
            valid_costs_list = [c for c in costs_list if c and len(c) > 0]
            if not valid_costs_list:
                continue

            # Basic statistics
            final_costs = [costs[-1] for costs in valid_costs_list]
            all_costs_array = self._pad_and_convert_costs(valid_costs_list)
            avg_costs_per_iteration = np.mean(all_costs_array, axis=0).tolist()
            std_costs_per_iteration = np.std(all_costs_array, axis=0).tolist()

            engine_stats[engine_name] = {
                "total_runs": len(valid_costs_list),
                "average_final_cost": float(np.mean(final_costs)),
                "std_final_cost": float(np.std(final_costs)),
                "min_final_cost": float(np.min(final_costs)),
                "max_final_cost": float(np.max(final_costs)),
                "median_final_cost": float(np.median(final_costs)),
                "average_iterations": float(
                    np.mean([len(costs) for costs in valid_costs_list])
                ),
                "std_iterations": float(
                    np.std([len(costs) for costs in valid_costs_list])
                ),
            }

            # Convergence analysis
            convergence_rates = []
            convergence_times = []
            improvement_rates = []

            for costs in valid_costs_list:
                if len(costs) > 1:
                    # Simple convergence detection (cost stabilization)
                    converged = self._detect_convergence(costs)
                    convergence_rates.append(1.0 if converged else 0.0)

                    if converged:
                        convergence_times.append(self._get_convergence_time(costs))

                    # Improvement rate (initial to final cost reduction)
                    initial_cost = costs[0]
                    final_cost = costs[-1]
                    if initial_cost > 0:
                        improvement = (initial_cost - final_cost) / initial_cost
                        improvement_rates.append(improvement)

            convergence_analysis[engine_name] = {
                "convergence_rate": float(np.mean(convergence_rates))
                if convergence_rates
                else 0.0,
                "average_convergence_time": float(np.mean(convergence_times))
                if convergence_times
                else 0.0,
                "std_convergence_time": float(np.std(convergence_times))
                if convergence_times
                else 0.0,
                "average_improvement_rate": float(np.mean(improvement_rates))
                if improvement_rates
                else 0.0,
                "cost_reduction_consistency": float(np.std(improvement_rates))
                if improvement_rates
                else 0.0,
            }

            # Store data for cross-engine comparison
            cost_convergence_comparison[engine_name] = avg_costs_per_iteration
            final_cost_distributions[engine_name] = final_costs

        # Performance comparison (relative rankings)
        performance_comparison = self._calculate_performance_rankings(
            engine_stats, convergence_analysis
        )

        return SimulatorAnalysisData(
            engine_stats=engine_stats,
            convergence_analysis=convergence_analysis,
            performance_comparison=performance_comparison,
            cost_convergence_comparison=cost_convergence_comparison,
            final_cost_distributions=final_cost_distributions,
            total_runs=sum(
                len(costs_list) for costs_list in simulator.results.values()
            ),
            graph_count=len(next(iter(simulator.results.values()), [])),
            timestamp=self.timestamp,
            config_summary={
                "engines": list(simulator.engine_configs.keys()),
                "log_level": getattr(simulator, "logger", {}).level
                if hasattr(simulator, "logger")
                else "unknown",
            },
        )

    def _extract_enhanced_engine_data(
        self, engine, include_performance: bool, include_convergence_detail: bool
    ) -> Dict[str, Any]:
        """Extract enhanced analysis data from a single engine"""
        enhanced_data = {
            "analysis_metadata": {
                "timestamp": self.timestamp,
                "engine_name": getattr(engine, "name", engine.__class__.__name__),
                "engine_type": engine.__class__.__name__,
                "include_performance": include_performance,
                "include_convergence_detail": include_convergence_detail,
            }
        }

        # Performance data from performance_monitor
        if (
            include_performance
            and hasattr(engine, "performance_monitor")
            and engine.performance_monitor
        ):
            try:
                performance_summary = engine.performance_monitor.get_summary()
                enhanced_data["performance_analysis"] = {
                    "summary": performance_summary,
                    "has_detailed_metrics": True,
                }
            except Exception as e:
                enhanced_data["performance_analysis"] = {
                    "error": f"Could not extract performance data: {str(e)}",
                    "has_detailed_metrics": False,
                }
        else:
            enhanced_data["performance_analysis"] = {
                "has_detailed_metrics": False,
                "reason": "Performance monitor not available or disabled",
            }

        # Enhanced convergence analysis
        if (
            include_convergence_detail
            and hasattr(engine, "convergence_monitor")
            and engine.convergence_monitor
        ):
            convergence_data = self._analyze_convergence_details(engine)
            enhanced_data["detailed_convergence_analysis"] = convergence_data

        # Cost analysis from history
        if hasattr(engine, "history") and engine.history and engine.history.costs:
            enhanced_data["cost_analysis"] = self._analyze_cost_progression(
                engine.history.costs
            )

        # Message passing analysis (if BCT history is available)
        if (
            hasattr(engine, "history")
            and engine.history
            and hasattr(engine.history, "use_bct_history")
            and engine.history.use_bct_history
        ):
            enhanced_data["message_analysis"] = self._analyze_message_patterns(
                engine.history
            )

        return enhanced_data

    def _get_base_engine_data(self, engine) -> Dict[str, Any]:
        """Get base engine data using existing save_simulation_data logic"""
        # Extract core data similar to save_simulation_data but return as dict
        agents_data = []
        for idx, agent in enumerate(engine.graph.variables):
            domain_values = []
            if isinstance(agent.domain, int):
                domain_values = [str(i) for i in range(agent.domain)]
            else:
                domain_values = [str(val) for val in agent.domain]

            agents_data.append(
                {
                    "id": f"agent{idx+1}",
                    "name": agent.name,
                    "domain": domain_values,
                }
            )

        factors_data = []
        for idx, factor in enumerate(engine.graph.factors):
            connected_agents = []
            if hasattr(factor, "connection_number"):
                name_to_id = {agent["name"]: agent["id"] for agent in agents_data}
                connected_agents = [
                    name_to_id.get(name, name)
                    for name in factor.connection_number.keys()
                ]

            factor_type = "binary" if len(connected_agents) > 1 else "unary"
            default_cost = -1
            if hasattr(factor, "cost_table") and factor.cost_table is not None:
                default_cost = float(factor.cost_table.mean())

            factors_data.append(
                {
                    "id": f"factor{idx+1}",
                    "name": factor.name,
                    "connectedAgents": connected_agents,
                    "type": factor_type,
                    "defaultCost": default_cost,
                }
            )

        base_data = {"agents": agents_data, "factors": factors_data}

        # Add convergence and history data if available
        if hasattr(engine, "convergence_monitor") and engine.convergence_monitor:
            convergence_data = {}
            if hasattr(engine.convergence_monitor, "convergence_history"):
                convergence_data[
                    "convergence_history"
                ] = engine.convergence_monitor.convergence_history
            base_data["convergence"] = convergence_data

        if hasattr(engine, "history") and engine.history:
            history_data = {}
            if hasattr(engine.history, "costs") and engine.history.costs:
                history_data["costs"] = [float(cost) for cost in engine.history.costs]

            if hasattr(engine.history, "beliefs") and engine.history.beliefs:
                beliefs_summary = {}
                for cycle, beliefs in engine.history.beliefs.items():
                    beliefs_summary[str(cycle)] = {
                        agent_name: _serialize_numpy_array(belief)
                        for agent_name, belief in beliefs.items()
                    }
                history_data["beliefs_summary"] = beliefs_summary

            if hasattr(engine.history, "assignments") and engine.history.assignments:
                assignments_data = {}
                for cycle, assignments in engine.history.assignments.items():
                    assignments_data[str(cycle)] = {
                        agent_name: int(assignment)
                        for agent_name, assignment in assignments.items()
                    }
                history_data["assignments"] = assignments_data

            base_data["history"] = history_data

        return base_data

    def _pad_and_convert_costs(self, costs_list: List[List[float]]) -> np.ndarray:
        """Pad cost lists to same length and convert to numpy array"""
        if not costs_list:
            return np.array([])

        max_len = max(len(costs) for costs in costs_list)
        padded_costs = []
        for costs in costs_list:
            padded = costs + [costs[-1]] * (max_len - len(costs))
            padded_costs.append(padded)

        return np.array(padded_costs)

    def _detect_convergence(
        self, costs: List[float], threshold: float = 1e-6, window: int = 10
    ) -> bool:
        """Simple convergence detection based on cost stabilization"""
        if len(costs) < window:
            return False

        recent_costs = costs[-window:]
        return np.std(recent_costs) < threshold

    def _get_convergence_time(
        self, costs: List[float], threshold: float = 1e-6, window: int = 10
    ) -> int:
        """Get iteration where convergence was first detected"""
        for i in range(window, len(costs)):
            if self._detect_convergence(costs[: i + 1], threshold, window):
                return i
        return len(costs)

    def _calculate_performance_rankings(
        self, engine_stats: Dict, convergence_analysis: Dict
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate relative performance rankings between engines"""
        engines = list(engine_stats.keys())
        if not engines:
            return {}

        rankings = {}

        # Rank by final cost (lower is better)
        final_costs = [
            (name, stats["average_final_cost"]) for name, stats in engine_stats.items()
        ]
        final_costs.sort(key=lambda x: x[1])

        # Rank by convergence rate (higher is better)
        conv_rates = [
            (name, analysis["convergence_rate"])
            for name, analysis in convergence_analysis.items()
        ]
        conv_rates.sort(key=lambda x: x[1], reverse=True)

        for i, engine in enumerate(engines):
            final_cost_rank = (
                next(j for j, (name, _) in enumerate(final_costs) if name == engine) + 1
            )
            conv_rate_rank = (
                next(j for j, (name, _) in enumerate(conv_rates) if name == engine) + 1
            )

            rankings[engine] = {
                "final_cost_rank": final_cost_rank,
                "convergence_rate_rank": conv_rate_rank,
                "overall_score": (final_cost_rank + conv_rate_rank) / 2,
                "total_engines_compared": len(engines),
            }

        return rankings

    def _analyze_convergence_details(self, engine) -> Dict[str, Any]:
        """Analyze detailed convergence patterns from convergence_monitor"""
        convergence_data = {
            "has_convergence_history": False,
            "convergence_events": [],
            "belief_convergence_pattern": [],
            "assignment_convergence_pattern": [],
        }

        if hasattr(engine.convergence_monitor, "convergence_history"):
            convergence_data["has_convergence_history"] = True
            history = engine.convergence_monitor.convergence_history

            for i, entry in enumerate(history):
                if isinstance(entry, dict):
                    convergence_data["convergence_events"].append(
                        {
                            "iteration": i,
                            "belief_converged": entry.get("belief_converged", False),
                            "assignment_converged": entry.get(
                                "assignment_converged", False
                            ),
                            "details": entry,
                        }
                    )

        return convergence_data

    def _analyze_cost_progression(self, costs: List[float]) -> Dict[str, Any]:
        """Analyze cost progression patterns"""
        if not costs:
            return {"has_cost_data": False}

        costs_array = np.array(costs)

        # Calculate derivatives to find improvement patterns
        if len(costs) > 1:
            cost_diff = np.diff(costs_array)
            improvements = cost_diff < 0

            analysis = {
                "has_cost_data": True,
                "total_iterations": len(costs),
                "initial_cost": float(costs[0]),
                "final_cost": float(costs[-1]),
                "total_improvement": float(costs[0] - costs[-1]),
                "improvement_rate": float((costs[0] - costs[-1]) / costs[0])
                if costs[0] != 0
                else 0.0,
                "iterations_with_improvement": int(np.sum(improvements)),
                "improvement_percentage": float(
                    np.sum(improvements) / len(cost_diff) * 100
                ),
                "largest_single_improvement": float(np.min(cost_diff))
                if len(cost_diff) > 0
                else 0.0,
                "average_improvement_per_step": float(np.mean(cost_diff[improvements]))
                if np.any(improvements)
                else 0.0,
                "cost_variance": float(np.var(costs)),
                "cost_std": float(np.std(costs)),
            }
        else:
            analysis = {
                "has_cost_data": True,
                "total_iterations": 1,
                "initial_cost": float(costs[0]),
                "final_cost": float(costs[0]),
                "total_improvement": 0.0,
                "improvement_rate": 0.0,
                "iterations_with_improvement": 0,
                "improvement_percentage": 0.0,
                "largest_single_improvement": 0.0,
                "average_improvement_per_step": 0.0,
                "cost_variance": 0.0,
                "cost_std": 0.0,
            }

        return analysis

    def _analyze_message_patterns(self, history) -> Dict[str, Any]:
        """Analyze message passing patterns from BCT history"""
        if not hasattr(history, "step_messages") or not history.step_messages:
            return {"has_message_data": False}

        total_messages = sum(
            len(messages) for messages in history.step_messages.values()
        )
        unique_flows = set()

        for step_messages in history.step_messages.values():
            for msg_data in step_messages:
                flow = f"{msg_data.sender}->{msg_data.recipient}"
                unique_flows.add(flow)

        analysis = {
            "has_message_data": True,
            "total_message_count": total_messages,
            "unique_message_flows": len(unique_flows),
            "average_messages_per_step": total_messages / len(history.step_messages)
            if history.step_messages
            else 0,
            "message_flow_list": list(unique_flows),
        }

        return analysis

    def _save_simulator_csv(self, analysis_data: SimulatorAnalysisData, filepath: str):
        """Save simulator analysis summary as CSV"""
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Header
            writer.writerow(
                [
                    "Engine",
                    "Total_Runs",
                    "Avg_Final_Cost",
                    "Std_Final_Cost",
                    "Min_Final_Cost",
                    "Max_Final_Cost",
                    "Convergence_Rate",
                    "Avg_Convergence_Time",
                    "Improvement_Rate",
                    "Performance_Rank",
                ]
            )

            # Data rows
            for engine_name in analysis_data.engine_stats.keys():
                stats = analysis_data.engine_stats[engine_name]
                conv = analysis_data.convergence_analysis[engine_name]
                perf = analysis_data.performance_comparison.get(engine_name, {})

                writer.writerow(
                    [
                        engine_name,
                        stats["total_runs"],
                        round(stats["average_final_cost"], 4),
                        round(stats["std_final_cost"], 4),
                        round(stats["min_final_cost"], 4),
                        round(stats["max_final_cost"], 4),
                        round(conv["convergence_rate"], 4),
                        round(conv["average_convergence_time"], 2),
                        round(conv["average_improvement_rate"], 4),
                        perf.get("overall_score", "N/A"),
                    ]
                )

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types and other non-serializable objects to JSON-compatible types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return self._make_json_serializable(obj.__dict__)
        return obj


# Convenience functions for backward compatibility and easy use
def save_simulator_comprehensive_analysis(
    simulator, filepath: Optional[str] = None, save_csv: bool = True
) -> str:
    """
    Convenience function to save comprehensive simulator analysis.

    Args:
        simulator: Simulator instance with results
        filepath: Optional custom filepath
        save_csv: Whether to also save CSV summary

    Returns:
        Path to saved JSON file
    """
    saver = EnhancedSaveModule()
    return saver.save_simulator_analysis(simulator, filepath, save_csv)


def save_enhanced_engine_analysis(
    engine,
    filepath: Optional[str] = None,
    include_performance: bool = True,
    include_convergence_detail: bool = True,
) -> str:
    """
    Convenience function to save enhanced single engine analysis.

    Args:
        engine: BPEngine instance
        filepath: Optional custom filepath
        include_performance: Include performance data
        include_convergence_detail: Include detailed convergence analysis

    Returns:
        Path to saved JSON file
    """
    saver = EnhancedSaveModule()
    return saver.save_enhanced_engine_data(
        engine, filepath, include_performance, include_convergence_detail
    )
