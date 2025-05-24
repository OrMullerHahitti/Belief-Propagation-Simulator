import time
import psutil
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Metrics for a single BP step."""

    step_number: int
    duration: float
    message_count: int
    avg_message_size: float
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0


@dataclass
class CycleMetrics:
    """Metrics for a complete BP cycle."""

    cycle_number: int
    total_duration: float
    steps: List[StepMetrics]
    belief_change: Optional[float] = None
    cost: Optional[float] = None


class PerformanceMonitor:
    """Monitor and report BP performance metrics."""

    def __init__(self, track_memory: bool = True, track_cpu: bool = True):
        self.step_metrics: List[StepMetrics] = []
        self.cycle_metrics: List[CycleMetrics] = []
        self.track_memory = track_memory
        self.track_cpu = track_cpu
        self.process = psutil.Process() if (track_memory or track_cpu) else None
        self._cycle_start_time: Optional[float] = None
        self._cycle_steps: List[StepMetrics] = []

    def start_step(self) -> float:
        """Mark start of step."""
        return time.time()

    def end_step(self, start_time: float, step_num: int, messages: List) -> StepMetrics:
        """Record step metrics."""
        duration = time.time() - start_time

        # Calculate message statistics
        message_count = len(messages) if messages else 0
        if messages and hasattr(messages[0], "data"):
            avg_size = np.mean([msg.data.size for msg in messages])
        else:
            avg_size = 0.0

        # Get system metrics
        memory_mb = 0.0
        cpu_percent = 0.0

        if self.process:
            if self.track_memory:
                try:
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                except:
                    pass

            if self.track_cpu:
                try:
                    cpu_percent = self.process.cpu_percent(interval=0.1)
                except:
                    pass

        # Create metrics
        metrics = StepMetrics(
            step_number=step_num,
            duration=duration,
            message_count=message_count,
            avg_message_size=avg_size,
            memory_usage_mb=memory_mb,
            cpu_percent=cpu_percent,
        )

        self.step_metrics.append(metrics)
        self._cycle_steps.append(metrics)

        logger.debug(
            f"Step {step_num}: {duration:.3f}s, {message_count} messages, "
            f"memory: {memory_mb:.1f}MB"
        )

        return metrics

    def start_cycle(self, cycle_num: int):
        """Mark start of a new cycle."""
        self._cycle_start_time = time.time()
        self._cycle_steps = []
        logger.debug(f"Starting cycle {cycle_num}")

    def end_cycle(
        self,
        cycle_num: int,
        belief_change: Optional[float] = None,
        cost: Optional[float] = None,
    ) -> CycleMetrics:
        """Record cycle metrics."""
        if self._cycle_start_time is None:
            logger.warning("end_cycle called without start_cycle")
            return None

        duration = time.time() - self._cycle_start_time

        metrics = CycleMetrics(
            cycle_number=cycle_num,
            total_duration=duration,
            steps=self._cycle_steps.copy(),
            belief_change=belief_change,
            cost=cost,
        )

        self.cycle_metrics.append(metrics)
        self._cycle_start_time = None

        logger.info(
            f"Cycle {cycle_num}: {duration:.3f}s, {len(self._cycle_steps)} steps, "
            f"cost: {cost:.2f}"
            if cost
            else ""
        )

        return metrics

    def get_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        if not self.step_metrics:
            return {}

        step_durations = [m.duration for m in self.step_metrics]
        message_counts = [m.message_count for m in self.step_metrics]
        memory_usages = [
            m.memory_usage_mb for m in self.step_metrics if m.memory_usage_mb > 0
        ]

        summary = {
            "total_steps": len(self.step_metrics),
            "total_time": sum(step_durations),
            "avg_step_time": np.mean(step_durations),
            "max_step_time": max(step_durations),
            "min_step_time": min(step_durations),
            "std_step_time": np.std(step_durations),
            "total_messages": sum(message_counts),
            "avg_messages_per_step": np.mean(message_counts),
        }

        if memory_usages:
            summary.update(
                {
                    "avg_memory_mb": np.mean(memory_usages),
                    "max_memory_mb": max(memory_usages),
                    "min_memory_mb": min(memory_usages),
                }
            )

        if self.cycle_metrics:
            cycle_durations = [c.total_duration for c in self.cycle_metrics]
            summary.update(
                {
                    "total_cycles": len(self.cycle_metrics),
                    "avg_cycle_time": np.mean(cycle_durations),
                    "max_cycle_time": max(cycle_durations),
                }
            )

        return summary

    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot performance metrics over time."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Step durations
            step_nums = [m.step_number for m in self.step_metrics]
            step_times = [m.duration for m in self.step_metrics]
            axes[0, 0].plot(step_nums, step_times)
            axes[0, 0].set_xlabel("Step")
            axes[0, 0].set_ylabel("Duration (s)")
            axes[0, 0].set_title("Step Duration Over Time")

            # Message counts
            msg_counts = [m.message_count for m in self.step_metrics]
            axes[0, 1].plot(step_nums, msg_counts)
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].set_ylabel("Message Count")
            axes[0, 1].set_title("Messages per Step")

            # Memory usage
            if self.track_memory:
                memory_usage = [m.memory_usage_mb for m in self.step_metrics]
                axes[1, 0].plot(step_nums, memory_usage)
                axes[1, 0].set_xlabel("Step")
                axes[1, 0].set_ylabel("Memory (MB)")
                axes[1, 0].set_title("Memory Usage")

            # Cycle costs
            if self.cycle_metrics and any(
                c.cost is not None for c in self.cycle_metrics
            ):
                cycle_nums = [
                    c.cycle_number for c in self.cycle_metrics if c.cost is not None
                ]
                costs = [c.cost for c in self.cycle_metrics if c.cost is not None]
                axes[1, 1].plot(cycle_nums, costs)
                axes[1, 1].set_xlabel("Cycle")
                axes[1, 1].set_ylabel("Cost")
                axes[1, 1].set_title("Cost per Cycle")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib not available for plotting")
