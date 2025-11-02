"""Integration coverage for the torch-based computator example."""

# Import numpy to build deterministic cost tables that mimic the example graph.
import numpy as np

# Import pytest so we can gracefully skip when torch is unavailable.
import pytest

# Ensure PyTorch is available before executing the integration test.
torch = pytest.importorskip("torch")

# Import the public engine and graph builder exactly as the example does.
from propflow import BPEngine, FGBuilder

# Import the reference Min-Sum computator to compare against the soft-min variant.
from propflow.bp.computators import MinSumComputator

# Import the torch-powered soft-min computator that the example showcases.
from propflow.nn.torch_computators import SoftMinTorchComputator


# Helper to construct the same cycle graph structure used in the example with deterministic costs.
def _build_cycle_graph(seed: int, num_vars: int = 4, domain: int = 3):
    # Create a dedicated RNG so each graph build yields identical tables when seeded the same way.
    rng = np.random.default_rng(seed)

    # Generate factor cost tables with stable values for every factor instantiation.
    def seeded_cost_table(n: int, domain_size: int, low: int = 0, high: int = 10):
        # Draw integer costs and cast to float so both computators receive the same numeric type.
        data = rng.integers(low=low, high=high, size=(domain_size,) * n).astype(float)
        return data

    # Build the cycle graph so the integration mirrors the example script.
    return FGBuilder.build_cycle_graph(
        num_vars=num_vars,
        domain_size=domain,
        ct_factory=seeded_cost_table,
        ct_params={"low": 0, "high": 10},
    )


# Verifies the example pathway by comparing soft-min torch assignments against Min-Sum on the same graph.
def test_softmin_example_matches_minsum_assignments():
    # Recreate the example graph twice so each engine operates on an independent copy.
    soft_graph = _build_cycle_graph(seed=7)
    minsum_graph = _build_cycle_graph(seed=7)

    # Instantiate the torch-based computator with a tiny temperature to approximate hard min-sum behavior.
    soft_computator = SoftMinTorchComputator(tau=1e-4, device="cpu")
    # Instantiate the classic Min-Sum computator for the reference engine.
    minsum_computator = MinSumComputator()

    # Drive the integration example through the public BPEngine without writing artifacts to disk.
    soft_engine = BPEngine(factor_graph=soft_graph, computator=soft_computator)
    minsum_engine = BPEngine(factor_graph=minsum_graph, computator=minsum_computator)

    # Run a few synchronous iterations, mirroring the example but disabling history dumps to keep tests isolated.
    soft_engine.run(max_iter=8, save_json=False, save_csv=False)
    minsum_engine.run(max_iter=8, save_json=False, save_csv=False)

    # Confirm both engines settle on identical variable assignments.
    assert soft_engine.assignments == minsum_engine.assignments

    # Validate that the resulting global costs match, ensuring the differentiable path preserves solution quality.
    assert np.isclose(soft_engine.graph.global_cost, minsum_engine.graph.global_cost)
