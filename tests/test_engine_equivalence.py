import numpy as np
import random
import pickle
import sys, os

import pytest

from debugging import create_factor_graph
from belief_propagation_simulator.bp_base.engines_realizations import SplitEngine, CostReductionOnceEngine
from belief_propagation_simulator.policies.convergance import ConvergenceConfig

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_split_vs_cost_reduction_once_equivalence():
    # Use fixed seed for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # Generate a random factor graph
    fg1 = create_factor_graph(
        graph_type="random",
        num_vars=5,
        domain_size=3,
        ct_factory="random_int",
        ct_params={"low": 1, "high": 10},
        density=0.4,
    )
    # Deep copy the graph for the second engine
    fg2 = pickle.loads(pickle.dumps(fg1))

    # Initialize engines with same configuration parameters
    config = ConvergenceConfig()
    engine2 = SplitEngine(
        factor_graph=fg1,
        convergence_config=config,
        monitor_performance=False,
        split_factor=0.5,
    )
    engine6 = CostReductionOnceEngine(
        factor_graph=fg2,
        convergence_config=config,
        monitor_performance=False,
        reduction_factor=0.5,
    )

    # Run both engines for the same number of iterations
    max_iter = 50
    engine2.run(max_iter=max_iter)
    engine6.run(max_iter=max_iter)

    # Compare the cost histories for equivalence
    costs2 = np.array(engine2.history.costs)
    costs6 = np.array(engine6.history.costs)
    assert costs2.shape == costs6.shape, "Cost histories length differ"
    assert np.allclose(
        costs2, costs6
    ), "Cost histories differ between SplitEngine and CostReductionOnceEngine"
