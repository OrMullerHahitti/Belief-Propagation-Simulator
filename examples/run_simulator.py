import numpy as np
import random
import time
import multiprocessing as mp

from propflow.simulator import Simulator
from propflow.utils import FGBuilder
from propflow.configs import CTFactory
from propflow.bp.engines import (
    BPEngine,
    DampingSCFGEngine,
    DampingEngine,
)

SEED = 42

if __name__ == "__main__":
    # Set multiprocessing start method for clean process spawning
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    # Initialize random seeds
    np.random.seed(SEED)
    random.seed(SEED)

    # --- Configuration ---
    NUM_GRAPHS = 10
    MAX_ITER = 5000
    LOG_LEVEL = "INFORMATIVE"  # Options: 'VERBOSE', 'MILD', 'INFORMATIVE', 'HIGH'

    # Engine configurations
    engine_configs = {
        "BPEngine": {"class": BPEngine},
        "DampingSCFGEngine_asymmetric": {
            "class": DampingSCFGEngine,
            "damping_factor": 0.9,
            "split_factor": 0.6,
        },
        "DampingEngine": {"class": DampingEngine, "damping_factor": 0.9},
    }

    # --- Graph Creation ---
    print(f"[{time.strftime('%H:%M:%S')}] Creating {NUM_GRAPHS} factor graphs...")
    ct_factory_fn = CTFactory.random_int.fn
    random_fg = [
        FGBuilder.build_random_graph(
            num_vars=50,
            domain_size=10,
            ct_factory=ct_factory_fn,
            ct_params={"low": 100, "high": 200},
            density=0.25,
        )
        for _ in range(NUM_GRAPHS)
    ]
    print(f"[{time.strftime('%H:%M:%S')}] Created {len(random_fg)} factor graphs.")

    # --- Simulation ---
    # Instantiate the simulator
    simulator = Simulator(engine_configs, log_level=LOG_LEVEL)

    # Run all simulations
    results = simulator.run_simulations(random_fg, max_iter=MAX_ITER)

    # --- Plotting ---
    if results:
        # Regular plot
        simulator.plot_results(max_iter=MAX_ITER, verbose=False)
        # Verbose plot      simulator.plot_results(max_iter=MAX_ITER, verbose=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] No results to plot.")

    print(f"[{time.strftime('%H:%M:%S')}] Main script finished.")
