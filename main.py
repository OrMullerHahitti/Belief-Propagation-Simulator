from pathlib import Path
import sys

import numpy as np
import random
import time
import multiprocessing as mp

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from propflow.simulator import Simulator
from propflow.utils.fg_utils import FGBuilder
from propflow.configs import CTFactories
from propflow.configs.global_config_mapping import (

    get_ct_factory,
    SIMULATOR_DEFAULTS,
    POLICY_DEFAULTS,
    ENGINE_DEFAULTS,
)
from propflow.bp.engines import (
    BPEngine,
    DampingSCFGEngine,
    DampingEngine,
    DampingCROnceEngine,
    CostReductionOnceEngine,
    SplitEngine,
)

SEED = 42

if __name__ == "__main__":
    # Set multiprocessing start method for clean process spawning
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Initialize random seeds
    np.random.seed(SEED)
    random.seed(SEED)

    # --- Configuration (uses centralized defaults, override as needed) ---
    NUM_GRAPHS = 10
    MAX_ITER = ENGINE_DEFAULTS["max_iterations"]  # Can override: MAX_ITER = 2000
    LOG_LEVEL = SIMULATOR_DEFAULTS[
        "default_log_level"
    ]  # Can override: LOG_LEVEL = "HIGH"

    # Engine configurations (using centralized defaults with explicit overrides when needed)
    engine_configs = {
        "BPEngine": {"class": BPEngine},
        "DampingSCFGEngine_symmetric": {
            "class": DampingSCFGEngine,
            "damping_factor": POLICY_DEFAULTS[
                "damping_factor"
            ],  # Can override: "damping_factor": 0.8,
            "split_factor": POLICY_DEFAULTS[
                "split_factor"
            ],  # Can override: "split_factor": 0.3,
        },
        "Split_0.5": {
            "class": SplitEngine,
            "split_factor": POLICY_DEFAULTS[
                "split_factor"
            ],  # Can override: "split_factor": 0.7,
        },
    }
    # --- Graph Creation ---
    print(f"[{time.strftime('%H:%M:%S')}] Creating {NUM_GRAPHS} factor graphs...")
    ct_factory_fn = CTFactories.RANDOM_INT.value  # or: get_ct_factory(CTFactories.RANDOM_INT)
    random_fg = [
        FGBuilder.build_random_graph(
            num_vars=50,
            domain_size=10,
            ct_factory=ct_factory_fn,
            ct_params={"low": 100, "high": 200},
            density=0.2,
        )
        for _ in range(NUM_GRAPHS)
    ]
    print(f"[{time.strftime('%H:%M:%S')}] Created {len(random_fg)} factor graphs.")

    # --- Simulation ---
    # Instantiate the simulator
    simulator = Simulator(engine_configs, log_level=LOG_LEVEL)

    # Run all simulations
    start_time = time.time()
    results = simulator.run_simulations(random_fg, max_iter=MAX_ITER)
    end_time = time.time()
    print(f"The total simulation time was {end_time - start_time:.2f} seconds.")

    # --- Plotting ---
    if results:
        simulator.plot_results(max_iter=MAX_ITER, verbose=False)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] No results to plot.")

    print(f"[{time.strftime('%H:%M:%S')}] Main script finished.")
