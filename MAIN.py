import numpy as np
import random
import time
import multiprocessing as mp

from simulator import Simulator
from utils.fg_utils import FGBuilder
from configs.global_config_mapping import CT_FACTORIES
from bp_base.engines_realizations import BPEngine, DampingSCFGEngine, DampingEngine
from utils.performance_optimizer import optimize_numpy_settings, MemoryMonitor

SEED = 42

if __name__ == '__main__':
    # Optimize performance settings
    optimize_numpy_settings()
    
    # Set multiprocessing start method for clean process spawning
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    # Initialize random seeds
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Initialize memory monitoring
    memory_monitor = MemoryMonitor(threshold_mb=2000)

    # --- Configuration ---
    NUM_GRAPHS = 10
    MAX_ITER = 2000
    LOG_LEVEL = 'INFORMATIVE'  # Options: 'VERBOSE', 'MILD', 'INFORMATIVE', 'HIGH' - Using efficient level

    # Engine configurations
    engine_configs = {
        "BPEngine": {"class": BPEngine},
        "DampingSCFGEngine_asymmetric": {"class": DampingSCFGEngine, "damping_factor": 0.9, "split_factor": 0.6},
        "DampingEngine": {"class": DampingEngine, "damping_factor": 0.9},
    }

    # --- Graph Creation ---
    print(f"[{time.strftime('%H:%M:%S')}] Creating {NUM_GRAPHS} factor graphs...")
    memory_monitor.check_memory("graph creation start")
    
    ct_factory_fn = CT_FACTORIES["random_int"]
    random_fg = [
        FGBuilder.build_random_graph(
            num_vars=50,
            domain_size=20,
            ct_factory=ct_factory_fn,
            ct_params={"low": 100, "high": 200},
            density=0.25,
        )
        for _ in range(NUM_GRAPHS)
    ]
    
    memory_monitor.check_memory("graph creation complete")
    print(f"[{time.strftime('%H:%M:%S')}] Created {len(random_fg)} factor graphs.")

    # --- Simulation ---
    # Instantiate the simulator
    simulator = Simulator(engine_configs, log_level=LOG_LEVEL)

    # Run all simulations
    results = simulator.run_simulations(random_fg, max_iter=MAX_ITER)

    # --- Plotting ---
    if results:
        memory_monitor.check_memory("plotting start")
        simulator.plot_results(max_iter=MAX_ITER,verbose=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] No results to plot.")

    # Final memory report
    memory_info = memory_monitor.get_memory_info()
    print(f"[{time.strftime('%H:%M:%S')}] Peak memory usage: {memory_info['peak_mb']:.1f}MB")
    print(f"[{time.strftime('%H:%M:%S')}] Main script finished.")
