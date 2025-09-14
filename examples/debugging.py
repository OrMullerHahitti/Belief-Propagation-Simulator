import cProfile
import pickle
import pstats
import time
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import numpy as np
import random
import matplotlib.pyplot as plt
import os

from propflow.bp.engine_base import BPEngine
from propflow.bp.engines import (
    DampingEngine,
    DampingSCFGEngine,
)
from propflow.utils import FGBuilder
from propflow.configs import CTFactory
from propflow.utils import find_project_root
from propflow.policies import ConvergenceConfig

# Module-level constants (safe for child processes)
PROJECT_ROOT = find_project_root()
SEED = 42
ct_factory_fn = CTFactory.random_int.fn


def profiling(func):
    """
    Decorator to profile a function using cProfile.
    """

    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("time").print_stats(10)
        return result

    return wrapper


def run_single_simulation(args):
    """
    Run a single simulation - HEAVY DEBUG VERSION
    """
    import os
    import sys
    import time as time_module
    import threading
    import psutil

    pid = os.getpid()
    tid = threading.get_ident()

    print(
        f"[{time.strftime('%H:%M:%S')}] PROCESS {pid} (thread {tid}): Starting simulation"
    )
    sys.stdout.flush()

    # CRITICAL: Disable all threading in child processes to prevent deadlocks
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    print(f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Thread limiting set")
    sys.stdout.flush()

    try:
        graph_index, engine_name, config, graph_data, max_iter = args
        print(
            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Args unpacked - graph {graph_index}, engine {engine_name}"
        )
        sys.stdout.flush()

        # Check memory before unpickling
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        print(
            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Memory before unpickling: {memory_before:.1f} MB"
        )
        sys.stdout.flush()

        # Recreate graph from pickled data
        print(
            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Starting pickle.loads()..."
        )
        sys.stdout.flush()
        start_time = time_module.time()

        fg_copy = pickle.loads(graph_data)

        unpickle_time = time_module.time() - start_time
        memory_after_unpickle = process.memory_info().rss / 1024 / 1024  # MB
        print(
            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: pickle.loads() completed in {unpickle_time:.2f}s, memory: {memory_after_unpickle:.1f} MB"
        )
        sys.stdout.flush()

        # Check for any child processes
        children = process.children()
        print(
            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Child processes after unpickling: {len(children)}"
        )
        sys.stdout.flush()

        # Instantiate the engine
        print(
            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Creating engine {engine_name}..."
        )
        sys.stdout.flush()
        start_time = time_module.time()

        engine_class = config["class"]
        engine_params = {k: v for k, v in config.items() if k != "class"}
        engine = engine_class(
            factor_graph=fg_copy,
            convergence_config=ConvergenceConfig(),
            **engine_params,
        )

        engine_creation_time = time_module.time() - start_time
        memory_after_engine = process.memory_info().rss / 1024 / 1024  # MB
        children = process.children()
        print(
            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Engine created in {engine_creation_time:.2f}s, memory: {memory_after_engine:.1f} MB, children: {len(children)}"
        )
        sys.stdout.flush()

        # Run simulation with ITERATION-LEVEL DEBUGGING
        print(
            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Starting engine.run() with max_iter={max_iter}..."
        )
        sys.stdout.flush()
        start_time = time_module.time()

        # Add iteration monitoring by wrapping the run method
        original_run = engine.run
        iteration_count = [0]  # Use list for mutable reference

        def monitored_run(*args, **kwargs):
            print(
                f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: engine.run() called with args={args}, kwargs={kwargs}"
            )
            sys.stdout.flush()

            # Check if engine has a step method we can monitor
            if hasattr(engine, "step"):
                original_step = engine.step

                def monitored_step(*step_args, **step_kwargs):
                    iteration_count[0] += 1
                    current_iter = iteration_count[0]

                    if (
                        current_iter % 50 == 0 or current_iter <= 10
                    ):  # Log every 50 iterations + first 10
                        print(
                            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Iteration {current_iter}/{max_iter}"
                        )
                        sys.stdout.flush()

                    if current_iter > max_iter + 100:  # Safety check
                        print(
                            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: ERROR - Exceeded max_iter by 100! Force stopping."
                        )
                        sys.stdout.flush()
                        raise RuntimeError(
                            f"Iteration count {current_iter} exceeded max_iter {max_iter}"
                        )

                    return original_step(*step_args, **step_kwargs)

                engine.step = monitored_step

            result = original_run(*args, **kwargs)
            print(
                f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: engine.run() completed normally after {iteration_count[0]} iterations"
            )
            sys.stdout.flush()
            return result

        engine.run = monitored_run
        engine.run(max_iter=max_iter)

        run_time = time_module.time() - start_time
        memory_final = process.memory_info().rss / 1024 / 1024  # MB
        children = process.children()
        print(
            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: engine.run() completed in {run_time:.2f}s, memory: {memory_final:.1f} MB, children: {len(children)}"
        )
        sys.stdout.flush()

        # Get final cost
        final_cost = engine.history.costs[-1] if engine.history.costs else "NO_COSTS"
        print(f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Final cost: {final_cost}")
        sys.stdout.flush()

        print(f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Creating return tuple...")
        sys.stdout.flush()

        result = (graph_index, engine_name, engine.history.costs)

        print(
            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: SUCCESS - Returning result for graph {graph_index + 1} with {engine_name}"
        )
        sys.stdout.flush()

        return result

    except Exception as e:
        import traceback

        print(f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: EXCEPTION OCCURRED!")
        print(
            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Exception type: {type(e).__name__}"
        )
        print(
            f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Exception message: {str(e)}"
        )
        print(f"[{time.strftime('%H:%M:%S')}] PROCESS {pid}: Traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        raise


def run_batch_safe(simulation_args, max_workers=None):
    """
    Run simulations with HEAVY DEBUGGING and deadlock prevention
    """
    if max_workers is None:
        max_workers = cpu_count()

    print(
        f"[{time.strftime('%H:%M:%S')}] MAIN: run_batch_safe called with {len(simulation_args)} args, max_workers={max_workers}"
    )
    print(f"[{time.strftime('%H:%M:%S')}] MAIN: Available CPU cores: {cpu_count()}")
    print(f"[{time.strftime('%H:%M:%S')}] MAIN: Current process PID: {os.getpid()}")

    # Check main process memory
    import psutil

    main_process = psutil.Process()
    main_memory = main_process.memory_info().rss / 1024 / 1024  # MB
    print(
        f"[{time.strftime('%H:%M:%S')}] MAIN: Main process memory: {main_memory:.1f} MB"
    )

    # Strategy 1: Try full multiprocessing first
    try:
        print(
            f"[{time.strftime('%H:%M:%S')}] MAIN: Attempting full multiprocessing with {max_workers} processes..."
        )
        print(f"[{time.strftime('%H:%M:%S')}] MAIN: Creating Pool...")

        with Pool(processes=max_workers) as pool:
            print(f"[{time.strftime('%H:%M:%S')}] MAIN: Pool created successfully")
            print(f"[{time.strftime('%H:%M:%S')}] MAIN: Starting map_async...")

            result = pool.map_async(run_single_simulation, simulation_args)
            print(
                f"[{time.strftime('%H:%M:%S')}] MAIN: map_async started, waiting for results with 600s timeout..."
            )

            # Wait with progress updates
            start_time = time.time()
            timeout = 600
            check_interval = 30  # Check every 30 seconds

            while True:
                try:
                    all_results = result.get(timeout=check_interval)
                    break  # Success!
                except mp.TimeoutError:
                    elapsed = time.time() - start_time
                    print(
                        f"[{time.strftime('%H:%M:%S')}] MAIN: Still waiting... {elapsed:.0f}s elapsed, {timeout - elapsed:.0f}s remaining"
                    )
                    if elapsed >= timeout:
                        raise mp.TimeoutError("Final timeout reached")

        total_time = time.time() - start_time
        print(
            f"[{time.strftime('%H:%M:%S')}] MAIN: SUCCESS - Full multiprocessing completed in {total_time:.2f}s"
        )
        print(f"[{time.strftime('%H:%M:%S')}] MAIN: Got {len(all_results)} results")
        return all_results

    except Exception as e:
        elapsed = time.time() - start_time if "start_time" in locals() else 0
        print(
            f"[{time.strftime('%H:%M:%S')}] MAIN: FULL MULTIPROCESSING FAILED after {elapsed:.2f}s"
        )
        print(f"[{time.strftime('%H:%M:%S')}] MAIN: Exception type: {type(e).__name__}")
        print(f"[{time.strftime('%H:%M:%S')}] MAIN: Exception message: {str(e)}")

        import traceback

        print(f"[{time.strftime('%H:%M:%S')}] MAIN: Full traceback:")
        traceback.print_exc()

        print(f"[{time.strftime('%H:%M:%S')}] MAIN: Trying batch processing...")

        # Strategy 2: Batch processing with reduced workers
        return run_in_batches(simulation_args, max_workers=max(1, max_workers // 2))


def run_in_batches(simulation_args, batch_size=None, max_workers=None):
    """
    Process simulations in batches for optimal performance and reliability
    """
    if max_workers is None:
        max_workers = min(6, cpu_count())  # Optimal worker count

    if batch_size is None:
        # Optimal batch sizing: 2x the number of workers
        batch_size = max(8, max_workers * 2)

    print(f"[{time.strftime('%H:%M:%S')}] BATCH: Starting batch processing")
    print(
        f"[{time.strftime('%H:%M:%S')}] BATCH: Total simulations: {len(simulation_args)}"
    )
    print(f"[{time.strftime('%H:%M:%S')}] BATCH: Batch size: {batch_size}")
    print(f"[{time.strftime('%H:%M:%S')}] BATCH: Max workers per batch: {max_workers}")

    all_results = []
    num_batches = (len(simulation_args) + batch_size - 1) // batch_size

    print(f"[{time.strftime('%H:%M:%S')}] BATCH: Will process {num_batches} batches")

    for i in range(0, len(simulation_args), batch_size):
        batch = simulation_args[i : i + batch_size]
        batch_num = i // batch_size + 1

        print(
            f"[{time.strftime('%H:%M:%S')}] BATCH {batch_num}/{num_batches}: Starting ({len(batch)} simulations)"
        )

        try:
            workers_for_batch = min(max_workers, len(batch))
            print(
                f"[{time.strftime('%H:%M:%S')}] BATCH {batch_num}: Using {workers_for_batch} workers"
            )

            with Pool(processes=workers_for_batch) as pool:
                start_time = time.time()
                batch_results = pool.map(run_single_simulation, batch)
                batch_time = time.time() - start_time

                print(
                    f"[{time.strftime('%H:%M:%S')}] BATCH {batch_num}: Completed in {batch_time:.2f}s"
                )

            all_results.extend(batch_results)
            print(
                f"[{time.strftime('%H:%M:%S')}] BATCH {batch_num}: Added {len(batch_results)} results, total so far: {len(all_results)}"
            )

        except Exception as e:
            print(
                f"[{time.strftime('%H:%M:%S')}] BATCH {batch_num}: FAILED - {type(e).__name__}: {str(e)}"
            )

            # Fallback to sequential for this batch
            print(
                f"[{time.strftime('%H:%M:%S')}] BATCH {batch_num}: Running sequentially as fallback..."
            )

            for j, args in enumerate(batch):
                try:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] BATCH {batch_num}: Sequential {j + 1}/{len(batch)}"
                    )
                    result = run_single_simulation(args)
                    all_results.append(result)
                except Exception as seq_e:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] BATCH {batch_num}: Sequential item {j + 1} FAILED: {seq_e}"
                    )

    print(
        f"[{time.strftime('%H:%M:%S')}] BATCH: All batches completed, total results: {len(all_results)}"
    )
    return all_results


def sequential_fallback(simulation_args):
    """
    Last resort: run everything sequentially
    """
    print(f"[{time.strftime('%H:%M:%S')}] Running all simulations sequentially...")
    return [run_single_simulation(args) for args in simulation_args]


if __name__ == "__main__":
    # Set multiprocessing start method for clean process spawning
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    # Initialize random seeds (only in main process)
    np.random.seed(SEED)
    random.seed(SEED)
    print(f"[{time.strftime('%H:%M:%S')}] Using seed: {SEED}")
    print(f"[{time.strftime('%H:%M:%S')}] Available CPU cores: {cpu_count()}")

    # Create random factor graphs
    print(f"[{time.strftime('%H:%M:%S')}] Creating factor graphs...")
    random_fg = []
    for i in range(10):
        random_fg.append(
            FGBuilder.build_random_graph(
                num_vars=50,
                domain_size=10,
                ct_factory=ct_factory_fn,
                ct_params={"low": 100, "high": 200},
                density=0.25,
            )
        )

    print(f"[{time.strftime('%H:%M:%S')}] Created {len(random_fg)} factor graphs")

    # Engine configurations
    max_iter = 5000
    engine_configs = {
        "BPEngine": {"class": BPEngine},
        "DampingSCFGEngine_asymmetric": {
            "class": DampingSCFGEngine,
            "damping_factor": 0.9,
            "split_factor": 0.6,
        },
        "DampingEngine": {"class": DampingEngine, "damping_factor": 0.9},
    }

    # Initialize results dictionary
    results = {engine_name: [] for engine_name in engine_configs.keys()}

    # Prepare all simulation arguments
    simulation_args = []
    for i, graph in enumerate(random_fg):
        # Pickle the graph once for all bp
        graph_data = pickle.dumps(graph)
        for engine_name, config in engine_configs.items():
            args = (i, engine_name, config, graph_data, max_iter)
            simulation_args.append(args)

    total_simulations = len(simulation_args)
    print(
        f"[{time.strftime('%H:%M:%S')}] Prepared {total_simulations} total simulations ({len(random_fg)} graphs × {len(engine_configs)} engines)"
    )

    # Run simulations with maximum CPU usage and deadlock safety
    start_time = time.time()

    print(f"[{time.strftime('%H:%M:%S')}] MAIN: About to start run_batch_safe...")
    print(
        f"[{time.strftime('%H:%M:%S')}] MAIN: Multiprocessing start method: {mp.get_start_method()}"
    )

    # Add system info
    import psutil

    print(
        f"[{time.strftime('%H:%M:%S')}] MAIN: System memory: {psutil.virtual_memory().total / 1024 ** 3:.1f} GB"
    )
    print(
        f"[{time.strftime('%H:%M:%S')}] MAIN: Available memory: {psutil.virtual_memory().available / 1024 ** 3:.1f} GB"
    )

    try:
        all_results = run_batch_safe(simulation_args, max_workers=cpu_count())
    except Exception as e:
        print(
            f"[{time.strftime('%H:%M:%S')}] MAIN: CRITICAL ERROR - All multiprocessing strategies failed!"
        )
        print(f"[{time.strftime('%H:%M:%S')}] MAIN: Exception type: {type(e).__name__}")
        print(f"[{time.strftime('%H:%M:%S')}] MAIN: Exception message: {str(e)}")
        import traceback

        traceback.print_exc()

        print(
            f"[{time.strftime('%H:%M:%S')}] MAIN: Falling back to sequential processing..."
        )
        all_results = sequential_fallback(simulation_args)

    end_time = time.time()
    total_time = end_time - start_time

    print(
        f"[{time.strftime('%H:%M:%S')}] All simulations completed in {total_time:.2f} seconds"
    )
    print(
        f"[{time.strftime('%H:%M:%S')}] Average time per simulation: {total_time / total_simulations:.2f} seconds"
    )

    # Verify we got all results
    if len(all_results) != total_simulations:
        print(
            f"[{time.strftime('%H:%M:%S')}] WARNING: Expected {total_simulations} results, got {len(all_results)}"
        )

    # Reorganize results back into the original structure
    print(f"[{time.strftime('%H:%M:%S')}] Organizing results...")
    for graph_index, engine_name, costs in all_results:
        results[engine_name].append(costs)

    # Verify results organization
    for engine_name, costs_list in results.items():
        print(
            f"[{time.strftime('%H:%M:%S')}] {engine_name}: {len(costs_list)} runs completed"
        )

    print(f"[{time.strftime('%H:%M:%S')}] Starting plotting...")

    # --- Plotting Logic ---
    plt.figure(figsize=(12, 8))
    colors = ["blue", "red", "green", "orange", "purple"]

    for idx, (engine_name, costs_list) in enumerate(results.items()):
        if not costs_list:
            print(
                f"[{time.strftime('%H:%M:%S')}] WARNING: No results for {engine_name}"
            )
            continue

        # Find the maximum length across all runs for this engine
        max_len = max(max(max_iter, len(c)) for c in costs_list)

        # Pad shorter cost lists to the max length for correct averaging
        padded_costs = [c + [c[-1]] * (max_len - len(c)) for c in costs_list]

        # Convert to a NumPy array and calculate the average across runs (axis=0)
        avg_costs = np.average(np.array(padded_costs), axis=0)

        # Plot with different colors
        color = colors[idx % len(colors)]
        plt.plot(avg_costs, label=engine_name, color=color, linewidth=2)

        print(
            f"[{time.strftime('%H:%M:%S')}] Plotted {engine_name}: avg final cost = {avg_costs[-1]:.2f}"
        )

    plt.title(
        f"Average Costs over {len(random_fg)} Runs on Random Factor Graphs (Density 0.25)\n"
        f"Total runtime: {total_time:.1f}s using {cpu_count()} CPU cores",
        fontsize=14,
    )
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Average Cost", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add performance info as text
    plt.figtext(
        0.02,
        0.02,
        f"Simulations: {total_simulations} | "
        f"CPU cores used: {cpu_count()} | "
        f"Avg time/sim: {total_time / total_simulations:.2f}s",
        fontsize=8,
        ha="left",
    )

    print(f"[{time.strftime('%H:%M:%S')}] Displaying plot...")
    plt.tight_layout()
    plt.show()

    print(f"[{time.strftime('%H:%M:%S')}] Simulation completed successfully!")
