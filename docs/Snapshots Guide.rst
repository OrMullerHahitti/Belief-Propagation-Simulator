.. _snapshots-guide:

PropFlow Snapshots: Comprehensive Guide
=======================================

.. contents:: Table of Contents
   :depth: 2

Overview
--------

**Snapshots** are periodic captures of the complete simulation state at each iteration of a belief propagation algorithm. They provide a detailed window into how messages, beliefs, and costs evolve throughout a run, enabling deep analysis of convergence dynamics, cycle behavior, and message-passing patterns.

Snapshots are:

- **Lightweight**: Stored efficiently in memory or on disk
- **Flexible**: Configurable to capture only the data you need
- **Composable**: Can be combined with analysis tools for convergence, Jacobian, and cycle metrics
- **Portable**: Can be persisted and reloaded for later analysis

Why Use Snapshots?
~~~~~~~~~~~~~~~~~~

- **Debug algorithm behavior**: Understand how beliefs and costs evolve step-by-step
- **Analyze convergence**: Detect when and how quickly the algorithm converges
- **Study message patterns**: Examine variable-to-factor and factor-to-variable messages
- **Investigate cycles**: Find feedback loops that affect convergence
- **Compare configurations**: Run the same problem with different engine settings and compare trajectories
- **Track assignments**: Monitor how variable assignments change over time

What Are Snapshots
------------------

A **snapshot** is a frozen view of the belief propagation algorithm's state at a single iteration. Unlike traditional "history" which stores only final results, snapshots capture:

- **Message values**: All Q (variable→factor) and R (factor→variable) messages
- **Runtime state**: Current variable assignments and beliefs
- **Graph structure**: Variable and factor neighbors
- **Metadata**: Damping factors, cost values, convergence metrics

Types of Information Captured
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    SnapshotRecord
    ├── SnapshotData (raw simulation state)
    │   ├── Messages: Q and R arrays
    │   ├── Beliefs/Assignments: Current variable values
    │   ├── Graph topology: Variable/factor neighborhoods
    │   ├── Costs: Global cost and factor cost functions
    │   └── Metadata: Engine config, timestamps, etc.
    ├── Jacobians (optional, if enabled)
    │   ├── Matrices A, P, B: Linearized message dependencies
    │   └── Block norms: Convergence certification metrics
    ├── CycleMetrics (optional, if enabled)
    │   ├── Cycle count: Total number of feedback loops
    │   ├── Aligned hops: Cycles amenable to contraction analysis
    │   └── Details: Per-cycle properties (if detailed analysis enabled)
    ├── Winners (optional)
    │   └── Factor-variable assignment preferences
    └── Min indices
         └── Argmin for each Q message

Snapshot Data Structure
-----------------------

SnapshotData Fields
~~~~~~~~~~~~~~~~~~~

The core snapshot data (captured at each step):

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Field
     - Type
     - Description
   * - ``step``
     - ``int``
     - Iteration number
   * - ``lambda_``
     - ``float``
     - Damping factor at this step
   * - ``dom``
     - ``Dict[str, List[str]]``
     - Variable domain labels: ``{var_name: ["0", "1", ...]}``
   * - ``N_var``
     - ``Dict[str, List[str]]``
     - Variable neighborhoods: ``{var: [factor_neighbors]}``
   * - ``N_fac``
     - ``Dict[str, List[str]]``
     - Factor neighborhoods: ``{factor: [variable_neighbors]}``
   * - ``Q``
     - ``Dict[(str, str), ndarray]``
     - Variable→factor messages: ``{(var, factor): [msg_values]}``
   * - ``R``
     - ``Dict[(str, str), ndarray]``
     - Factor→variable messages: ``{(factor, var): [msg_values]}``
   * - ``cost``
     - ``Dict[str, callable]``
     - Factor cost functions: ``{factor: lambda assignment: cost_value}``
   * - ``unary``
     - ``Dict[str, ndarray]``
     - Unary potential per variable (usually zeros)
   * - ``beliefs``
     - ``Dict[str, float]``
     - Current belief (min-sum value) per variable
   * - ``assignments``
     - ``Dict[str, int]``
     - Current assignment (argmin) per variable
   * - ``global_cost``
     - ``float`` (optional)
     - Total cost across all factors
   * - ``metadata``
     - ``Dict[str, Any]``
     - Additional info: engine type, convergence status, etc.

SnapshotRecord Fields
~~~~~~~~~~~~~~~~~~~~~

Wraps ``SnapshotData`` with optional analysis results:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Field
     - Type
     - Description
   * - ``data``
     - ``SnapshotData``
     - The raw captured state
   * - ``jacobians``
     - ``Jacobians`` (optional)
     - Linearized message dependencies (A, P, B matrices)
   * - ``cycles``
     - ``CycleMetrics`` (optional)
     - Cycle analysis results
   * - ``winners``
     - ``Dict`` (optional)
     - Winning assignments for factor-to-variable edges
   * - ``min_idx``
     - ``Dict`` (optional)
     - Argmin indices for Q messages
   * - ``captured_at``
     - ``datetime``
     - When this snapshot was recorded

Configuration
-------------

Enabling Snapshots
~~~~~~~~~~~~~~~~~~

To capture snapshots, pass a ``SnapshotsConfig`` to your engine:

.. code-block:: python

    from propflow import BPEngine, DampingEngine
    from propflow.snapshots import SnapshotsConfig

    # Configure what to capture
    snapshot_cfg = SnapshotsConfig(
        compute_jacobians=True,        # Compute Jacobian matrices A, P, B
        compute_block_norms=True,      # Compute infinity norms for convergence bounds
        compute_cycles=True,           # Analyze feedback cycles
        include_detailed_cycles=False, # Include per-cycle metrics (slower)
        compute_numeric_cycle_gain=False, # Estimate numeric gain per cycle (slower)
        max_cycle_len=12,              # Only find cycles up to length 12
        retain_last=25,                # Keep only the last 25 snapshots in memory
        save_each_step=False,          # Auto-save each snapshot to disk
        save_dir=None,                 # Directory for auto-save (required if save_each_step=True)
    )

    # Create engine with snapshot support
    engine = DampingEngine(
        factor_graph=graph,
        damping_factor=0.9,
        snapshots_config=snapshot_cfg,
        use_bct_history=True,  # Also enables belief/assignment tracking
    )

    # Run normally
    engine.run(max_iter=100)

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 12 15 53

   * - Option
     - Type
     - Default
     - Purpose
   * - ``compute_jacobians``
     - ``bool``
     - ``True``
     - Enable Jacobian matrix computation
   * - ``compute_block_norms``
     - ``bool``
     - ``True``
     - Compute convergence bound norms
   * - ``compute_cycles``
     - ``bool``
     - ``True``
     - Analyze feedback cycles
   * - ``include_detailed_cycles``
     - ``bool``
     - ``False``
     - Store per-cycle metrics (memory-intensive)
   * - ``compute_numeric_cycle_gain``
     - ``bool``
     - ``False``
     - Estimate numeric gain (slow)
   * - ``max_cycle_len``
     - ``int``
     - ``12``
     - Maximum cycle length to enumerate
   * - ``retain_last``
     - ``int`` or ``None``
     - ``25``
     - Keep N snapshots in memory; ``None`` = unlimited
   * - ``save_each_step``
     - ``bool``
     - ``True``
     - Auto-persist snapshots to disk
   * - ``save_dir``
     - ``str`` or ``Path`` or ``None``
     - ``None``
     - Directory for disk persistence

.. note::
   **Performance Tip**: If you only care about messages and beliefs (not analysis), disable ``compute_jacobians``, ``compute_block_norms``, and ``compute_cycles`` to speed up snapshot capture.

Capturing and Accessing Snapshots
----------------------------------

During Simulation
~~~~~~~~~~~~~~~~~

Snapshots are automatically captured at each iteration when configured:

.. code-block:: python

    engine = BPEngine(factor_graph=graph, snapshots_config=snapshot_cfg)
    engine.run(max_iter=100)

    # At this point, engine has captured up to 100 snapshots (or retain_last worth)

Accessing Snapshots After Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from propflow.snapshots.utils import (
        get_snapshot,
        latest_snapshot,
        latest_jacobians,
        latest_cycles,
        latest_winners,
    )

    # Get a specific step's snapshot
    snapshot_at_step_5 = get_snapshot(engine, 5)
    print(snapshot_at_step_5.data.step)        # 5
    print(snapshot_at_step_5.data.assignments) # {"x1": 0, "x2": 1, ...}
    print(snapshot_at_step_5.data.global_cost) # 42.5

    # Get the most recent snapshot
    latest = latest_snapshot(engine)
    print(latest.data.step)

    # Get analysis artifacts from the latest snapshot
    jac = latest_jacobians(engine)
    cycles = latest_cycles(engine)
    winners = latest_winners(engine)

Collecting All Snapshots
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Gather all snapshots captured during the run
    all_snapshots = [
        get_snapshot(engine, i)
        for i in range(len(engine.history.step_costs))
    ]

    print(f"Total snapshots: {len(all_snapshots)}")

Analyzing Snapshots
-------------------

1. Belief and Assignment Trajectories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track how variable beliefs and assignments evolve:

.. code-block:: python

    from propflow.snapshots import SnapshotAnalyzer

    # Create analyzer from snapshots
    analyzer = SnapshotAnalyzer(all_snapshots)

    # Get belief trajectories (argmin over messages)
    beliefs = analyzer.beliefs_per_variable()
    print(beliefs["x1"])  # [0, 0, 1, 1, 2, 2, ...] - assignment over time

    # Or manually extract from snapshots
    beliefs_manual = {}
    for var in ["x1", "x2"]:
        beliefs_manual[var] = [
            snap.data.assignments.get(var)
            for snap in all_snapshots
        ]

2. Convergence Analysis (BCT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze how each variable's belief evolved and when it converged:

.. code-block:: python

    from propflow.snapshots import SnapshotVisualizer

    visualizer = SnapshotVisualizer(all_snapshots)

    # Create and return a BCT creator object
    bct_creator = visualizer.plot_bct("x1", show=True)

    # Analyze convergence for a variable
    analysis = bct_creator.analyze_convergence("x1")
    print(f"Variable x1 converged: {analysis['converged']}")
    print(f"Final belief: {analysis['final_belief']}")
    print(f"Total change: {analysis['total_change']}")
    print(f"Convergence iteration: {analysis['convergence_iteration']}")

    # Compare multiple variables
    comparison = bct_creator.compare_variables(["x1", "x2", "x3"])
    print(comparison["summary"]["all_converged"])

    # Export detailed analysis
    bct_creator.export_analysis("bct_analysis.json")

3. Jacobian and Block Norms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examine linearized dynamics and convergence bounds:

.. code-block:: python

    from propflow.snapshots.utils import latest_jacobians

    # Get Jacobian for latest snapshot
    jac = latest_jacobians(engine)

    if jac:
        # Check convergence certification via block norms
        norms = jac.block_norms
        print(f"||BPA||_inf = {norms['||BPA||_inf']:.4f}")
        print(f"||B||_inf = {norms['||B||_inf']:.4f}")
        print(f"||PA||_inf = {norms['||PA||_inf']:.4f}")
        print(f"||M||_inf_upper = {norms['||M||_inf_upper']:.4f}")

        # If ||M||_inf_upper < 1.0, convergence is certified
        if norms['||M||_inf_upper'] < 1.0:
            print("✓ Convergence certified!")

        # Access raw matrices (sparse CSR format)
        print(jac.A.shape, jac.P.shape, jac.B.shape)

4. Cycle Analysis
~~~~~~~~~~~~~~~~~~~

Investigate feedback loops in the message-passing graph:

.. code-block:: python

    from propflow.snapshots.utils import latest_cycles

    cycles = latest_cycles(engine)

    if cycles:
        print(f"Total cycles: {cycles.num_cycles}")
        print(f"Cycles with aligned hops: {cycles.aligned_hops_total}")
        print(f"Contraction certified: {cycles.has_certified_contraction}")

        # Per-cycle details (if enabled in config)
        if cycles.details:
            for i, detail in enumerate(cycles.details[:5]):
                print(f"Cycle {i}: length={detail['length']}, aligned={detail['aligned']}")

5. Analysis Report
~~~~~~~~~~~~~~~~~~~

Generate a comprehensive summary:

.. code-block:: python

    from propflow.snapshots import AnalysisReport

    report = AnalysisReport(analyzer)

    # Get summary at a specific step
    summary_at_last = report.to_json(step_idx=len(all_snapshots) - 1)
    print(summary_at_last["block_norms"])
    print(summary_at_last["cycle_metrics"])

Visualizing Snapshots
---------------------

1. Belief Trajectories
~~~~~~~~~~~~~~~~~~~~~~

Plot how variable assignments evolve:

.. code-block:: python

    from propflow.snapshots import SnapshotVisualizer

    visualizer = SnapshotVisualizer(all_snapshots)

    # Get all variable names
    variables = visualizer.variables()
    print(f"Variables: {variables}")

    # Plot trajectories for a subset
    visualizer.plot_argmin_per_variable(
        vars_filter=variables[:6],
        figsize=(10, 12),
        show=True,
        savepath="belief_trajectories.png"
    )

2. Backtrack Cost Trees (BCT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize how costs backtrack through iterations:

.. code-block:: python

    # Plot BCT for a single variable
    bct_creator = visualizer.plot_bct(
        "x1",
        iteration=None,  # Use -1 (last iteration)
        show=True,
        savepath="bct_x1.png"
    )

    # The returned BCTCreator can be reused for analysis
    analysis = bct_creator.analyze_convergence("x1")

3. Argmin Series
~~~~~~~~~~~~~~~~~

Extract and manually plot belief trajectories:

.. code-block:: python

    series = visualizer.argmin_series(vars_filter=["x1", "x2"])
    # series = {"x1": [0, 0, 1, 1, 2, ...], "x2": [1, 1, 1, 2, 2, ...]}

    import matplotlib.pyplot as plt

    for var, trajectory in series.items():
        plt.plot(range(len(trajectory)), trajectory, label=var, marker="o")

    plt.xlabel("Iteration")
    plt.ylabel("Assignment")
    plt.legend()
    plt.show()

Exporting and Persisting
------------------------

Save Individual Snapshots
~~~~~~~~~~~~~~~~~~~~~~~~~

The engine now collects the retained `SnapshotRecord` objects under :attr:`engine.snapshots`
and exposes a small :mod:`engine.save_snapshot` helper for exporting them.

.. code-block:: python

    from pathlib import Path

    snap_dir = Path("snapshot_output")
    snap_dir.mkdir(exist_ok=True)

    # Save a single snapshot to disk
    latest = engine.latest_snapshot()
    if latest:
        json_path = engine.save_snapshot.save_json(
            snap_dir / f"snapshot_step_{latest.data.step:04d}.json",
            step=latest.data.step,
        )
        csv_path = engine.save_snapshot.save_csv(
            snap_dir / "snapshot_summary.csv",
            step=latest.data.step,
        )
        print(f"Wrote JSON to {json_path}")
        print(f"Appended CSV summary to {csv_path}")

    # Creates human-readable JSON alongside a per-step CSV summary

Auto-Persist During Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configure automatic saving during run
    snapshot_cfg = SnapshotsConfig(
        compute_jacobians=True,
        save_each_step=True,
        save_dir="results/snapshots",  # Created automatically
    )

    engine = BPEngine(
        factor_graph=graph,
        snapshots_config=snapshot_cfg,
    )
    engine.run(max_iter=100)

    # Now all snapshots are persisted

Snapshot Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    results/snapshots/
    ├── index.json                    # Manifest of all saved steps
    ├── step_0000/
    │   ├── meta.json                # Metadata, analysis results
    │   ├── messages_q.npz           # Q messages
    │   ├── messages_r.npz           # R messages
    │   ├── unary.npz                # Unary potentials
    │   ├── A.npz, P.npz, B.npz      # Jacobian matrices
    ├── step_0001/
    └── ...

BCT Export
~~~~~~~~~~

.. code-block:: python

    # Export complete BCT analysis as JSON
    bct_creator.export_analysis("bct_complete_analysis.json")

    # File structure:
    # {
    #   "metadata": {
    #     "damping_factor": 0.9,
    #     "total_variables": 5,
    #     "total_steps": 100
    #   },
    #   "variable_analyses": {
    #     "x1": {
    #       "variable": "x1",
    #       "total_iterations": 100,
    #       "initial_belief": 2.5,
    #       "final_belief": 0.1,
    #       "converged": true,
    #       "convergence_iteration": 45,
    #       ...
    #     },
    #     ...
    #   },
    #   "global_data": { ... }
    # }

Use Cases and Examples
----------------------

Use Case 1: Debug Non-Convergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Algorithm runs but beliefs don't stabilize.

**Solution**:

.. code-block:: python

    visualizer = SnapshotVisualizer(all_snapshots)
    variables = visualizer.variables()

    # Check belief trajectories
    series = visualizer.argmin_series(vars_filter=variables[:3])
    for var, traj in series.items():
        if len(set(traj[-10:])) > 1:  # Last 10 still oscillating?
            print(f"⚠ {var} is still oscillating!")
            visualizer.plot_argmin_per_variable(vars_filter=[var], show=True)

    # Check cycle metrics
    cycles = latest_cycles(engine)
    if cycles and cycles.num_cycles > 0 and not cycles.has_certified_contraction:
        print(f"⚠ Found {cycles.num_cycles} cycles, contraction not certified")
        print(f"  Aligned hops: {cycles.aligned_hops_total}")

Use Case 2: Compare Two Engine Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Which damping factor converges faster?

**Solution**:

.. code-block:: python

    from propflow.snapshots import SnapshotVisualizer
    import matplotlib.pyplot as plt

    # Run two experiments
    configs = [0.7, 0.9]
    results = {}

    for damp in configs:
        engine = DampingEngine(
            factor_graph=graph,
            damping_factor=damp,
            snapshots_config=SnapshotsConfig(),
        )
        engine.run(max_iter=100)

        snaps = [
            get_snapshot(engine, i)
            for i in range(len(engine.history.step_costs))
        ]

        results[damp] = snaps

    # Compare cost trajectories
    fig, ax = plt.subplots()
    for damp, snaps in results.items():
        costs = [s.data.global_cost for s in snaps if s.data.global_cost]
        ax.plot(range(len(costs)), costs, label=f"damp={damp}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Global Cost")
    ax.legend()
    ax.grid()
    plt.show()

Use Case 3: Validate Convergence Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Need proof that algorithm will converge.

**Solution**:

.. code-block:: python

    from propflow.snapshots.utils import latest_jacobians

    engine = BPEngine(
        factor_graph=graph,
        snapshots_config=SnapshotsConfig(
            compute_jacobians=True,
            compute_block_norms=True,
            compute_cycles=True,
        )
    )
    engine.run(max_iter=100)

    latest = latest_snapshot(engine)
    jac = latest.jacobians

    if jac and jac.block_norms:
        M_upper = jac.block_norms["||M||_inf_upper"]

        if M_upper < 1.0:
            print(f"✓ Convergence proven! ||M||_inf_upper = {M_upper:.4f} < 1.0")
        else:
            print(f"✗ Convergence not proven. ||M||_inf_upper = {M_upper:.4f} >= 1.0")

Use Case 4: Analyze Per-Variable Convergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Some variables converge faster than others; why?

**Solution**:

.. code-block:: python

    bct_creator = visualizer.plot_bct("x1", show=False)

    # Get analysis for all variables
    all_analyses = {}
    for var in visualizer.variables():
        all_analyses[var] = bct_creator.analyze_convergence(var)

    # Rank by convergence speed
    sorted_vars = sorted(
        all_analyses.items(),
        key=lambda item: item[1]["convergence_iteration"] or float("inf")
    )

    print("Convergence ranking (fastest to slowest):")
    for var, analysis in sorted_vars:
        conv_iter = analysis["convergence_iteration"]
        status = "✓" if analysis["converged"] else "✗"
        print(f"{status} {var}: iteration {conv_iter}")

Use Case 5: Study Message Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Understand which factors send large messages.

**Solution**:

.. code-block:: python

    # Analyze message magnitudes
    latest = latest_snapshot(engine)
    data = latest.data

    # Factor-to-variable message magnitudes
    r_magnitudes = {}
    for (factor, var), r_msg in data.R.items():
        magnitude = float(np.linalg.norm(r_msg))
        key = f"{factor}->{var}"
        r_magnitudes[key] = magnitude

    # Find largest messages
    sorted_msgs = sorted(
        r_magnitudes.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("Top 10 largest R-messages:")
    for msg, mag in sorted_msgs[:10]:
        print(f"  {msg}: {mag:.2f}")

Advanced Topics
---------------

Accessing Raw Matrices
~~~~~~~~~~~~~~~~~~~~~~

If you need the Jacobian matrices for custom analysis:

.. code-block:: python

    jac = latest_jacobians(engine)

    # jac.A: R -> Q dependencies (sparse CSR matrix)
    # jac.P: Projection for min-sum operator (sparse CSR matrix)
    # jac.B: Q -> R dependencies (sparse CSR matrix)

    # Convert to dense for small matrices
    if jac.A.shape[0] < 100:
        A_dense = jac.A.toarray()
        print(A_dense)

    # Or work directly with sparse format
    from scipy.sparse import linalg
    eigenvalues = linalg.eigsh(jac.A.T @ jac.A, k=1)[0]

Custom Analysis with SnapshotAnalyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from propflow.snapshots import SnapshotAnalyzer

    analyzer = SnapshotAnalyzer(all_snapshots)

    # Compute difference coordinates (for linearization analysis)
    delta_q, delta_r = analyzer.difference_coordinates(step_idx=50)

    # Construct Jacobian in difference coordinates
    jac_matrix = analyzer.jacobian(step_idx=50)

Summary
-------

Snapshots provide a comprehensive window into belief propagation dynamics:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Task
     - Tool
   * - Track variable beliefs over time
     - ``SnapshotVisualizer.argmin_series()``
   * - Visualize belief trajectories
     - ``SnapshotVisualizer.plot_argmin_per_variable()``
   * - Analyze convergence
     - ``BCTCreator.analyze_convergence()``
   * - Prove convergence (bounds)
     - Check ``Jacobians.block_norms["||M||_inf_upper"]``
   * - Find feedback loops
     - ``CycleMetrics`` from snapshot
   * - Compare configurations
     - Run multiple engines, collect snapshots, compare
   * - Export for later analysis
     - ``manager.save_step()`` or ``bct_creator.export_analysis()``

Start with **configuration**, move to **visualization** (is algorithm converging?), then **analysis** (why/why not?), and finally **export** results for reporting.
