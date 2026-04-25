Snapshot Guide
==============

Snapshots capture the per-step state of a PropFlow engine and are collected
automatically by ``BPEngine``.

Capture Snapshots
-----------------

.. code-block:: python

   from propflow import BPEngine, FGBuilder
   from propflow.configs import create_random_int_table

   graph = FGBuilder.build_cycle_graph(
       num_vars=6,
       domain_size=3,
       ct_factory=create_random_int_table,
       ct_params={"low": 0, "high": 5},
   )
   engine = BPEngine(graph)
   engine.run(max_iter=60)

   latest = engine.latest_snapshot()
   all_snapshots = list(engine.snapshots)

Analyze and Visualize
---------------------

.. code-block:: python

   from propflow.snapshots import SnapshotAnalyzer, SnapshotVisualizer

   analyzer = SnapshotAnalyzer(all_snapshots)
   norms = analyzer.block_norms(step_idx=len(all_snapshots) - 1)

   viz = SnapshotVisualizer(all_snapshots)
   fig, payload = viz.plot_global_cost(show=False, return_data=True)

Persist a Compact Trace
-----------------------

.. code-block:: python

   import json
   from pathlib import Path

   out_dir = Path("snapshots")
   out_dir.mkdir(exist_ok=True)

   for snap in engine.snapshots:
       payload = {
           "step": snap.step,
           "global_cost": snap.global_cost,
           "assignments": snap.assignments,
           "metadata": snap.metadata,
       }
       (out_dir / f"snapshot_{snap.step:04d}.json").write_text(
           json.dumps(payload, indent=2),
           encoding="utf-8",
       )

BCT Data
--------

Snapshots contain enough Q/R message and cost-table detail for
``SnapshotVisualizer.plot_bct()`` to reconstruct Backtrack Cost Trees.
