Snapshot Guide (Updated)
========================

Snapshots capture the per-step state of a PropFlow engine and are now collected
without additional configuration.

Overview
--------
- `EngineSnapshot` stores messages, beliefs, assignments, metadata, and optional
  analysis placeholders.
- Snapshots are appended to ``engine.snapshots`` automatically; call
  :meth:`engine.latest_snapshot` for the most recent entry.
- Set ``use_bct_history=True`` when constructing an engine if you need
  Backtrack Cost Tree (BCT) data.

Capturing Snapshots
-------------------

.. code-block:: python

   from propflow import BPEngine, FGBuilder

   graph = FGBuilder.build_cycle_graph(num_vars=6, domain_size=3, ct_factory="random_int", ct_params={"low": 0, "high": 5})
   engine = BPEngine(graph, use_bct_history=True)
   engine.run(max_iter=60)

   latest = engine.latest_snapshot()
   all_snapshots = engine.snapshots

Persisting Snapshots
--------------------

Use the built-in recorder helper or serialise manually:

.. code-block:: python

   import json
from pathlib import Path

   import json
   from pathlib import Path

   out_dir = Path("snapshots")
   out_dir.mkdir(exist_ok=True)
   for snap in engine.snapshots:
       payload = {
           "step": snap.step,
           "global_cost": snap.global_cost,
           "assignments": snap.assignments,
       }
       (out_dir / f"snapshot_{snap.step:04d}.json").write_text(json.dumps(payload, indent=2))

Analysis and Visualisation
--------------------------

.. code-block:: python

   from propflow.snapshots import SnapshotAnalyzer
   from propflow.snapshots import SnapshotVisualizer

   analyzer = SnapshotAnalyzer(engine.snapshots)
   norms = analyzer.block_norms(engine.latest_snapshot().step)

   viz = SnapshotVisualizer(engine.snapshots)
   fig, payload = viz.plot_global_cost(show=False, return_data=True)

BCT Data
--------

When ``use_bct_history=True`` is specified, snapshots contain enough detail to
reconstruct Backtrack Cost Trees via ``propflow.utils.tools.bct``.

Future Work
-----------

Upcoming releases will reintroduce optional Jacobian and cycle analysis as
post-processing modules built on top of `EngineSnapshot`.
