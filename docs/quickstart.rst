Quick Start Guide
=================

This guide follows PropFlow’s recommended top-down flow: build a graph with
``FGBuilder``, run an engine, compare engine variants, and optionally dig into
analysis tooling. Use the :doc:`user_guide` when you need deeper detail on each
layer.

Prepare the Environment
-----------------------

Install PropFlow and its development extras (for docs/tests) into an activated
virtual environment:

.. code-block:: bash

   pip install -e '.[dev]'

Step 1 — Build a Factor Graph (FGBuilder First)
-----------------------------------------------

``FGBuilder`` gives you a fully initialised :class:`propflow.bp.factor_graph.FactorGraph`
without hand-wiring agents. Pick a topology, choose a cost-table factory, and
specify parameters.

.. code-block:: python

   from propflow import FGBuilder
   from propflow.configs import create_random_int_table

   graph = FGBuilder.build_cycle_graph(
       num_vars=6,
       domain_size=3,
       ct_factory=create_random_int_table,
       ct_params={"low": 0, "high": 25},
   )

``FGBuilder`` ensures factors, variables, and edges are consistent, and it
initialises cost tables via the provided factory.

Step 2 — Run Belief Propagation
-------------------------------

Instantiate an engine (``BPEngine`` by default) with the graph. Engines attach
computators, seed mailboxes, and iterate until convergence or a max iteration
count.

.. code-block:: python

   from propflow import BPEngine

   engine = BPEngine(graph)
   engine.run(max_iter=50)

   print("Assignments:", engine.assignments)
   print("Global cost:", engine.graph.global_cost)

Step 3 — Try Engine Variants and Policies
-----------------------------------------

Swap computators or engines to experiment with different BP behaviours. Here we
compare plain BP to a damped variant.

.. code-block:: python

   from propflow import DampingEngine, MinSumComputator

   baseline = BPEngine(graph, computator=MinSumComputator())
   baseline.run(max_iter=50)

   damped = DampingEngine(graph, damping_factor=0.85)
   damped.run(max_iter=50)

   print("Baseline cost:", baseline.history.costs[-1])
   print("Damped cost:", damped.history.costs[-1])

Step 4 — Scale Out with the Simulator
-------------------------------------

Run several engine configurations across a batch of graphs using
:class:`propflow.simulator.Simulator`. It will execute runs in parallel when
possible and aggregate cost histories.

.. code-block:: python

   from propflow import Simulator, FGBuilder, BPEngine, DampingEngine
   from propflow.configs import CTFactory

   configs = {
       "baseline": {"class": BPEngine},
       "damped": {"class": DampingEngine, "damping_factor": 0.85},
   }

   graphs = [
       FGBuilder.build_random_graph(
           num_vars=12,
           domain_size=3,
           ct_factory=CTFactory.random_int.fn,
           ct_params={"low": 5, "high": 30},
           density=0.3,
       )
       for _ in range(5)
   ]

   simulator = Simulator(configs)
   results = simulator.run_simulations(graphs, max_iter=200)
   simulator.plot_results()

Optional — Build a Custom Graph Manually
----------------------------------------

When you need a structure that the helpers do not cover, create agents directly
and pass an explicit ``edges`` mapping into :class:`propflow.bp.factor_graph.FactorGraph`.
Remember that the list of variables for each factor is ordered—the position in
the list matches the axis in the cost table.

.. code-block:: python

   from propflow import FactorGraph, VariableAgent, FactorAgent, BPEngine
   from propflow.configs import create_uniform_float_table

   x1 = VariableAgent("x1", domain=2)
   x2 = VariableAgent("x2", domain=2)

   parity = FactorAgent(
       name="f12",
       domain=2,
       ct_creation_func=create_uniform_float_table,
   )

   graph = FactorGraph(
       variable_li=[x1, x2],
       factor_li=[parity],
       edges={parity: [x1, x2]},
   )

   engine = BPEngine(graph)
   engine.run(max_iter=25)

Checklist for manual builds:

* Every factor appears exactly once in ``factor_li`` and as a key in ``edges``.
* Each value in ``edges`` is an ordered list of variables; the order defines
  tensor dimensions.
* Cost-table factories must accept ``num_vars`` and ``domain_size`` (PropFlow
  passes both arguments automatically).
* Use deterministic parameters (seeds, bounds) when you want reproducibility.

Inspecting Runs with Analyzer Tooling
-------------------------------------

Capture rich per-iteration data by pairing built-in snapshots with the external
recorder.

.. code-block:: python

   from propflow import BPEngine, FGBuilder, SnapshotsConfig
   from propflow.configs import create_random_int_table
   from analyzer.snapshot_recorder import EngineSnapshotRecorder

   fg = FGBuilder.build_cycle_graph(
       num_vars=8,
       domain_size=3,
       ct_factory=create_random_int_table,
       ct_params={"low": 1, "high": 20},
   )

   snapshots = SnapshotsConfig(compute_cycles=True, retain_last=40)
   instrumented = BPEngine(fg, snapshots_config=snapshots)

   recorder = EngineSnapshotRecorder(instrumented)
   recorder.record_run(max_steps=80, break_on_convergence=True)
   recorder.to_json("results/demo/run.json")

   # Inspect latest built-in snapshot (if needed)
   latest = instrumented.latest_snapshot()

Where to Go Next
----------------

* Read the :doc:`user_guide` for a deeper explanation of each layer.
* Explore :doc:`examples` for more advanced scenarios and patterns.
* Consult the :doc:`handbook/index` when you need operational practices or
  deployment guidance.
* Browse :doc:`api/index` for the full API reference.
