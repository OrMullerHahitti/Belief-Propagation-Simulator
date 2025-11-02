PropFlow User Guide
===================

This guide presents PropFlow from the top down so you can understand the
high-level architecture before working through concrete APIs. Follow the chain
from individual agents, through factor graphs and engines, all the way to
full-blown simulator runs and analysis tooling.

.. contents::
   :local:
   :depth: 2


Top-Down Architecture
---------------------

PropFlow can be viewed as a layered pipeline. Each layer builds on the previous
one, and you can exit early if you only need part of the stack:

1. **Agents** (:class:`propflow.core.agents.VariableAgent`,
   :class:`propflow.core.agents.FactorAgent`) exchange messages.
2. **Factor graphs** (:class:`propflow.bp.factor_graph.FactorGraph`) connect
   agents and initialize cost tables. Helper builders live in
   :mod:`propflow.utils`.
3. **Engines** (:class:`propflow.bp.engine_base.BPEngine` and subclasses) run
   belief propagation, manage convergence policies, and capture history.
4. **Simulations** (:class:`propflow.simulator.Simulator`) execute batches of
   engine configurations across many graphs for fair comparisons.
5. **Analysis tooling** (:mod:`propflow.snapshots`) records data and reports
   metrics for offline inspection.


Agents Layer
------------

Agents are the smallest active components in PropFlow. They inherit from
:class:`propflow.core.agents.FGAgent`, which embeds an inbox/outbox, message
history, and a link to a ``computator`` implementing the BP math.

Variable Agents
~~~~~~~~~~~~~~~

:class:`propflow.core.agents.VariableAgent` models a discrete decision variable.
Important attributes and behaviours:

* ``name`` – identifier used in logs and assignments.
* ``domain`` – number of discrete values this variable may take.
* ``belief`` – vector computed by the attached ``computator`` (defaults to
  uniform if messages are missing).
* ``curr_assignment`` – best value implied by the current belief.
* ``compute_messages()`` – calls ``computator.compute_Q`` to prepare messages
  for neighbouring factors.

Example:

.. code-block:: python

   from propflow.core import VariableAgent

   temperature = VariableAgent(name="temp_room_a", domain=4)


Factor Agents
~~~~~~~~~~~~~

:class:`propflow.core.agents.FactorAgent` encodes the local relationships
between several variables. Each factor owns a cost table that is lazily created
from a factory function.

Key fields:

* ``cost_table`` – ``numpy.ndarray`` scoring each variable assignment tuple.
* ``ct_creation_func`` / ``ct_creation_params`` – factory for building the
  table. The factor graph calls :meth:`~propflow.core.agents.FactorAgent.initiate_cost_table`
  once the neighbourhood is known.
* ``connection_number`` – mapping of variable names to axis indices. Maintained
  automatically when you add edges.
* ``compute_messages()`` – uses ``computator.compute_R`` to send responses back
  to variables.

.. code-block:: python

   from propflow.core import FactorAgent
   from propflow.configs import create_random_int_table

   penalty = FactorAgent(
       name="f_xy",
       domain=3,
       ct_creation_func=create_random_int_table,
       param={"low": 0, "high": 10},
   )


Message Lifecycle
~~~~~~~~~~~~~~~~~

Agents exchange :class:`propflow.core.components.Message` objects stored within
a :class:`propflow.core.components.MailHandler`. The handler:

* Deduplicates messages per sender.
* Seeds zero-messages so every neighbour pair can exchange information on the
  very first engine iteration.
* Stages outgoing messages until the engine triggers delivery.

You seldom interact with messages directly unless you’re implementing new BP
variants.


Factor Graph Layer
------------------

With agents in hand, :class:`propflow.bp.factor_graph.FactorGraph` wires them
into a bipartite structure, initializes cost tables, and exposes convenience
properties such as the graph diameter and current assignments. Most users
should rely on :class:`propflow.utils.FGBuilder` to create graphs. The helpers
ensure domain sizes line up, edges are valid, and factors receive their cost
tables automatically.

Using FGBuilder
~~~~~~~~~~~~~~~

``FGBuilder`` covers common topologies so you can focus on experiments instead
of plumbing. The snippet below builds a cycle and runs a plain BP engine:

.. code-block:: python

   from propflow import FGBuilder, BPEngine
   from propflow.configs import create_random_int_table

   fg = FGBuilder.build_cycle_graph(
       num_vars=5,
       domain_size=3,
       ct_factory=create_random_int_table,
       ct_params={"low": 0, "high": 10},
   )

   engine = BPEngine(fg)
   engine.run(max_iter=25)
   print(engine.assignments)

Other helpers such as :meth:`propflow.utils.fg_utils.FGBuilder.build_random_graph`
return fully initialised :class:`FactorGraph` objects as well.


Config-Driven Graphs
~~~~~~~~~~~~~~~~~~~~

For reproducible benchmarks, create a :class:`propflow.utils.create.GraphConfig`
and hand it to :class:`propflow.utils.create.FactorGraphBuilder`:

.. code-block:: python

   from pathlib import Path
   from propflow.utils.create import FactorGraphBuilder

   cfg_path = Path("configs/factor_graphs/cycle_demo.pkl")
   builder = FactorGraphBuilder()
   fg = builder.build_and_return(cfg_path)

The builder loads the config, resolves registered graph/cost factories, and
produces a :class:`FactorGraph`. Use :meth:`FactorGraphBuilder.build_and_save`
to persist generated graphs for later reuse.


Manual Graph Assembly
~~~~~~~~~~~~~~~~~~~~~

When you need a structure that the helpers do not cover—custom agents, hybrid
domains—build the graph yourself. Provide explicit lists of variables, factors,
and an ordered ``edges`` mapping.

.. code-block:: python

   from propflow import FactorGraph, VariableAgent, FactorAgent
   from propflow.configs import create_uniform_float_table

   x1 = VariableAgent("x1", domain=2)
   x2 = VariableAgent("x2", domain=2)
   parity = FactorAgent(
       name="f12",
       domain=2,
       ct_creation_func=create_uniform_float_table,
   )

   fg = FactorGraph(
       variable_li=[x1, x2],
       factor_li=[parity],
       edges={parity: [x1, x2]},
   )

Checklist for manual graphs:

* Every factor supplied in ``factor_li`` appears as a key in ``edges``.
* Each value in ``edges`` is an ordered list; the index order defines tensor
  axes, so be deliberate when mapping variables to dimensions.
* ``ct_creation_func`` must accept ``num_vars`` and ``domain_size`` arguments;
  PropFlow passes them automatically.
* Use deterministic parameters (bounds, seeds) when you want reproducible runs.


Engine Layer
------------

Engines coordinate message passing, convergence behaviour, history tracking,
and optional snapshots. The base :class:`propflow.bp.engine_base.BPEngine`
implements synchronous belief propagation: variables update first, then factors,
for each iteration.

Core responsibilities:

* Assign the chosen :class:`propflow.core.dcop_base.Computator` to every agent.
* Seed inboxes with zero-messages so computation can start immediately.
* Execute ``step`` loops until convergence or a maximum iteration cap.
* Record costs, beliefs, and assignments in :class:`propflow.bp.engine_components.History`.
* Expose hook methods (``pre_factor_compute`` etc.) that subclasses override to
  implement policies.


Selecting a Computator
~~~~~~~~~~~~~~~~~~~~~~

Computators contain the algorithmic math. PropFlow ships with:

* :class:`propflow.bp.computators.MinSumComputator` (default)
* :class:`propflow.bp.computators.MaxSumComputator`
* :class:`propflow.bp.computators.SumProductComputator`
* :class:`propflow.bp.computators.MaxProductComputator`

Swap variants by passing the desired instance to the engine:

.. code-block:: python

   from propflow import BPEngine, MaxSumComputator

   engine = BPEngine(fg, computator=MaxSumComputator())


Engine Variants and Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specialised engines extend ``BPEngine`` with additional behaviour:

* :class:`propflow.bp.engines.DampingEngine` – smooths messages.
* :class:`propflow.bp.engines.SplitEngine` – splits factors to alter dynamics.
* :class:`propflow.bp.engines.CostReductionOnceEngine` – reduces costs once at
  startup.
* :class:`propflow.bp.engines.MessagePruningEngine` – prunes messages using
  policies.

Complement engines with policies and utilities:

* :class:`propflow.policies.convergance.ConvergenceConfig` to define minimum
  iterations, tolerance, and patience.
* :func:`propflow.policies.normalize_cost.normalize_inbox` to shift messages and
  avoid numerical blow-ups.
* Built-in snapshot capture (``engine.snapshots``) to inspect per-step state;
  set ``use_bct_history=True`` on the engine if you need message traces for BCT
  tooling.


Running a Single Engine
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from propflow import BPEngine, MinSumComputator

   engine = BPEngine(
       factor_graph=fg,
       computator=MinSumComputator(),
       use_bct_history=True,  # optional: retain message traces for BCT tools
   )

   engine.run(max_iter=100)
   final_cost = engine.snapshots[-1].global_cost
   beliefs = engine.get_beliefs()

Inspect :attr:`engine.assignments` or iterate over :attr:`engine.snapshots`
for detailed per-step data. :meth:`engine.latest_snapshot` returns the most
recent snapshot object.


Simulation Layer
----------------

The :class:`propflow.simulator.Simulator` orchestrates multiple engine
configurations running over many graphs—perfect for benchmarking or tuning.

1. Prepare a configuration dictionary mapping experiment names to engine
   classes plus keyword arguments.
2. Build a list of factor graphs (reuse ``FGBuilder`` helpers or load pickled
   graphs).
3. Call :meth:`Simulator.run_simulations` to execute everything. The simulator
   attempts to run in parallel using ``multiprocessing`` but falls back to
   sequential processing if required.
4. Use :meth:`Simulator.plot_results` to visualise mean cost trajectories.

.. code-block:: python

   from propflow import Simulator, BPEngine, DampingEngine, FGBuilder
   from propflow.configs import create_random_int_table

   configs = {
       "baseline": {"class": BPEngine},
       "damped": {"class": DampingEngine, "damping_factor": 0.85},
   }

   graphs = [
       FGBuilder.build_random_graph(
           num_vars=12,
           domain_size=3,
           ct_factory=create_random_int_table,
           ct_params={"low": 0, "high": 15},
           density=0.25,
       )
       for _ in range(4)
   ]

   simulator = Simulator(configs)
   aggregated = simulator.run_simulations(graphs, max_iter=150)
   simulator.plot_results(verbose=True)


Analysis Layer
--------------

Advanced studies often require visibility into per-iteration behaviour. The
:mod:`propflow.snapshots` package provides the core tooling:

* :class:`propflow.snapshots.SnapshotAnalyzer` and :class:`propflow.snapshots.AnalysisReport`
  derive metrics, block norms, and summaries directly from ``engine.snapshots``.
* :class:`propflow.snapshots.SnapshotVisualizer` renders belief argmin trajectories and message norms.

Example workflow:

.. code-block:: python

   import json
   from pathlib import Path

   from propflow.snapshots import SnapshotAnalyzer, AnalysisReport
   from propflow.snapshots import SnapshotVisualizer

   engine = BPEngine(fg, use_bct_history=True)
   engine.run(max_iter=75)

   snapshots = list(engine.snapshots)

   out_path = Path("results/run_001.json")
   out_path.parent.mkdir(parents=True, exist_ok=True)
   out_path.write_text(json.dumps([
       {
           "step": snap.step,
           "assignments": snap.assignments,
           "global_cost": snap.global_cost,
       }
       for snap in snapshots
   ], indent=2))

   viz = SnapshotVisualizer(snapshots)
   viz.plot_argmin_per_variable(show=True)

   analyzer = SnapshotAnalyzer(snapshots)
   report = AnalysisReport(analyzer)
   summary = report.to_json(step_idx=len(snapshots) - 1)


Chain of Creation
-----------------

Use this checklist when building your own experiments:

1. **Choose a graph strategy**

   - Prefer :class:`propflow.utils.FGBuilder` for standard cycles or random
     graphs.
   - Fall back to manual agent construction when you need custom structures.
2. **Instantiate the factor graph**

   - Pass lists of variable and factor agents plus an ordered ``edges`` map.
   - Confirm domain sizes match the factor expectations.
3. **Pick an engine configuration**

   - Select a ``computator`` and, if needed, an engine variant with policies.
   - Enable snapshots or convergence rules to match your evaluation criteria.
4. **Run experiments**

   - Call :meth:`BPEngine.run` for single cases.
   - Use :class:`Simulator` to fan out across many graphs/configurations.
5. **Analyse results**

   - Inspect :attr:`engine.snapshots` for per-step assignments, messages, and costs.
   - Use :mod:`propflow.snapshots` for visualisation and metric reporting.


Custom Graph Checklist
~~~~~~~~~~~~~~~~~~~~~~

If you bypass ``FGBuilder``:

* Ensure every :class:`propflow.core.agents.FactorAgent` references each
  neighbouring :class:`propflow.core.agents.VariableAgent` exactly once.
* Provide cost-table factories that honour the ``(num_vars, domain_size)``
  signature—the FactorGraph constructor will call them for you.
* Call :class:`FactorGraph` only after all agents exist; it registers edges and
  triggers cost table creation automatically.
* Stick to deterministic seeds and bounds inside your cost factories for
  reproducible results.


Next Steps
----------

* Jump to :doc:`quickstart` for runnable snippets.
* Browse :doc:`examples` for complete demonstrations and notebooks.
* Consult :doc:`api/index` for the full API surface.
* Review :doc:`handbook/index` for deeper dives, patterns, and practices.
