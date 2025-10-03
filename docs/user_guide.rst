User Guide
==========

This comprehensive guide covers all aspects of using PropFlow.

.. contents:: Table of Contents
   :local:
   :depth: 2

Core Concepts
-------------

Factor Graphs
~~~~~~~~~~~~~

A factor graph is a bipartite graph that represents the factorization of a function.
It consists of:

* **Variable nodes**: Represent variables in your problem
* **Factor nodes**: Represent constraints or cost functions between variables
* **Edges**: Connect factors to the variables they constrain

Variables
~~~~~~~~~

Variables are created using :class:`~propflow.core.agents.VariableAgent`:

.. code-block:: python

   from propflow.core import VariableAgent

   var = VariableAgent(name="x1", domain=3)  # Variable with 3 possible values

Factors
~~~~~~~

Factors encode relationships between variables via cost tables:

.. code-block:: python

   from propflow.core import FactorAgent
   from propflow.configs import create_random_int_table

   factor = FactorAgent(
       name="f12",
       domain=3,  # Domain size of connected variables
       ct_creation_func=create_random_int_table,
       param={"low": 0, "high": 100}
   )

Belief Propagation Algorithms
------------------------------

PropFlow supports several BP variants through different computators:

Min-Sum
~~~~~~~

Minimizes the sum of costs (default):

.. code-block:: python

   from propflow import BPEngine, MinSumComputator

   engine = BPEngine(fg, computator=MinSumComputator())

Max-Sum
~~~~~~~

Maximizes the sum of values:

.. code-block:: python

   from propflow import MaxSumComputator

   engine = BPEngine(fg, computator=MaxSumComputator())

Sum-Product
~~~~~~~~~~~

Computes marginal probabilities:

.. code-block:: python

   from propflow import SumProductComputator

   engine = BPEngine(fg, computator=SumProductComputator())

Policies and Engine Variants
-----------------------------

Damping
~~~~~~~

Smooths messages to prevent oscillations:

.. code-block:: python

   from propflow import DampingEngine

   engine = DampingEngine(
       factor_graph=fg,
       damping_factor=0.9  # 0 = no damping, 1 = full damping
   )

Factor Splitting
~~~~~~~~~~~~~~~~

Splits factors to alter convergence:

.. code-block:: python

   from propflow import SplitEngine

   engine = SplitEngine(
       factor_graph=fg,
       split_factor=0.5
   )

Cost Reduction
~~~~~~~~~~~~~~

Applies one-time cost reduction:

.. code-block:: python

   from propflow import CostReductionOnceEngine

   engine = CostReductionOnceEngine(
       factor_graph=fg,
       reduction_factor=0.8
   )

Convergence Monitoring
----------------------

Configure convergence detection:

.. code-block:: python

   from propflow.policies import ConvergenceConfig

   config = ConvergenceConfig(
       min_iterations=10,
       belief_threshold=1e-6,
       patience=15
   )

   engine = BPEngine(fg, convergence_config=config)

Graph Construction
------------------

Cycle Graphs
~~~~~~~~~~~~

.. code-block:: python

   fg = FGBuilder.build_cycle_graph(
       num_vars=10,
       domain_size=3,
       ct_factory=create_random_int_table,
       ct_params={'low': 0, 'high': 100}
   )

Random Graphs
~~~~~~~~~~~~~

.. code-block:: python

   fg = FGBuilder.build_random_graph(
       num_vars=20,
       domain_size=4,
       ct_factory=create_random_int_table,
       ct_params={'low': 0, 'high': 100},
       density=0.3
   )

Custom Graphs
~~~~~~~~~~~~~

Build graphs manually for full control:

.. code-block:: python

   from propflow import FactorGraph, VariableAgent, FactorAgent

   # Create variables
   vars = [VariableAgent(f"x{i}", domain=2) for i in range(5)]

   # Create factors
   factors = [
       FactorAgent(f"f{i}", domain=2,
                   ct_creation_func=create_random_int_table,
                   param={"low": 1, "high": 10})
       for i in range(4)
   ]

   # Define connections
   edges = {
       factors[0]: [vars[0], vars[1]],
       factors[1]: [vars[1], vars[2]],
       factors[2]: [vars[2], vars[3]],
       factors[3]: [vars[3], vars[4]],
   }

   fg = FactorGraph(vars, factors, edges)

Advanced Topics
---------------

Custom Cost Tables
~~~~~~~~~~~~~~~~~~

Create custom cost table factories:

.. code-block:: python

   import numpy as np

   def my_cost_table(num_vars, domain_size, **kwargs):
       shape = tuple([domain_size] * num_vars)
       # Your custom logic here
       return np.random.exponential(scale=10, size=shape)

Snapshots and Analysis
~~~~~~~~~~~~~~~~~~~~~~

Enable snapshot recording for detailed analysis:

.. code-block:: python

   from propflow.snapshots import SnapshotsConfig

   config = SnapshotsConfig(
       compute_jacobians=True,
       compute_cycles=True,
       retain_last=20
   )

   engine = BPEngine(fg, snapshots_config=config)
   engine.run(max_iter=50)

   snapshot = engine.latest_snapshot()
   # Analyze convergence patterns

Search Algorithms
~~~~~~~~~~~~~~~~~

Use local search instead of BP:

.. code-block:: python

   from propflow.search import DSAEngine, DSAComputator

   dsa = DSAEngine(
       factor_graph=fg,
       computator=DSAComputator(probability=0.7)
   )
   results = dsa.run(max_iter=100)
