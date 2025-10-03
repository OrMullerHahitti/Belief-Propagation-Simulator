Examples
========

This page contains complete examples for common use cases.

Basic Graph Coloring
--------------------

Solve a simple graph coloring problem using belief propagation:

.. code-block:: python

   from propflow import FGBuilder, BPEngine
   from propflow.configs import create_random_int_table

   # Create a cycle graph (each node must differ from neighbors)
   fg = FGBuilder.build_cycle_graph(
       num_vars=6,
       domain_size=3,  # 3 colors
       ct_factory=create_random_int_table,
       ct_params={'low': 0, 'high': 50}
   )

   # Run belief propagation
   engine = BPEngine(fg)
   engine.run(max_iter=30)

   print("Color assignments:", engine.assignments)
   print("Total cost:", engine.graph.global_cost)

Comparing Engine Variants
--------------------------

Compare different BP variants on the same problem:

.. code-block:: python

   from propflow import (
       Simulator, FGBuilder, BPEngine,
       DampingEngine, SplitEngine
   )
   from propflow.configs import CTFactory

   # Create test graphs
   graphs = [
       FGBuilder.build_cycle_graph(
           num_vars=15,
           domain_size=4,
           ct_factory=CTFactory.random_int.fn,
           ct_params={'low': 1, 'high': 100}
       ) for _ in range(10)
   ]

   # Define engine configurations
   configs = {
       "Standard BP": {
           "class": BPEngine
       },
       "Damped BP (0.5)": {
           "class": DampingEngine,
           "damping_factor": 0.5
       },
       "Damped BP (0.9)": {
           "class": DampingEngine,
           "damping_factor": 0.9
       },
       "Split Factor": {
           "class": SplitEngine,
           "split_factor": 0.6
       },
   }

   # Run comparison
   sim = Simulator(configs)
   results = sim.run_simulations(graphs, max_iter=200)

   # Plot results
   sim.plot_results(verbose=True)

Custom Cost Function
--------------------

Create a problem with custom cost functions:

.. code-block:: python

   import numpy as np
   from propflow import FactorGraph, VariableAgent, FactorAgent, BPEngine

   def attractive_cost(num_vars=2, domain_size=3, **kwargs):
       """Cost function that prefers same values."""
       table = np.zeros((domain_size, domain_size))
       for i in range(domain_size):
           for j in range(domain_size):
               # Lower cost when values match
               table[i, j] = 0 if i == j else 10
       return table

   # Build graph with custom costs
   vars = [VariableAgent(f"x{i}", domain=3) for i in range(4)]
   factors = [
       FactorAgent(f"f{i}", domain=3,
                   ct_creation_func=attractive_cost)
       for i in range(3)
   ]

   edges = {
       factors[0]: [vars[0], vars[1]],
       factors[1]: [vars[1], vars[2]],
       factors[2]: [vars[2], vars[3]],
   }

   fg = FactorGraph(vars, factors, edges)

   # Solve
   engine = BPEngine(fg)
   engine.run(max_iter=20)
   print(engine.assignments)

Large-Scale Problem
-------------------

Handle larger problems efficiently:

.. code-block:: python

   from propflow import FGBuilder, DampingEngine
   from propflow.policies import ConvergenceConfig
   from propflow.configs import CTFactory

   # Create large random graph
   fg = FGBuilder.build_random_graph(
       num_vars=50,
       domain_size=10,
       ct_factory=CTFactory.random_int.fn,
       ct_params={'low': 100, 'high': 200},
       density=0.25  # 25% of possible edges
   )

   # Configure convergence
   conv_config = ConvergenceConfig(
       min_iterations=20,
       belief_threshold=1e-5,
       patience=30
   )

   # Use damping for stability
   engine = DampingEngine(
       factor_graph=fg,
       damping_factor=0.8,
       convergence_config=conv_config
   )

   # Run with monitoring
   engine.run(max_iter=1000)

   print(f"Converged in {engine.iteration_count} iterations")
   print(f"Final cost: {engine.graph.global_cost}")

Analyzing Convergence
---------------------

Track and analyze convergence behavior:

.. code-block:: python

   from propflow import BPEngine, FGBuilder
   from propflow.snapshots import SnapshotsConfig
   import matplotlib.pyplot as plt

   fg = FGBuilder.build_cycle_graph(
       num_vars=10,
       domain_size=3,
       ct_factory=create_random_int_table,
       ct_params={'low': 1, 'high': 50}
   )

   # Enable detailed snapshots
   snap_config = SnapshotsConfig(
       compute_jacobians=True,
       retain_last=100
   )

   engine = BPEngine(fg, snapshots_config=snap_config)
   engine.run(max_iter=100)

   # Plot cost over time
   costs = engine.history.costs
   plt.plot(costs)
   plt.xlabel('Iteration')
   plt.ylabel('Global Cost')
   plt.title('Convergence Plot')
   plt.show()

   # Analyze snapshots
   snapshot = engine.latest_snapshot()
   print("Latest message statistics:", snapshot.metadata)
