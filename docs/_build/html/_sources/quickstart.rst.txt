Quick Start Guide
=================

This guide will get you started with PropFlow in minutes.

Your First Factor Graph
-----------------------

Let's create a simple factor graph and run belief propagation:

.. code-block:: python

   from propflow import FactorGraph, VariableAgent, FactorAgent, BPEngine
   from propflow.configs import create_random_int_table

   # 1. Create variables
   x1 = VariableAgent("x1", domain=2)
   x2 = VariableAgent("x2", domain=2)

   # 2. Create a factor connecting them
   factor = FactorAgent(
       "f12",
       domain=2,
       ct_creation_func=create_random_int_table,
       param={"low": 1, "high": 10}
   )

   # 3. Build the factor graph
   fg = FactorGraph(
       variable_li=[x1, x2],
       factor_li=[factor],
       edges={factor: [x1, x2]}
   )

   # 4. Run belief propagation
   engine = BPEngine(fg)
   engine.run(max_iter=10)

   # 5. Get results
   print(f"Assignments: {engine.assignments}")
   print(f"Global cost: {engine.graph.global_cost}")

Using the Graph Builder
-----------------------

For common graph structures, use the ``FGBuilder``:

.. code-block:: python

   from propflow import FGBuilder, BPEngine
   from propflow.configs import create_random_int_table

   # Create a cycle graph
   fg = FGBuilder.build_cycle_graph(
       num_vars=5,
       domain_size=3,
       ct_factory=create_random_int_table,
       ct_params={'low': 0, 'high': 100}
   )

   # Run BP
   engine = BPEngine(fg)
   engine.run(max_iter=20)
   print(engine.assignments)

Applying Policies
-----------------

Use different engine variants for different behaviors:

.. code-block:: python

   from propflow import DampingEngine, SplitEngine

   # Damping helps with oscillations
   damped = DampingEngine(
       factor_graph=fg,
       damping_factor=0.9
   )
   damped.run(max_iter=20)

   # Splitting alters message flow
   split = SplitEngine(
       factor_graph=fg,
       split_factor=0.5
   )
   split.run(max_iter=20)

Running Experiments
-------------------

Compare multiple configurations:

.. code-block:: python

   from propflow import Simulator, BPEngine, DampingEngine
   from propflow.configs import CTFactory

   # Define configurations
   configs = {
       "Standard": {"class": BPEngine},
       "Damped": {"class": DampingEngine, "damping_factor": 0.9},
   }

   # Create test graphs
   graphs = [
       FGBuilder.build_cycle_graph(
           num_vars=10,
           domain_size=3,
           ct_factory=CTFactory.random_int.fn,
           ct_params={'low': 0, 'high': 100}
       ) for _ in range(5)
   ]

   # Run simulations
   sim = Simulator(configs)
   results = sim.run_simulations(graphs, max_iter=100)
   sim.plot_results()

Next Steps
----------

* Read the :doc:`user_guide` for detailed explanations
* Check out :doc:`examples` for more use cases
* Explore the :doc:`api/index` for complete API reference
