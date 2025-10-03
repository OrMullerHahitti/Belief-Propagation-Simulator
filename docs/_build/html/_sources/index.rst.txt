PropFlow Documentation
=====================

**PropFlow** is a Python toolkit for building and experimenting with belief propagation
and other distributed constraint optimization (DCOP) algorithms on factor graphs.

.. image:: https://img.shields.io/pypi/v/propflow.svg
   :target: https://pypi.org/project/propflow/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/propflow.svg
   :target: https://pypi.org/project/propflow/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/OrMullerHahitti/Belief-Propagation-Simulator.svg
   :target: https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/blob/main/LICENSE
   :alt: License

Quick Start
-----------

Install PropFlow from PyPI:

.. code-block:: bash

   pip install propflow

Basic Example:

.. code-block:: python

   from propflow import BPEngine, FGBuilder
   from propflow.configs import create_random_int_table

   # Create a simple factor graph
   fg = FGBuilder.build_cycle_graph(
       num_vars=5,
       domain_size=2,
       ct_factory=create_random_int_table,
       ct_params={'low': 1, 'high': 10}
   )

   # Run belief propagation
   engine = BPEngine(fg)
   engine.run(max_iter=20)

   print(f"Solution: {engine.assignments}")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api/index
   examples
   contributing

Key Features
------------

* **Belief Propagation Variants**: Min-Sum, Max-Sum, Sum-Product algorithms
* **Search-Based DCOP Solvers**: DSA, MGM, K-Opt MGM algorithms
* **Extensible Policy Framework**: Damping, splitting, cost reduction, and more
* **Dynamic Graph Construction**: Cycle graphs, random graphs, custom topologies
* **Simulation and Analysis**: Parallel execution and result comparison
* **Visualization Tools**: Factor graph plotting and analysis visualization

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
