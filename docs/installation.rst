Installation
============

From PyPI (Recommended)
-----------------------

Install the latest stable version from PyPI:

.. code-block:: bash

   pip install propflow

From Source
-----------

For development or to get the latest changes:

.. code-block:: bash

   git clone https://github.com/OrMullerHahitti/Belief-Propagation-Simulator.git
   cd Belief-Propagation-Simulator
   pip install -e .

Development Installation
------------------------

To install with development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This includes testing tools (pytest), linting (black, flake8), and documentation tools.

To build the Sphinx documentation locally, also ensure the documentation theme
and Markdown/type-hint extensions are present:

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints

Requirements
------------

* Python 3.10 or higher
* NumPy >= 2.2.1
* NetworkX >= 3.4.2
* Matplotlib >= 3.10.0
* SciPy >= 1.15.2

See the full dependency list in the `pyproject.toml` file.

Verifying Installation
----------------------

After installation, verify everything works:

.. code-block:: python

   from propflow import BPEngine, FGBuilder
   print("PropFlow installed successfully!")

Or check the version:

.. code-block:: bash

   bp-sim --version
