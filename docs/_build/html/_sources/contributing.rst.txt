Contributing
============

We welcome contributions to PropFlow! This guide will help you get started.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/YOUR-USERNAME/Belief-Propagation-Simulator.git
   cd Belief-Propagation-Simulator

3. Create a virtual environment and install development dependencies:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"

4. Create a new branch for your feature:

.. code-block:: bash

   git checkout -b feature/your-feature-name

Development Workflow
--------------------

Running Tests
~~~~~~~~~~~~~

Run the full test suite:

.. code-block:: bash

   pytest

Run with coverage:

.. code-block:: bash

   pytest --cov=src --cov-report=html

Code Quality
~~~~~~~~~~~~

Format code with Black:

.. code-block:: bash

   black src/ tests/

Lint with Flake8:

.. code-block:: bash

   flake8 src/ tests/

Type check with MyPy:

.. code-block:: bash

   mypy src/

Or run all checks with pre-commit:

.. code-block:: bash

   pre-commit run --all-files

Documentation
~~~~~~~~~~~~~

Build documentation locally:

.. code-block:: bash

   cd docs
   make html
   open _build/html/index.html  # On Mac
   # Or: start _build/html/index.html  # On Windows

Contribution Guidelines
-----------------------

Code Style
~~~~~~~~~~

* Follow PEP 8
* Use Black for formatting (line length 120)
* Add type hints to all functions
* Write descriptive docstrings in Google/NumPy format

Commit Messages
~~~~~~~~~~~~~~~

Use conventional commit format:

* ``feat: Add new feature``
* ``fix: Fix bug in module``
* ``docs: Update documentation``
* ``test: Add tests``
* ``refactor: Refactor code``
* ``chore: Update dependencies``

Pull Requests
~~~~~~~~~~~~~

1. Ensure all tests pass
2. Add tests for new features
3. Update documentation as needed
4. Keep PRs focused on a single change
5. Write a clear PR description

Testing
~~~~~~~

* Write unit tests for new code
* Maintain test coverage above 90%
* Use pytest fixtures for common setups
* Test edge cases and error handling

What to Contribute
------------------

Ideas for contributions:

* 🐛 **Bug fixes**: Fix reported issues
* ✨ **New features**: Add new algorithms or policies
* 📚 **Documentation**: Improve docs and examples
* 🧪 **Tests**: Increase test coverage
* 🎨 **Visualization**: Add plotting capabilities
* 🚀 **Performance**: Optimize algorithms
* 🔧 **Tools**: Add utilities and helpers

Need Ideas?
~~~~~~~~~~~

Check out:

* `GitHub Issues <https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/issues>`_
* Issues labeled "good first issue"
* Issues labeled "help wanted"

Questions?
----------

* Open a GitHub Issue for bugs or feature requests
* Start a GitHub Discussion for questions
* Contact the maintainers

License
-------

By contributing, you agree that your contributions will be licensed under the MIT License.
