# Belief-Propagation-Simulator

## Overview
The **Belief-Propagation-Simulator** is a Python toolkit for building and experimenting with belief propagation algorithms on factor graphs. It was designed for research and education purposes and provides a flexible framework for implementing and testing new policies and engine variants.

## Key Features
- **Belief Propagation Subproblems**: Simulates a variety of belief propagation variants.
- **Factor Graph Support**: Operates on arbitrary factor graphs with built-in tools for graph construction and configuration.
- **Extensible Framework**: Modular policy system designed to support additional inference algorithms.
- **Graph State Persistence**: Save and load graph states efficiently using pickle files.
- **Debugging and Logging**: Integrated logging for debugging and monitoring.

## Installation
Install the package from PyPI:

```bash
pip install belief-propagation-simulator
```

To work with the latest development version:

```bash
git clone https://github.com/OrMullerHahitti/Belief-Propagation-Simulator.git
cd Belief-Propagation-Simulator
pip install -e .
```

## Quick Start
The following example creates a tiny factor graph and runs a damping engine.

```python
from src.propflow import (
    FactorGraph,
    VariableAgent,
    FactorAgent,
    DampingEngine,
)

# create two variables
v1 = VariableAgent("v1", domain=2)
v2 = VariableAgent("v2", domain=2)


# simple factor cost table
def table(num_vars=None, domain_size=None, **kwargs):
    return np.array([[0, 1], [1, 0]])


f = FactorAgent("f", domain=2, ct_creation_func=table)

fg = FactorGraph(variable_li=[v1, v2], factor_li=[f], edges={f: [v1, v2]})
engine = DampingEngine(factor_graph=fg)
engine.run(max_iter=5)
```

## Documentation
See the project repository for detailed API documentation and additional examples.
