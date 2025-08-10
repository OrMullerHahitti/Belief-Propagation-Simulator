"""Quick start example for Belief Propagation Simulator."""

import numpy as np
from propflow import (
    FactorGraph,
    VariableAgent,
    FactorAgent,
    DampingEngine,
)


def main() -> None:
    v1 = VariableAgent("v1", domain=2)
    v2 = VariableAgent("v2", domain=2)

    def table(num_vars=None, domain_size=None, **kwargs):
        return np.array([[0, 1], [1, 0]])

    f = FactorAgent("f", domain=2, ct_creation_func=table)
    fg = FactorGraph(variable_li=[v1, v2], factor_li=[f], edges={f: [v1, v2]})
    engine = DampingEngine(factor_graph=fg)
    engine.run(max_iter=5)


if __name__ == "__main__":
    main()
