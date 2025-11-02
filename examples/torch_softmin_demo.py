import numpy as np
from propflow import BPEngine, FGBuilder
from propflow.configs import CTFactories
from propflow.nn.torch_computators import SoftMinTorchComputator

if __name__ == "__main__":
    fg = FGBuilder.build_cycle_graph(
        num_vars=5,
        domain_size=3,
        ct_factory=CTFactories.RANDOM_INT,
        ct_params={"low": 1, "high": 20},
    )

    engine = BPEngine(factor_graph=fg, computator=SoftMinTorchComputator(tau=0.2))
    engine.run(max_iter=30)
    print("Assignments:", engine.assignments)
    print("Global cost:", engine.graph.global_cost)
