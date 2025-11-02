# Torch Integration (Soft-Min Computator)

This adds a differentiable **soft-min** computator implemented in PyTorch that plugs
into the standard `BPEngine` loop. Replace the computator when constructing the
engine:

```python
from propflow import BPEngine, FGBuilder
from propflow.configs import CTFactory
from propflow.nn.torch_computators import SoftMinTorchComputator

fg = FGBuilder.build_cycle_graph(
    num_vars=6, domain_size=3,
    ct_factory=CTFactory.random_int.fn, ct_params={"low": 0, "high": 50}
)

engine = BPEngine(factor_graph=fg, computator=SoftMinTorchComputator(tau=0.2))
engine.run(max_iter=50)
```

**Notes**
- Soft-min uses `-τ·logsumexp(-x/τ)` as a smooth approximation of `min(x)`.
- When `τ → 0`, behaviour approaches Min-Sum. Larger `τ` increases smoothing.
- Requires `torch`; install via `pip install 'propflow[torch]'`.
