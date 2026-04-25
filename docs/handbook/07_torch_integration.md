# Torch Integration Status

The current `propflow` package does not include a `propflow.nn` module or a
Torch-backed computator. Older drafts referenced `SoftMinTorchComputator`, but
that API is not present in the implementation exported from `src/propflow`.

To experiment with differentiable BP today, implement a custom computator by
subclassing or wrapping `propflow.bp.computators.BPComputator`, then pass it to
`BPEngine(factor_graph=..., computator=...)`. Keep the computator API compatible
with:

- `compute_Q(messages)`
- `compute_R(cost_table, incoming_messages)`
- `compute_belief(messages, domain)`
- `get_assignment(belief)`

Document and test any future Torch module before adding it back to the public
guide, including the optional dependency declaration in `pyproject.toml`.
