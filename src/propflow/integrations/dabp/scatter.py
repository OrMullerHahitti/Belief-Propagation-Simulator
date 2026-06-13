"""Drop-in replacements for the three ``torch_scatter`` ops DABP uses.

The original DABP code imports ``scatter_add``, ``scatter_mean`` and
``scatter_softmax`` from the ``torch_scatter`` package, which ships compiled
extensions that are painful to install. We instead wrap the segment ops that
PyTorch Geometric maintains in pure Python (``torch_geometric.utils``), which
only needs ``torch`` + ``torch_geometric``. Semantics match torch_scatter for a
1-D ``index`` scattering along ``dim`` (the only usage in the model).
"""

from torch_geometric.utils import scatter as _pyg_scatter
from torch_geometric.utils import softmax as _pyg_softmax


def scatter_add(src, index, dim: int = 0, dim_size: int | None = None):
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")


def scatter_mean(src, index, dim: int = 0, dim_size: int | None = None):
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")


def scatter_softmax(src, index, dim: int = 0):
    return _pyg_softmax(src, index, dim=dim)
