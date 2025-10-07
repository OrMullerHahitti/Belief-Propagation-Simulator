from __future__ import annotations

from typing import List, Tuple
import numpy as np

from ..core.components import Message

try:
    import torch
    from torch import nn
except Exception as _e:  # keep import error to raise on first use
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = _e  # type: ignore
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch():
    if torch is None:
        raise ImportError(
            "SoftMinTorchComputator requires PyTorch. "
            "Install with `pip install torch` or `pip install 'propflow[torch]'`."
        ) from _TORCH_IMPORT_ERROR


class SoftMinTorchComputator:
    """
    Differentiable soft-min computator (log-sum-exp smoothing) for factor→variable messages.

    Plug-in replacement for Min-Sum semantics inside PropFlow:
      - Q messages (variable→factor): additive combine of incoming R minus the recipient (numpy fast path)
      - R messages (factor→variable): soft-min over all other variables using PyTorch logsumexp

    Args:
        tau: temperature for soft-min; smaller ≈ harder min. Must be > 0.
        device: torch device; defaults to 'cuda' if available else 'cpu'
        dtype: torch dtype; defaults to torch.float32
    """

    def __init__(self, tau: float = 0.2, device: str | None = None, dtype=None):
        self.tau = float(max(tau, 1e-8))
        self._device = device
        self._dtype = dtype

    # ----- Variable → Factor (Q) : numpy fast path (no need for torch) -----
    def compute_Q(self, messages: List[Message]) -> List[Message]:
        if not messages:
            return []
        if len(messages) == 1:
            variable = messages[0].recipient
            return [
                Message(
                    np.zeros_like(messages[0].data),
                    sender=variable,
                    recipient=messages[0].sender,
                )
            ]
        variable = messages[0].recipient
        msg_data = np.stack([m.data for m in messages])  # (n_msgs, D)
        total = msg_data.sum(axis=0)
        out = [
            Message(data=total - msg_data[i], sender=variable, recipient=m.sender)
            for i, m in enumerate(messages)
        ]
        return out

    # ----- Factor → Variable (R) : soft-min via torch -----
    def compute_R(self, cost_table: np.ndarray, incoming_messages: List[Message]) -> List[Message]:
        if not incoming_messages:
            return []

        _require_torch()
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self._dtype or torch.float32

        k = cost_table.ndim
        shape = cost_table.shape

        ct = torch.as_tensor(cost_table, device=device, dtype=dtype)

        b_msgs: List["torch.Tensor"] = []
        axes_cache: List[Tuple[int, ...]] = []
        for axis, msg in enumerate(incoming_messages):
            q = torch.as_tensor(msg.data, device=device, dtype=dtype)
            br = q.reshape([shape[axis] if i == axis else 1 for i in range(k)])
            b_msgs.append(br)
            axes_cache.append(tuple(j for j in range(k) if j != axis))

        # Aggregate once: C + sum_j Q_j
        agg = ct
        for q in b_msgs:
            agg = agg + q

        outs: List[Message] = []
        for axis, br in enumerate(b_msgs):
            # Remove this recipient's Q before reducing (as in standard min-sum)
            temp = agg - br
            # Soft-min over all other axes
            r_vec = -self.tau * torch.logsumexp(-temp / self.tau, dim=axes_cache[axis])
            outs.append(
                Message(
                    data=r_vec.detach().cpu().numpy(),
                    sender=incoming_messages[axis].recipient,   # factor
                    recipient=incoming_messages[axis].sender,   # variable
                )
            )
        return outs

    # ----- Beliefs & assignment mirrors Min-Sum semantics -----
    def compute_belief(self, messages: List[Message], domain: int) -> np.ndarray:
        if not messages:
            return np.ones(domain) / domain
        belief = np.zeros(domain, dtype=float)
        for m in messages:
            belief += m.data
        return belief

    def get_assignment(self, belief: np.ndarray) -> int:
        return int(np.argmin(belief))


class SoftMaxSumPairwise(nn.Module):
    """
    Optional differentiable message-passing layer for pairwise graphs (GNN-style).
    Not used by the engine directly; provided for experiments and notebooks.

    Inputs:
      - x_beliefs: (n_vars, D)   current variable beliefs (log-space OK)
      - f_costs:   (m, D, D)     pairwise cost tables
      - edges:     (m, 2)        variable indices (i, j) per factor
    """

    def __init__(self, D: int, tau: float = 0.2, damp: float = 0.6):
        super().__init__()
        self.D = int(D)
        self.tau = float(max(tau, 1e-8))
        self.damp = float(damp)
        self.var_lin = nn.Linear(D, D, bias=False)
        self.fac_lin = nn.Linear(D, D, bias=False)

    def forward(self, x_beliefs: "torch.Tensor", f_costs: "torch.Tensor", edges: "torch.LongTensor"):
        i_idx, j_idx = edges[:, 0], edges[:, 1]
        v2f_i = self.var_lin(x_beliefs[i_idx])
        v2f_j = self.var_lin(x_beliefs[j_idx])

        scores_i = -f_costs + v2f_j.unsqueeze(1)              # (m, D, D)
        scores_j = -f_costs.transpose(1, 2) + v2f_i.unsqueeze(1)
        r_i = self.tau * torch.logsumexp(scores_i / self.tau, dim=-1)  # (m, D)
        r_j = self.tau * torch.logsumexp(scores_j / self.tau, dim=-1)
        r_i = self.fac_lin(r_i)
        r_j = self.fac_lin(r_j)

        accum = torch.zeros_like(x_beliefs)
        accum.index_add_(0, i_idx, r_i)
        accum.index_add_(0, j_idx, r_j)

        new_beliefs = (1 - self.damp) * x_beliefs + self.damp * (x_beliefs + accum)
        new_beliefs = new_beliefs - torch.logsumexp(new_beliefs, dim=-1, keepdim=True)
        return new_beliefs

