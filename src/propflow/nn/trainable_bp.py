from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np

from ..core.agents import VariableAgent, FactorAgent
from ..bp.factor_graph import FactorGraph
from .torch_computators import SoftMinTorchComputator

try:
    import torch
    from torch import nn
    from torch.optim import Adam
except Exception as _e:
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = _e  # type: ignore
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch():
    if torch is None:
        raise ImportError(
            "TrainableBP requires PyTorch. "
            "Install with `pip install torch` or `pip install 'propflow[torch]'`."
        ) from _TORCH_IMPORT_ERROR


class TrainableBPModule(nn.Module):
    """
    Differentiable BP module with learnable cost tables for end-to-end training.

    This wrapper enables gradient-based optimization of factor cost tables by:
    1. Converting factor cost tables to learnable PyTorch parameters
    2. Running soft-min BP iterations with SoftMinTorchComputator
    3. Computing loss based on final assignments vs. ground truth
    4. Backpropagating gradients to update cost tables

    Args:
        factor_graph: The factor graph structure (variables, factors, edges)
        tau: Temperature for soft-min approximation (smaller = harder min)
        device: Torch device ('cuda' or 'cpu')
        dtype: Torch data type (defaults to float32)
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        tau: float = 0.2,
        device: str | None = None,
        dtype=None,
    ):
        _require_torch()
        super().__init__()

        self.fg = factor_graph
        self.tau = tau
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") # type: ignore
        self.dtype = dtype or torch.float32 # type: ignore

        # Create learnable cost tables as nn.Parameters
        self.cost_tables = nn.ParameterDict()
        for factor in self.fg.factors:
            # Initialize from current cost table
            ct_np = factor.cost_table.costs # type: ignore
            ct_tensor = torch.tensor(ct_np, device=self.device, dtype=self.dtype) # type: ignore
            self.cost_tables[factor.name] = nn.Parameter(ct_tensor)

        # Store variable info
        self.var_names = [v.name for v in self.fg.variables]
        self.var_domains = {v.name: v.domain for v in self.fg.variables}

        # Build edge mapping: factor_name -> list of variable names
        self.factor_to_vars: Dict[str, List[str]] = {}
        for factor in self.fg.factors:
            neighbors = list(self.fg.G.neighbors(factor))
            self.factor_to_vars[factor.name] = [v.name for v in neighbors]

    def forward(self, max_iter: int = 20) -> Tuple["torch.Tensor", Dict[str, int]]: # type: ignore
        """
        Run BP iterations and return final beliefs + assignments.

        Args:
            max_iter: Maximum number of BP iterations

        Returns:
            beliefs: Tensor of shape (n_vars, max_domain) with final belief values
            assignments: Dict mapping variable names to their argmin assignments
        """
        # Initialize messages: {(sender, recipient): tensor}
        messages: Dict[Tuple[str, str], "torch.Tensor"] = {} # type: ignore

        # Initialize all messages to zeros
        for factor_name, var_names in self.factor_to_vars.items():
            for var_name in var_names:
                domain = self.var_domains[var_name]
                # Q: var -> factor
                messages[(var_name, factor_name)] = torch.zeros( # type: ignore
                    domain, device=self.device, dtype=self.dtype
                )
                # R: factor -> var
                messages[(factor_name, var_name)] = torch.zeros(# type: ignore
                    domain, device=self.device, dtype=self.dtype
                )

        # BP iterations
        for _ in range(max_iter):
            new_messages = {}

            # Phase 1: Compute Q messages (variable -> factor)
            for var_name in self.var_names:
                domain = self.var_domains[var_name]
                # Get all incoming R messages from factors
                incoming_R = []
                for factor_name in self.factor_to_vars.keys():
                    if var_name in self.factor_to_vars[factor_name]:
                        incoming_R.append((factor_name, messages[(factor_name, var_name)]))

                if not incoming_R:
                    continue

                # Sum all incoming messages
                total = sum(r for _, r in incoming_R)

                # Send Q to each factor (total - R_from_that_factor)
                for factor_name, r_msg in incoming_R:
                    q_msg = total - r_msg
                    new_messages[(var_name, factor_name)] = q_msg

            # Phase 2: Compute R messages (factor -> variable) with soft-min
            for factor_name, var_names in self.factor_to_vars.items():
                ct = self.cost_tables[factor_name]
                k = len(var_names)

                if k == 0:
                    continue

                # Gather incoming Q messages
                q_msgs = [new_messages.get((vn, factor_name), messages[(vn, factor_name)])
                          for vn in var_names]

                # Build broadcasted Q tensors
                shape = [self.var_domains[vn] for vn in var_names]
                b_msgs = []
                axes_cache = []
                for axis, q in enumerate(q_msgs):
                    br = q.reshape([shape[axis] if i == axis else 1 for i in range(k)])
                    b_msgs.append(br)
                    axes_cache.append(tuple(j for j in range(k) if j != axis))

                # Aggregate: C + sum(Q)
                agg = ct
                for q in b_msgs:
                    agg = agg + q

                # Compute soft-min R messages
                for axis, (var_name, br) in enumerate(zip(var_names, b_msgs)):
                    temp = agg - br
                    r_vec = -self.tau * torch.logsumexp(-temp / self.tau, dim=axes_cache[axis])
                    new_messages[(factor_name, var_name)] = r_vec

            # Update messages
            messages.update(new_messages)

        # Compute final beliefs: sum of all incoming R messages
        beliefs_dict = {}
        for var_name in self.var_names:
            domain = self.var_domains[var_name]
            belief = torch.zeros(domain, device=self.device, dtype=self.dtype)
            for factor_name in self.factor_to_vars.keys():
                if var_name in self.factor_to_vars[factor_name]:
                    belief += messages[(factor_name, var_name)]
            beliefs_dict[var_name] = belief

        # Stack beliefs into tensor (pad to max domain size for batching)
        max_domain = max(self.var_domains.values())
        beliefs_list = []
        for var_name in self.var_names:
            b = beliefs_dict[var_name]
            if len(b) < max_domain:
                # Pad with large positive values (so argmin ignores them)
                padding = torch.full((max_domain - len(b),), 1e9, device=self.device, dtype=self.dtype)
                b = torch.cat([b, padding])
            beliefs_list.append(b)

        beliefs = torch.stack(beliefs_list)  # (n_vars, max_domain)

        # Get assignments
        assignments = {var_name: int(beliefs_dict[var_name].argmin())
                      for var_name in self.var_names}

        return beliefs, assignments

    def get_cost_tables_numpy(self) -> Dict[str, np.ndarray]:
        """Extract learned cost tables as numpy arrays."""
        return {name: param.detach().cpu().numpy()
                for name, param in self.cost_tables.items()}


class BPTrainer:
    """
    Training loop for optimizing cost tables via differentiable BP.

    Trains cost tables to minimize a loss function (e.g., total cost, assignment error).

    Args:
        model: TrainableBPModule instance
        learning_rate: Optimizer learning rate
        device: Torch device
    """

    def __init__(self, model: TrainableBPModule, learning_rate: float = 0.01, device: str | None = None):
        _require_torch()
        self.model = model
        self.device = device or model.device
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.loss_history: List[float] = []

    def compute_cost(self, assignments: Dict[str, int]) -> "torch.Tensor":
        """
        Compute total cost given assignments using learned cost tables.
        """
        total_cost = torch.tensor(0.0, device=self.device, dtype=self.model.dtype)

        for factor_name, var_names in self.model.factor_to_vars.items():
            ct = self.model.cost_tables[factor_name]
            # Get indices from assignments
            indices = tuple(assignments[vn] for vn in var_names)
            total_cost += ct[indices]

        return total_cost

    def train_step(self, max_iter: int = 20) -> float:
        """
        Single training step: run BP, compute loss, backprop, update.

        Args:
            max_iter: Number of BP iterations per forward pass

        Returns:
            loss_value: Scalar loss for this step
        """
        self.optimizer.zero_grad()

        # Forward pass: run BP
        beliefs, assignments = self.model(max_iter=max_iter)

        # Loss: total cost of current assignments
        loss = self.compute_cost(assignments)

        # Optional: add regularization to prevent cost tables from exploding
        # reg_loss = sum(torch.norm(ct) for ct in self.model.cost_tables.values()) * 0.001
        # loss = loss + reg_loss

        # Backward pass
        loss.backward()

        # Update cost tables
        self.optimizer.step()

        loss_val = float(loss.detach().cpu())
        self.loss_history.append(loss_val)

        return loss_val

    def train(
        self,
        num_epochs: int = 100,
        bp_iterations: int = 20,
        verbose: bool = True,
        convergence_threshold: float = 1e-4,
    ) -> Dict[str, np.ndarray]:
        """
        Full training loop.

        Args:
            num_epochs: Number of gradient descent epochs
            bp_iterations: BP iterations per forward pass
            verbose: Print progress
            convergence_threshold: Stop if loss change < threshold

        Returns:
            final_cost_tables: Learned cost tables as numpy arrays
        """
        if verbose:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Device: {self.device}, BP iterations: {bp_iterations}")

        best_loss = float("inf")
        patience_counter = 0
        patience = 10

        for epoch in range(num_epochs):
            loss = self.train_step(max_iter=bp_iterations)

            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

            # Track best loss
            if loss < best_loss - convergence_threshold:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"Converged at epoch {epoch} (loss: {loss:.6f})")
                break

        if verbose:
            print(f"Training complete. Final loss: {self.loss_history[-1]:.6f}")

        return self.model.get_cost_tables_numpy()

    def get_final_assignments(self, max_iter: int = 50) -> Dict[str, int]:
        """Run BP with learned cost tables and return final assignments."""
        self.model.eval()
        with torch.no_grad():
            _, assignments = self.model(max_iter=max_iter)
        return assignments
