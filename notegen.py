# Create a ready-to-run Jupyter notebook that visualizes:
# 1) a 3-cycle factor graph,
# 2) the Jacobian of the synchronous min-sum (soft-min) update without difference-normalization,
# 3) the Jacobian with per-message difference-normalization,
# 4) the message dependency graph.
#
# The notebook uses Graphviz for all visualizations.
#
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# 0) Title + overview
cells.append(
    nbf.v4.new_markdown_cell(
        """
# Cycle Graph → Jacobian (raw vs. difference) → Message Dependency Graph

This notebook builds a 3-variable cycle factor-graph and visualizes:
- the factor graph,
- the Jacobian of one synchronous **min-sum** (soft-min) message-passing update **without** difference-normalization,
- the Jacobian **with** per-message difference-normalization (subtracting the mean across states),
- the **message dependency graph**.

> Notes
> - Uses **Graphviz** for rendering. If Graphviz is missing, install the OS package and the Python package.
> - The factor-to-variable update uses a **soft-min** approximation with temperature `tau` for differentiability.
> - Pairwise factors only; domain size is small for clarity.
"""
    )
)

# 1) Dependencies
cells.append(
    nbf.v4.new_code_cell(
        """
# If needed, install the Python package interface for Graphviz.
# You may still need to install the Graphviz system package (dot).
import sys, subprocess
try:
    import graphviz  # noqa: F401
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "graphviz"])

import numpy as np
from graphviz import Graph, Digraph

np.set_printoptions(precision=3, suppress=True)
"""
    )
)

# 2) Build the cycle factor graph + helpers
cells.append(
    nbf.v4.new_code_cell(
        """
# Parameters
d = 3          # domain size per variable
tau = 0.7      # soft-min temperature (smaller -> closer to hard min)
seed = 0
rng = np.random.default_rng(seed)

# Variables and pairwise factors for a 3-cycle
variables = ["X1", "X2", "X3"]
factors = {
    "F12": ("X1", "X2"),
    "F23": ("X2", "X3"),
    "F31": ("X3", "X1"),
}

# Neighborhood maps
var_neighbors = {v: [] for v in variables}
for f, (a, b) in factors.items():
    var_neighbors[a].append(f)
    var_neighbors[b].append(f)

# Pairwise cost tables φ_ij(a, b)
phis = {f: rng.normal(loc=0.0, scale=1.0, size=(d, d)) for f in factors.keys()}

def difference_matrix(d: int) -> np.ndarray:
    \"\"\"Return Q = I - (1/d) 11^T used for 'difference' normalization over states.\"\"\"
    I = np.eye(d)
    J = np.ones((d, d)) / d
    return I - J

Qd = difference_matrix(d)

def softmin_message_and_jacobian(phi: np.ndarray, msg_in: np.ndarray, tau: float):
    \"\"\"
    Compute factor->variable soft-min message and its Jacobian wrt msg_in.
    Args:
        phi: (d x d) pairwise cost table for (xi, xj).
        msg_in: length-d vector for m[xj->f].
        tau: temperature.
    Returns:
        pre_out: length-d vector, BEFORE any difference-normalization.
        W: (d x d) Jacobian block d pre_out / d msg_in.
           Rows index xi-state 'a', columns index xj-state 'b'.
    \"\"\"
    # Energy E[a, b] = phi[a, b] + msg_in[b]
    E = phi + msg_in[None, :]
    # We need softmin over b for each a:  softmin(z) = -tau * log sum_b exp(-z_b / tau)
    # For numerical stability, use log-sum-exp trick:
    M = (-E / tau).max(axis=1, keepdims=True)
    Z = np.exp((-E / tau) - M)  # shape (d, d)
    S = Z.sum(axis=1, keepdims=True)  # shape (d, 1)
    pre_out = -tau * (np.log(S) + M.squeeze())  # shape (d,)

    # Derivative wrt msg_in[b] is softmax prob over b for each row a:
    # W[a, b] = exp(-E[a,b]/tau) / sum_b' exp(-E[a,b']/tau)
    W = (Z / S)  # shape (d, d)
    return pre_out, W

# Message set: for each factor (u, v) we have four directional messages:
# u->F, v->F, F->u, F->v .  Each message is a length-d vector.
all_messages = []
meta = {}  # message -> metadata

for f, (u, v) in factors.items():
    # var->fac
    m_uv = f\"{u}->{f}\"
    m_vu = f\"{v}->{f}\"
    # fac->var
    m_fu = f\"{f}->{u}\"
    m_fv = f\"{f}->{v}\"
    all_messages += [m_uv, m_vu, m_fu, m_fv]

    meta[m_uv] = {\"type\": \"v2f\", \"factor\": f, \"variable\": u}
    meta[m_vu] = {\"type\": \"v2f\", \"factor\": f, \"variable\": v}
    meta[m_fu] = {\"type\": \"f2v\", \"factor\": f, \"variable\": u, \"other\": v}
    meta[m_fv] = {\"type\": \"f2v\", \"factor\": f, \"variable\": v, \"other\": u}

# Indexing for the big Jacobian
msg_index = {m: i for i, m in enumerate(all_messages)}
M = len(all_messages) * d  # total dimension
"""
    )
)

# 3) Render the factor graph
cells.append(
    nbf.v4.new_code_cell(
        """
def render_factor_graph(variables, factors):
    g = Graph(\"FactorGraph\", engine=\"dot\")
    g.attr(rankdir=\"LR\", nodesep=\"0.6\", ranksep=\"0.5\")
    # Variable nodes
    for x in variables:
        g.node(x, shape=\"circle\", style=\"filled\", fillcolor=\"#e0e0e0\", fontsize=\"12\")
    # Factor nodes
    for f in factors:
        g.node(f, shape=\"box\", style=\"filled\", fillcolor=\"#ffffff\", fontsize=\"12\")
    # Edges
    for f, (a, b) in factors.items():
        g.edge(a, f)
        g.edge(b, f)
    return g

render_factor_graph(variables, factors)
"""
    )
)

# 4) Build Jacobian (raw vs. difference)
cells.append(
    nbf.v4.new_code_cell(
        """
def build_jacobian(phis, tau: float, use_difference: bool) -> np.ndarray:
    \"\"\"Construct the block Jacobian of one synchronous update.\n
    Rows: outputs; Cols: inputs. Each block is d x d.\n
    v2f output depends on f2v inputs from *other* factors into the same variable.\n
    f2v output depends on v2f input along the *same* factor from the other variable.\n
    If use_difference is True, apply Qd to the output side of each block.\n
    \"\"\"
    J = np.zeros((M, M))
    # Current messages (only needed for f2v soft-min Jacobian; we use zeros -> depends on φ)
    zero = np.zeros(d)

    for m_out in all_messages:
        out_type = meta[m_out][\"type\"]
        out_i0 = msg_index[m_out] * d
        out_i1 = out_i0 + d

        if out_type == \"v2f\":
            var = meta[m_out][\"variable\"]
            f_out = meta[m_out][\"factor\"]
            # Sum of incoming f2v messages from other factors into 'var'
            for f_in in var_neighbors[var]:
                if f_in == f_out:
                    continue
                m_in = f\"{f_in}->{var}\"
                in_j0 = msg_index[m_in] * d
                in_j1 = in_j0 + d
                B = np.eye(d)
                if use_difference:
                    B = Qd @ B
                J[out_i0:out_i1, in_j0:in_j1] = B

        elif out_type == \"f2v\":
            f = meta[m_out][\"factor\"]
            var = meta[m_out][\"variable\"]
            other = meta[m_out][\"other\"]
            # Depends on the opposite var->fac message along the same factor
            m_in = f\"{other}->{f}\"
            in_j0 = msg_index[m_in] * d
            in_j1 = in_j0 + d

            phi = phis[f]
            _pre, W = softmin_message_and_jacobian(phi, zero, tau)  # (d,), (d x d)
            B = W
            if use_difference:
                B = Qd @ B
            J[out_i0:out_i1, in_j0:in_j1] = B

    return J

J_raw = build_jacobian(phis, tau=tau, use_difference=False)
J_diff = build_jacobian(phis, tau=tau, use_difference=True)

J_raw.shape, J_diff.shape
"""
    )
)

# 5) Visualize Jacobian as message-to-message weighted graph
cells.append(
    nbf.v4.new_code_cell(
        """
def block_max_abs(J: np.ndarray, i_out: int, j_in: int) -> float:
    r0, r1 = i_out * d, (i_out + 1) * d
    c0, c1 = j_in * d, (j_in + 1) * d
    blk = J[r0:r1, c0:c1]
    return float(np.max(np.abs(blk)))

def jacobian_graph(J: np.ndarray, title: str, threshold: float = 1e-6):
    g = Digraph(title, engine=\"dot\")
    g.attr(rankdir=\"LR\", nodesep=\"0.4\", ranksep=\"1.0\")
    # Nodes: shape by type
    for m in all_messages:
        shape = \"ellipse\" if meta[m][\"type\"] == \"v2f\" else \"box\"
        fill = \"#e8f0fe\" if meta[m][\"type\"] == \"v2f\" else \"#fff3e0\"
        g.node(m, shape=shape, style=\"filled\", fillcolor=fill, fontsize=\"11\")
    # Edges: from input to output, labeled by block max-abs derivative
    n = len(all_messages)
    for i_out, m_out in enumerate(all_messages):
        for j_in, m_in in enumerate(all_messages):
            w = block_max_abs(J, i_out, j_in)
            if w > threshold:
                pen = 1.0 + 4.0 * min(1.0, w)
                g.edge(m_in, m_out, label=f\"{w:.2f}\", penwidth=str(pen))
    return g

jacobian_graph(J_raw, title=\"Jacobian_without_difference\")
"""
    )
)

# 6) Jacobian with difference-normalization
cells.append(
    nbf.v4.new_code_cell(
        """
jacobian_graph(J_diff, title=\"Jacobian_with_difference\")
"""
    )
)

# 7) Message dependency graph (unweighted)
cells.append(
    nbf.v4.new_code_cell(
        """
def dependency_graph(J: np.ndarray, title: str, threshold: float = 1e-6):
    g = Digraph(title, engine=\"dot\")
    g.attr(rankdir=\"LR\", nodesep=\"0.4\", ranksep=\"1.0\")
    for m in all_messages:
        shape = \"ellipse\" if meta[m][\"type\"] == \"v2f\" else \"box\"
        fill = \"#e8f0fe\" if meta[m][\"type\"] == \"v2f\" else \"#fff3e0\"
        g.node(m, shape=shape, style=\"filled\", fillcolor=fill, fontsize=\"11\")
    n = len(all_messages)
    for i_out in range(n):
        for j_in in range(n):
            r0, r1 = i_out * d, (i_out + 1) * d
            c0, c1 = j_in * d, (j_in + 1) * d
            blk = J[r0:r1, c0:c1]
            if np.any(np.abs(blk) > threshold):
                g.edge(all_messages[j_in], all_messages[i_out])
    return g

dependency_graph(J_raw, title=\"Message_Dependency_Graph\")
"""
    )
)

# 8) Show φ tables (optional)
cells.append(
    nbf.v4.new_code_cell(
        """
print(\"Pairwise cost tables φ (rows = target-var states, cols = other-var states):\\n\")
for f, phi in phis.items():
    print(f\"{f}:\\n{phi}\\n\")
"""
    )
)

nb["cells"] = cells

out_path = Path("cycle_bp_jacobian_graphviz.ipynb")
with out_path.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

str(out_path)
