"""Step 1: Build and pickle 11 factor graphs with 0→100 random factors.

Run::

    uv run python expiriments/structured_vs_random/generate_graphs.py

Outputs
-------
graphs/graph_00.pkl  … graph_10.pkl   (11 FactorGraph objects)
metadata.json                          (seeds, parameters, replacement order)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

# sys.path guard so the script runs from anywhere in the repo
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from propflow import FGBuilder, FactorAgent, VariableAgent, create_random_int_table  # noqa: E402
from experiments.aij.code._ct_utils import _FixedCostTable  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────

TOPOLOGY_SEED    = 1234   # controls which variable pairs connect (erdos-renyi)
REPLACEMENT_SEED = 5678   # controls which factors get swapped per variant
RANDOM_CT_SEED   = 9999   # controls random cost-table values for replaced factors

NUM_VARS         = 40
DOMAIN           = 10
DENSITY          = 0.7    # dense random graph → many short cycles

N_VARIANTS       = 10     # 10 steps of 10% each: 0%, 10%, …, 100% randomised

CT_RANGE         = (100, 200)  # shared range for ALL cells (structured and random)

# ── fixed topology ─────────────────────────────────────────────────────────────

# build once just to extract the stable edge-pair names
_topo_graph = FGBuilder.build_random_graph(
    num_vars=NUM_VARS,
    domain_size=DOMAIN,
    ct_factory=create_random_int_table,
    ct_params={"low": 0, "high": 1},
    density=DENSITY,
    seed=TOPOLOGY_SEED,
)

# list of (var_name_a, var_name_b) — same for all 11 graphs
edge_pairs_names = [
    (vlist[0].name, vlist[1].name)
    for _factor, vlist in _topo_graph.edges.items()
]
n_factors = len(edge_pairs_names)
# each step randomises exactly 10% of all factors
REPLACEMENT_STEP = max(1, n_factors // N_VARIANTS)

# ── cost tables ───────────────────────────────────────────────────────────────

# structured CT: only CT[0,0]=0, everything else in [100, 200)
# factors prefer xi=0 AND xj=0; any other assignment costs at least 100
_rng_struct = np.random.RandomState(TOPOLOGY_SEED)
STRUCTURED_CT = _rng_struct.randint(CT_RANGE[0], CT_RANGE[1], size=(DOMAIN, DOMAIN)).astype(float)
STRUCTURED_CT[0, 0] = 0.0

# random CT: entire table in [100, 200), no zeros at all
# replacing a structured factor removes the only free (0,0) cell
_rng_ct = np.random.RandomState(RANDOM_CT_SEED)
random_tables = [
    _rng_ct.randint(CT_RANGE[0], CT_RANGE[1], size=(DOMAIN, DOMAIN)).astype(float)
    for _ in range(n_factors)
]

# pre-generate replacement order (which factor indices get swapped first)
_rng_repl = np.random.RandomState(REPLACEMENT_SEED)
replacement_order = list(_rng_repl.permutation(n_factors))


# ── builder ───────────────────────────────────────────────────────────────────

def build_variant(n_random: int) -> object:
    """Build a fresh FactorGraph with exactly n_random factors using random CTs."""
    random_set = set(replacement_order[:n_random])

    variables = [VariableAgent(f"x{i + 1}", domain=DOMAIN) for i in range(NUM_VARS)]
    var_map = {v.name: v for v in variables}
    edges = {}

    for idx, (name_a, name_b) in enumerate(edge_pairs_names):
        ct = random_tables[idx] if idx in random_set else STRUCTURED_CT.copy()
        # use a picklable callable instead of the lambda inside create_from_cost_table
        factor = FactorAgent(
            name=f"f{name_a[1:]}{name_b[1:]}",
            domain=DOMAIN,
            ct_creation_func=_FixedCostTable(ct),
            param=None,
            cost_table=ct,
        )
        edges[factor] = [var_map[name_a], var_map[name_b]]

    return FGBuilder.build_from_edges(variables, list(edges.keys()), edges)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    aij_dir = Path(__file__).resolve().parent.parent  # experiments/aij/
    data_dir = aij_dir / "data"
    graphs_dir = data_dir / "structured_vs_random_graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate_graphs] topology seed={TOPOLOGY_SEED}, n_factors={n_factors}")
    print(f"[generate_graphs] building {N_VARIANTS + 1} graph variants …")

    for i in range(N_VARIANTS + 1):  # 0 … 10
        n_random = i * REPLACEMENT_STEP
        fg = build_variant(n_random)
        pkl_path = graphs_dir / f"graph_{i:02d}.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(fg, fh)
        pct = round(100 * n_random / n_factors)
        print(f"  graph_{i:02d}.pkl  ({pct:3d}% random, {n_random} random / {n_factors} total)")

    # write metadata.json for reproducibility
    metadata = {
        "topology_seed": TOPOLOGY_SEED,
        "replacement_seed": REPLACEMENT_SEED,
        "random_ct_seed": RANDOM_CT_SEED,
        "num_vars": NUM_VARS,
        "domain": DOMAIN,
        "density": DENSITY,
        "n_factors": n_factors,
        "replacement_step": REPLACEMENT_STEP,
        "n_variants": N_VARIANTS,
        "ct_range": list(CT_RANGE),
        "structured_ct_note": "diagonal=0 (a,a free), off-diagonal in CT_RANGE",
        "random_ct_note": "entire table in CT_RANGE, no forced zeros",
        "replacement_order": [int(x) for x in replacement_order],
        "edge_pairs": edge_pairs_names,
    }
    meta_path = data_dir / "structured_vs_random_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"[generate_graphs] metadata written to {meta_path}")


if __name__ == "__main__":
    main()
