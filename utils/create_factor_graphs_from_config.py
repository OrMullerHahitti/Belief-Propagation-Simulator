from __future__ import annotations
import pickle
import os
import random
from pathlib import Path
from importlib import import_module
from typing import List, Dict, Callable, Any, Tuple
import networkx as nx
import re

# Function to get project root directory


# ──────────────────────────────────────────────────────────────
# 1.  Registries – reuse the same ones from config_creator.py
# ──────────────────────────────────────────────────────────────
from configs.global_config_mapping import (
    GRAPH_TYPES,  # str  -> dotted path for a *graph‑topology* builder
    CT_FACTORIES,  # str  -> cost‑table factory fn
    # helper that can load configs
)
from utils.create_factor_graph_config import ConfigCreator, GraphConfig

# Optional: make sure agents & FactorGraph are importable
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.factor_graph import FactorGraph
from utils.path_utils import find_project_root
from itertools import combinations


# ──────────────────────────────────────────────────────────────
# 2.  Helpers
# ──────────────────────────────────────────────────────────────
def _resolve(dotted: str) -> Any:
    """Import dotted path and return the attribute (module.attr)."""
    mod, attr = dotted.rsplit(".", 1)
    return getattr(import_module(mod), attr)


def _next_index(base: Path, stem: str) -> int:
    """Return the next integer suffix for files that start with <stem>."""
    pattern = f"factor-graph-{stem}-number*.pkl"
    existing = sorted(base.glob(pattern), key=lambda p: int(p.stem.split("number")[-1]))

    if not existing:
        return 0
    # Extract last number
    last = existing[-1].stem.split("number")[-1]
    return int(last) + 1


# ──────────────────────────────────────────────────────────────
# 3.  The high-level builder
# ──────────────────────────────────────────────────────────────
class FactorGraphBuilder:
    """Build & pickle a FactorGraph from a GraphConfig."""

    def __init__(
        self, output_dir: str | Path = find_project_root() / "configs/factor_graphs"
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------------------------------------
    def build_and_save(self, cfg_path: str | Path) -> Path:
        """
        Load <cfg_path> (a pickled GraphConfig), create a FactorGraph
        instance, pickle it, return full path to the new file.
        """
        # 1. Load config
        cfg: GraphConfig = ConfigCreator.load_config(cfg_path)

        # 2. Resolve callables/classes
        graph_builder_fn: Callable = _resolve(GRAPH_TYPES[cfg.graph_type])
        ct_factory_fn: Callable = CT_FACTORIES[cfg.ct_factory_name]

        # 3. Build agents & edges
        variables, factors, edges = graph_builder_fn(
            num_vars=cfg.num_variables,
            domain_size=cfg.domain_size,
            ct_factory=ct_factory_fn,
            ct_params=cfg.ct_factory_params,
            density=cfg.density,
        )

        # 4. Create the FactorGraph object
        fg = FactorGraph(variable_li=variables, factor_li=factors, edges=edges)

        # 5. Pickle‑dump under incremental filename
        cfg_stem = Path(
            cfg.filename()
        ).stem  # e.g.  max-sum-cycle-8-random_intlow0,high5
        index = _next_index(self.output_dir, cfg_stem)
        out_name = f"factor-graph-{cfg_stem}-number{index}.pkl"
        out_path = self.output_dir / out_name
        with out_path.open("wb") as fh:
            pickle.dump(fg, fh, protocol=pickle.HIGHEST_PROTOCOL)

        return out_path

    @staticmethod
    def load_graph(path: str | Path) -> FactorGraph:
        """Load a pickled FactorGraph from <path>."""
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


# ──────────────────────────────────────────────────────────────
# 4.  Example graph‑topology builder functions
#     (plug in as many as you like; just register them in GRAPH_TYPES)
# ──────────────────────────────────────────────────────────────
def _make_variable(idx: int, domain: int) -> VariableAgent:
    name = f"x{idx}"
    return VariableAgent(name=name, domain=domain)


def _make_factor(
    name: str, domain: int, ct_factory: Callable, ct_params: dict
) -> FactorAgent:
    # we postpone cost‑table creation until FactorGraph initialises
    return FactorAgent(
        name=name,
        domain=domain,
        ct_creation_func=ct_factory,
        param=ct_params,
    )


def _make_connections_density(
    variable_list: List[VariableAgent], density: float
) -> List[Tuple[VariableAgent, VariableAgent]]:
    """ """
    r_graph = nx.erdos_renyi_graph(len(variable_list), density)
    variable_map = {i: variable for i, variable in enumerate(variable_list)}
    full_graph = nx.relabel_nodes(r_graph, variable_map)
    return list(list(full_graph.edges()))


def _build_factor_edge_list(
    edges: List[Tuple[VariableAgent, VariableAgent]], domain_size, ct_factory, ct_params
) -> Dict[FactorAgent, List[VariableAgent]]:
    """
    Build a dictionary of edges from a list of edges.
    :param edges: List of edges
    :return: Dictionary of edges
    """
    edge_dict = {}
    for edge in edges:
        a, b = edge
        fname = f"f{a.name[1:]}{b.name[1:]}"
        fnode = _make_factor(fname, domain_size, ct_factory, ct_params)
        edge_dict[fnode] = [a, b]
    return edge_dict


def build_random_graph(
    num_vars: int,
    domain_size: int,
    ct_factory: Callable,
    ct_params: Dict[str, Any],
    density: float,
):
    """
    Build a random binary constraints graph.
    :param num_vars: Number of variables
    :param domain_size: Size of the domain
    :param ct_factory: Cost table factory
    :param ct_params: Cost table parameters
    :param density: Density of the graph
    :return: List of variables, list of factors, dictionary of edges
    """
    variables: List[VariableAgent] = [
        _make_variable(i + 1, domain_size) for i in range(num_vars)
    ]
    connections = _make_connections_density(variables, density)
    edges: Dict[FactorAgent, List[VariableAgent]] = _build_factor_edge_list(
        connections, domain_size, ct_factory, ct_params
    )
    factors = list(edges.keys())

    return variables, factors, edges


### ------------ IMPORTANT:  DO NOT CHANGE ------------------ ###
def build_cycle_graph(
    *,
    num_vars: int,
    domain_size: int,
    ct_factory: Callable,
    ct_params: Dict[str, Any],
    density: float,
):
    variables: List[VariableAgent] = [
        _make_variable(i + 1, domain_size) for i in range(num_vars)
    ]
    edges = []
    """
    Simple N‑variable cycle: x1–f12–x2–f23–…–xn–fn1–x1
    Returns (variables, factors, edges).
    """
    variables: List[VariableAgent] = [
        _make_variable(i + 1, domain_size) for i in range(num_vars)
    ]

    factors: List[FactorAgent] = []
    edges: Dict[FactorAgent, List[VariableAgent]] = {}

    for i in range(num_vars):
        a, b = variables[i], variables[(i + 1) % num_vars]
        fname = f"f{a.name[1:]}{b.name[1:]}"
        fnode = _make_factor(fname, domain_size, ct_factory, ct_params)
        factors.append(fnode)
        edges[fnode] = [a, b]

    return variables, factors, edges


# finally register builder
GRAPH_TYPES.setdefault("cycle", "create_factor_graphs_from_config.py.build_cycle_graph")


# TODO: implement other functions to build different graph topologies
