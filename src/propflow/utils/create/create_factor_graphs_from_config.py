from __future__ import annotations
import pickle
import os
from pathlib import Path
from importlib import import_module
from typing import Callable, Any

# Function to get project root directory


# ──────────────────────────────────────────────────────────────
# 1.  Registries – reuse the same ones from config_creator.py
# ──────────────────────────────────────────────────────────────
from src.propflow.configs.global_config_mapping import (
    GRAPH_TYPES,  # str  -> dotted path for a *graph‑topology* builder
    CT_FACTORIES,  # str  -> cost‑table factory fn
    # helper that can load configs
)
from src.propflow.utils.create.create_factor_graph_config import (
    ConfigCreator,
    GraphConfig,
)

# Optional: make sure agents & FactorGraph are importable
from src.propflow.bp.factor_graph import FactorGraph
from src.propflow.utils.path_utils import find_project_root


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

    def build_and_return(self, cfg_path: str | Path) -> FactorGraph:
        """
        Load <cfg_path> (a pickled GraphConfig), create a FactorGraph
        instance, return the FactorGraph object.
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

        return fg

    @staticmethod
    def load_graph(path: str | Path) -> FactorGraph:
        """Load a pickled FactorGraph from <path>."""
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


# ──────────────────────────────────────────────────────────────
# 4.  Example graph‑topology builder functions
#     (plug in as many as you like; just register them in GRAPH_TYPES)
# ──────────────────────────────────────────────────────────────


# finally register builder
GRAPH_TYPES.setdefault("cycle", "create_factor_graphs_from_config.py.build_cycle_graph")


# TODO: implement other functions to build different graph topologies
