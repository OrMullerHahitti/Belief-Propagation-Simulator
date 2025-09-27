"""A factory for building factor graphs from configuration files.

This module defines the `FactorGraphBuilder` class, which is responsible for
loading a `GraphConfig` from a file and using its parameters to construct
a `FactorGraph` object. It relies on registries of graph builders and cost
table factories to dynamically create graphs with different topologies and
cost structures.
"""
from __future__ import annotations
import pickle
import os
from pathlib import Path
from importlib import import_module
from typing import Callable, Any

from ...configs.global_config_mapping import GRAPH_TYPES, CT_FACTORIES
from .create_factor_graph_config import ConfigCreator, GraphConfig
from ...bp.factor_graph import FactorGraph
from ..path_utils import find_project_root


def _resolve(dotted: str) -> Any:
    """Imports a dotted path and returns the final attribute.

    Args:
        dotted: A string representing a dotted path (e.g., 'my_module.my_class').

    Returns:
        The imported attribute.
    """
    mod, attr = dotted.rsplit(".", 1)
    return getattr(import_module(mod), attr)


def _next_index(base: Path, stem: str) -> int:
    """Finds the next available integer suffix for a filename.

    This is used to create incrementally numbered files (e.g., `graph-0`, `graph-1`).

    Args:
        base: The directory to search for existing files.
        stem: The base name of the file to check for.

    Returns:
        The next integer to use as a suffix.
    """
    pattern = f"factor-graph-{stem}-number*.pkl"
    existing = sorted(base.glob(pattern), key=lambda p: int(p.stem.split("number")[-1]))
    if not existing:
        return 0
    last = existing[-1].stem.split("number")[-1]
    return int(last) + 1


class FactorGraphBuilder:
    """Builds and saves `FactorGraph` instances from configuration files."""

    def __init__(self, output_dir: str | Path = find_project_root() / "configs/factor_graphs"):
        """Initializes the FactorGraphBuilder.

        Args:
            output_dir: The directory where generated factor graph pickle files
                will be saved.
        """
        self.output_dir = Path(output_dir).expanduser().resolve()
        os.makedirs(self.output_dir, exist_ok=True)

    def build_and_save(self, cfg_path: str | Path) -> Path:
        """Builds a factor graph from a config file and saves it as a pickle.

        The method performs these steps:
        1. Loads the `GraphConfig` from the specified path.
        2. Resolves the graph builder and cost table factory functions from the registries.
        3. Calls the builder function to generate the graph components.
        4. Creates the `FactorGraph` instance.
        5. Saves the `FactorGraph` object to a new versioned pickle file.

        Args:
            cfg_path: The path to the pickled `GraphConfig` file.

        Returns:
            The absolute path to the newly created factor graph pickle file.
        """
        cfg: GraphConfig = ConfigCreator.load_config(cfg_path)
        graph_builder_fn: Callable[..., Any] = _resolve(GRAPH_TYPES[cfg.graph_type])
        ct_factory_fn: Callable[..., Any] = CT_FACTORIES[cfg.ct_factory_name]

        variables, factors, edges = graph_builder_fn(
            num_vars=cfg.num_variables,
            domain_size=cfg.domain_size,
            ct_factory=ct_factory_fn,
            ct_params=cfg.ct_factory_params,
            density=cfg.density,
        )

        fg = FactorGraph(variable_li=variables, factor_li=factors, edges=edges)

        cfg_stem = Path(cfg.filename()).stem
        index = _next_index(self.output_dir, cfg_stem)
        out_name = f"factor-graph-{cfg_stem}-number{index}.pkl"
        out_path = self.output_dir / out_name
        with out_path.open("wb") as fh:
            pickle.dump(fg, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return out_path

    def build_and_return(self, cfg_path: str | Path) -> FactorGraph:
        """Builds a factor graph from a config file and returns the object.

        This method is similar to `build_and_save` but returns the `FactorGraph`
        instance directly instead of saving it to a file.

        Args:
            cfg_path: The path to the pickled `GraphConfig` file.

        Returns:
            The constructed `FactorGraph` object.
        """
        cfg: GraphConfig = ConfigCreator.load_config(cfg_path)
        graph_builder_fn: Callable[..., Any] = _resolve(GRAPH_TYPES[cfg.graph_type])
        ct_factory_fn: Callable[..., Any] = CT_FACTORIES[cfg.ct_factory_name]

        variables, factors, edges = graph_builder_fn(
            num_vars=cfg.num_variables,
            domain_size=cfg.domain_size,
            ct_factory=ct_factory_fn,
            ct_params=cfg.ct_factory_params,
            density=cfg.density,
        )

        return FactorGraph(variable_li=variables, factor_li=factors, edges=edges)

    @staticmethod
    def load_graph(path: str | Path) -> FactorGraph:
        """Loads a pickled `FactorGraph` from a file.

        Args:
            path: The path to the factor graph pickle file.

        Returns:
            The loaded `FactorGraph` object.
        """
        with Path(path).open("rb") as fh:
            return pickle.load(fh)
