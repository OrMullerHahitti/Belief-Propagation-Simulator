"""Utilities for creating and managing configuration files.

This module provides dataclasses (`GraphConfig`, `EngineConfig`) to represent
configurations in a structured way and a `ConfigCreator` class to validate,
build, and save these configurations as pickle files.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pickle
import inspect
import os
from typing import Any, Dict, List, Optional

from ...configs.global_config_mapping import GRAPH_TYPES, CT_FACTORIES
from ..path_utils import find_project_root


@dataclass(slots=True)
class EngineConfig:
    """A dataclass to hold the configuration for a simulation engine.

    Note:
        This class appears to be a work in progress and is not fully
        utilized in the current implementation.

    Attributes:
        computator: The name of the computator to use.
        factor_graph: The path to the factor graph configuration file.
        message_policies: A list of message-level policies to apply.
        factor_policies: A list of factor-level policies to apply.
    """
    computator: str
    factor_graph: Path | str
    message_policies: List[str]
    factor_policies: List[str]

    def filename(self) -> str:
        """Generates a descriptive filename for the engine configuration."""
        # Note: This method is incomplete and references attributes not in the dataclass.
        param_str = ",".join(f"{k}{v}" for k, v in getattr(self, "damping_params", {}).items())
        return f"{self.computator}-{self.factor_graph}-{getattr(self, 'damping_type', '')}{param_str}.pkl"


@dataclass(slots=True)
class GraphConfig:
    """A dataclass representing the configuration for a single factor graph.

    This object stores all the parameters needed to procedurally generate a
    factor graph.

    Attributes:
        graph_type: The topology of the graph (e.g., 'cycle', 'random').
        num_variables: The number of variable nodes in the graph.
        domain_size: The domain size for each variable.
        ct_factory_name: The name of the factory function for creating cost tables.
        ct_factory_params: A dictionary of parameters for the cost table factory.
        density: The density of the graph (for random graphs).
    """
    graph_type: str
    num_variables: int
    domain_size: int
    ct_factory_name: str
    ct_factory_params: Dict[str, Any]
    density: Optional[float] = None

    def filename(self) -> str:
        """Generates a descriptive filename for the graph configuration."""
        param_str = "".join(f"{k}{v}" for k, v in self.ct_factory_params.items())
        density_str = f"-{self.density}" if self.density is not None else ""
        return f"{self.graph_type}-{self.num_variables}-{self.ct_factory_name}{param_str}{density_str}.pkl"


class ConfigCreator:
    """A factory class for creating, saving, and loading graph configurations."""

    def __init__(self, base_dir: str | Path = "configs/factor_graph_configs"):
        """Initializes the ConfigCreator.

        Args:
            base_dir: The base directory where configuration files will be saved,
                relative to the project root.
        """
        if not os.path.isabs(str(base_dir)):
            base_dir = find_project_root() / base_dir
        self.base_dir = Path(base_dir).expanduser().resolve()

    def create_graph_config(
        self, *,
        graph_type: str,
        num_variables: int,
        domain_size: int,
        ct_factory: str,
        ct_params: Optional[Dict[str, Any]] = None,
        density: Optional[float] = None,
    ) -> Path:
        """Validates parameters, creates a `GraphConfig`, and saves it as a pickle file.

        Args:
            graph_type: The topology of the graph.
            num_variables: The number of variable nodes.
            domain_size: The domain size for each variable.
            ct_factory: The name of the cost table factory.
            ct_params: Parameters for the cost table factory.
            density: The density for random graphs.

        Returns:
            The absolute path to the newly created configuration file.
        """
        ct_params = ct_params or {}
        self._validate(graph_type, num_variables, domain_size, ct_factory, ct_params)

        cfg = GraphConfig(
            graph_type=graph_type, num_variables=num_variables, domain_size=domain_size,
            ct_factory_name=ct_factory, ct_factory_params=ct_params, density=density,
        )

        os.makedirs(self.base_dir, exist_ok=True)
        file_path = self.base_dir / cfg.filename()
        with file_path.open("wb") as fh:
            pickle.dump(cfg, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return file_path

    def create_engine_config(self, *args: Any, **kwargs: Any) -> None:
        """Placeholder for creating engine configurations.

        Note:
            This method is not fully implemented.
        """
        base_dir = find_project_root() / "configs/engine_configs"
        os.makedirs(self.base_dir, exist_ok=True)
        # Implementation for creating and saving EngineConfig would go here.

    @staticmethod
    def load_config(path: str | Path) -> GraphConfig:
        """Loads a `GraphConfig` object from a pickle file.

        Args:
            path: The path to the configuration file.

        Returns:
            The loaded `GraphConfig` object.
        """
        with Path(path).open("rb") as fh:
            return pickle.load(fh)

    @staticmethod
    def _validate(
        graph_type: str, num_variables: int, domain_size: int,
        ct_factory: str, ct_params: Dict[str, Any]
    ) -> None:
        """Validates the parameters for creating a graph configuration."""
        if graph_type not in GRAPH_TYPES:
            raise ValueError(f"Unknown graph_type '{graph_type}'. Allowed: {list(GRAPH_TYPES)}")
        if not isinstance(num_variables, int) or num_variables <= 0:
            raise ValueError("num_variables must be a positive int")
        if not isinstance(domain_size, int) or domain_size <= 0:
            raise ValueError("domain_size must be a positive int")
        if ct_factory not in CT_FACTORIES:
            raise ValueError(f"Unknown ct_factory '{ct_factory}'. Allowed: {list(CT_FACTORIES)}")

        sig = inspect.signature(CT_FACTORIES[ct_factory])
        for name in ct_params:
            if name not in sig.parameters:
                raise ValueError(f"Parameter '{name}' not accepted by CT factory '{ct_factory}'")
