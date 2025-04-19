from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import inspect
import os
import sys
from typing import Any, Dict, Callable

########################################################################
# ---- 1. Internal registries, global configs  ----------------------------------------
########################################################################

GRAPH_TYPES: Dict[str, str] = {        # str  -> dotted‑path or import key
    #all the graph types builders will be registered here, and with build_... typing
    "cycle": "utils.create_factor_graphs_from_config.build_cycle_graph",
    "octet-variable": "my_bp.graph_builders.build_octet_variable", #TODO : implemnt octec-variable and octec-factor, not for MST
    "octet-factor": "my_bp.graph_builders.build_octet_factor",
}

COMPUTATORS: Dict[str, str] = {        # str -> dotted‑path to BPComputator subclass
    "max-sum": "bp_base.computators.MaxSumComputator",
    "min-sum": "bp_base.computators.MinSumComputator",
    "sum-product": "my_bp.computators.SumProductComputator",
}

CT_FACTORIES: Dict[str, Callable] = {}   # filled in by decorator below


def ct_factory(name: str):
    """Decorator to register a cost‑table factory under a short name."""
    def decorator(fn: Callable):
        CT_FACTORIES[name] = fn
        return fn
    return decorator

########################################################################
# ------ all the function creators here:
#########################################################################
#TODO : add the rest of the cost table factories which i already made, i think it would be better if theyre al in one place, can still leave the other one for future uses
#TODO: add docstrings to the functions
@ct_factory("random_int")
def create_random_int_table(n: int, domain: int, low: int = 0, high: int = 10):
    import numpy as np
    shape = (domain,) * n
    return np.random.randint(low=low, high=high, size=shape)

@ct_factory("uniform_float")
def create_uniform_float_table(n: int, domain: int, low: float = 0.0, high: float = 1.0):
    import numpy as np
    shape = (domain,) * n
    return np.random.uniform(low=low, high=high, size=shape)

########################################################################
# ---- 2. Project root determination -----------------------------------
########################################################################

def get_project_root() -> Path:
    """Return the path to the project root directory."""
    # Try to find the project root by looking for a specific directory/file
    current_path = Path(__file__).resolve().parent
    while current_path.name != "Belief_propagation_simulator_" and current_path != current_path.parent:
        current_path = current_path.parent

    # If we didn't find the project root, use the current file's parent
    if current_path == current_path.parent:
        # Fallback to assuming we're running from somewhere within the project
        current_path = Path(__file__).resolve().parent.parent

    return current_path

########################################################################
# ---- 3. Dataclass that represents *one* configuration ---------------
########################################################################

@dataclass(slots=True)
class GraphConfig:
    graph_type: str
    computator: str
    num_variables: int
    domain_size: int
    ct_factory_name: str
    ct_factory_params: Dict[str, Any]

    # Anything else you want (seed, max_iters…) can be added later.

    # ------------------------------------------------------------------
    def filename(self) -> str:
        """<computator>-<type>-<numV>-<factory><compactParams>.pkl"""
        param_str = ",".join(f"{k}{v}" for k, v in self.ct_factory_params.items())
        return f"{self.computator}-{self.graph_type}-{self.num_variables}-{self.ct_factory_name}{param_str}.pkl"


########################################################################
# ---- 4. High‑level helper to build + save a configs -------------------
########################################################################

class ConfigCreator:
    def __init__(self, base_dir: str | Path = "configs/factor_graph_configs"):
        # Convert relative path to absolute path using project root
        if not os.path.isabs(str(base_dir)):
            base_dir = get_project_root() / base_dir
        self.base_dir = Path(base_dir).expanduser().resolve()

    # ------------------------------------------------------------------
    def create_config(
        self,
        *,
        graph_type: str,
        computator: str,
        num_variables: int,
        domain_size: int,
        ct_factory: str,
        ct_params: Dict[str, Any] | None = None,
    ) -> Path:
        """Validate, build GraphConfig, dump to pickle, return full path."""
        ct_params = ct_params or {}

        self._validate(graph_type, computator, num_variables, domain_size, ct_factory, ct_params)

        cfg = GraphConfig(
            graph_type=graph_type,
            computator=computator,
            num_variables=num_variables,
            domain_size=domain_size,
            ct_factory_name=ct_factory,
            ct_factory_params=ct_params,
        )

        # Ensure directory exists
        os.makedirs(self.base_dir, exist_ok=True)

        # Pickle‑dump
        file_path = self.base_dir / cfg.filename()
        with file_path.open("wb") as fh:
            pickle.dump(cfg, fh, protocol=pickle.HIGHEST_PROTOCOL)

        return file_path

    # ------------------------------------------------------------------
    @staticmethod
    def load_config(path: str | Path) -> GraphConfig:
        with Path(path).open("rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------
    @staticmethod
    def _validate(
        graph_type: str,
        computator: str,
        num_variables: int,
        domain_size: int,
        ct_factory: str,
        ct_params: Dict[str, Any],
    ):
        if graph_type not in GRAPH_TYPES:
            raise ValueError(f"Unknown graph_type '{graph_type}'.  Allowed: {list(GRAPH_TYPES)}")

        if computator not in COMPUTATORS:
            raise ValueError(f"Unknown computator '{computator}'.  Allowed: {list(COMPUTATORS)}")

        if not isinstance(num_variables, int) or num_variables <= 0:
            raise ValueError("num_variables must be a positive int")

        if not isinstance(domain_size, int) or domain_size <= 0:
            raise ValueError("domain_size must be a positive int")

        if ct_factory not in CT_FACTORIES:
            raise ValueError(f"Unknown ct_factory '{ct_factory}'.  Allowed: {list(CT_FACTORIES)}")

        # Extra safety: make sure ct_factory signature matches given params
        sig = inspect.signature(CT_FACTORIES[ct_factory])
        for name in ct_params:
            if name not in sig.parameters:
                raise ValueError(f"Parameter '{name}' not accepted by CT factory '{ct_factory}'")

if __name__ == "__main__":
    # Use project root for relative paths
    config_path = get_project_root() / "configs/factor_graph_configs"
    ConfigCreator(config_path).create_config(graph_type="cycle",
                                             computator="max-sum",
                                             domain_size=3,
                                             num_variables=3,
                                             ct_factory="random_int",
                                             ct_params={"low": 2,
                                                        'high': 100})


__doc__="""this module is made to create a config file for the factor graph
the parameters are:
    graph_type: str
        type of the graph, e.g., cycle, octet-variable, octet-factor
    computator: str
        type of the computator to use, e.g., max-sum, min-sum, sum-product
    num_variables: int
        number of variables in the graph
    domain_size: int
        size of the domain for each variable
    ct_factory: str
        name of the cost table factory to use, e.g., random_int, uniform_float
    ct_params: dict
        parameters for the cost table factory
    to use the function ConfigCreator(str{path you want}).create_config(the parameters above as key :argument pair  
    ) to create a config file
    
    also, in this module, configure the mapping for both : CT_FACTORIES and GRAPH_TYPES , and COMPUTATORS
    for CT_FACTORIES register the functions at the number 2 section, and for GRAPH_TYPES and COMPUTATORS
    give the dotted path to the class you want to use.
"""

