"""Type stub file for IDE type inference and hover information."""
from typing import Callable, Dict, Any, Optional
import numpy as np
from numpy.typing import NDArray

# Cost table factory functions with full signatures
def create_poisson_table(
    n: int,
    domain: int,
    rate: float = 1.0,
    strength: float | None = None
) -> NDArray: ...

def create_random_int_table(
    n: int,
    domain: int,
    low: int = 0,
    high: int = 10
) -> NDArray: ...

def create_uniform_float_table(
    n: int,
    domain: int,
    low: float = 0.0,
    high: float = 1.0
) -> NDArray: ...

# CTFactories namespace - declare as static methods for proper IDE support
class CTFactories:
    """Cost table factory namespace. Use: CTFactories.RANDOM_INT"""

    @staticmethod
    def UNIFORM(n: int, domain: int, low: float = 0.0, high: float = 1.0) -> NDArray: ...

    @staticmethod
    def RANDOM_INT(n: int, domain: int, low: int = 0, high: int = 10) -> NDArray: ...

    @staticmethod
    def POISSON(n: int, domain: int, rate: float = 1.0, strength: float | None = None) -> NDArray: ...

# Helper functions
def get_ct_factory(factory: Callable | str) -> Callable: ...
def validate_engine_config(config: Dict[str, Any]) -> bool: ...
def validate_policy_config(config: Dict[str, Any]) -> bool: ...
def validate_convergence_config(config: Dict[str, Any]) -> bool: ...
def get_validated_config(config_type: str, user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...

# Registries
CT_FACTORIES: Dict[str, Callable]
