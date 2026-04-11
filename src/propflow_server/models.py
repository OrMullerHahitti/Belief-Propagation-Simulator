from typing import List, Dict, Union, Optional, Any
from pydantic import BaseModel, Field

class VariableSpec(BaseModel):
    name: str
    domain_size: int = 2
    unary_cost: Optional[List[float]] = None  # cost per domain value

class FactorSpec(BaseModel):
    name: str
    neighbors: List[str]
    # Cost table can be a list (unary) or list of lists (binary)
    cost_table: Union[List[float], List[List[float]]] 

class EngineConfig(BaseModel):
    max_iters: int = Field(default=10, ge=1)
    damping: float = Field(default=0.0, ge=0.0, lt=1.0)  # Q damping (var -> factor)
    r_damping: float = Field(default=0.0, ge=0.0, lt=1.0)  # R damping (factor -> var)
    engine_type: str = "min_sum"  # min_sum, max_sum, sum_product

class GraphSpec(BaseModel):
    variables: List[VariableSpec]
    factors: List[FactorSpec]
    config: EngineConfig = Field(default_factory=EngineConfig)

class SnapshotJSON(BaseModel):
    """JSON-friendly representation of EngineSnapshot."""
    step: int
    dom: Dict[str, List[str]]
    # Key is "src->dst"
    Q: Dict[str, List[float]]
    R: Dict[str, List[float]]
    assignments: Dict[str, int]
    global_cost: Optional[float] = None
    cost_tables: Dict[str, Union[List[float], List[List[float]]]]  # 1D or 2D
    cost_labels: Dict[str, List[str]]

class RunResponse(BaseModel):
    run_id: str
    total_steps: int
    snapshots: List[SnapshotJSON]
