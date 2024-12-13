
from fastapi import APIRouter, HTTPException
from ..models import FactorNode, FactorRequest

router = APIRouter()

@router.post("/add_factor/")
async def add_factor(factor: FactorRequest):
    global factor_graph
    if factor_graph is None:
        raise HTTPException(status_code=400, detail="Factor graph is not initialized.")
    variables = [factor_graph.variables[var_name] for var_name in factor.variables]
    cost_dict = {tuple(map(int, k.split(','))): v for k, v in factor.cost_function.items()}
    cost_function = lambda assignment: cost_dict.get(tuple(assignment[var.name] for var in variables), float('inf'))
    factor_node = FactorNode(factor.name, variables, cost_function)
    factor_graph.factors[factor.name] = factor_node
    return {"message": f"Factor {factor.name} added successfully."}