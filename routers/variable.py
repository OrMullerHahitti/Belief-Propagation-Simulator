
from fastapi import APIRouter, HTTPException
from ..models import VariableNode, VariableRequest

router = APIRouter()

@router.post("/add_variable/")
async def add_variable(variable: VariableRequest):
    global factor_graph
    if factor_graph is None:
        raise HTTPException(status_code=400, detail="Factor graph is not initialized.")
    variable_node = VariableNode(variable.name, variable.domain)
    factor_graph.variables[variable.name] = variable_node
    return {"message": f"Variable {variable.name} added successfully."}