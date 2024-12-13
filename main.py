from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import json

# Import or copy-paste the FactorGraph, VariableNode, and FactorNode classes here
# Assuming the above code is available in the same file or imported.

# FastAPI app instance
app = FastAPI()

# Global variable for the factor graph
factor_graph = None

# Request Models
class VariableRequest(BaseModel):
    name: str
    domain: List[int]

class FactorRequest(BaseModel):
    name: str
    variables: List[str]
    cost_function: Dict[str, float]  # Example: {"0,0": 0, "0,1": 1, "1,1": 0}

class RunRequest(BaseModel):
    alpha: Optional[float] = 0.0001
    max_iterations: Optional[int] = 50

# Initialize Factor Graph
@app.post("/initialize/")
def initialize_graph():
    global factor_graph
    factor_graph = FactorGraph([], [])
    return {"message": "Factor graph initialized successfully."}

# Add Variable Node
@app.post("/add_variable/")
def add_variable(variable: VariableRequest):
    global factor_graph
    if factor_graph is None:
        raise HTTPException(status_code=400, detail="Factor graph is not initialized.")
    variable_node = VariableNode(variable.name, variable.domain)
    factor_graph.variables[variable.name] = variable_node
    return {"message": f"Variable {variable.name} added successfully."}

# Add Factor Node
@app.post("/add_factor/")
def add_factor(factor: FactorRequest):
    global factor_graph
    if factor_graph is None:
        raise HTTPException(status_code=400, detail="Factor graph is not initialized.")
    # Find variables by name
    variables = [factor_graph.variables[var_name] for var_name in factor.variables]
    # Convert cost_function to a dict
    cost_dict = {
        tuple(map(int, k.split(','))): v for k, v in factor.cost_function.items()
    }
    cost_function = lambda assignment: cost_dict.get(
        tuple(assignment[var.name] for var in variables), float('inf')
    )
    factor_node = FactorNode(factor.name, variables, cost_function)
    factor_graph.factors[factor.name] = factor_node
    return {"message": f"Factor {factor.name} added successfully."}

# Run Min-Sum Algorithm
@app.post("/run/")
def run_algorithm(run_request: RunRequest):
    global factor_graph
    if factor_graph is None:
        raise HTTPException(status_code=400, detail="Factor graph is not initialized.")
    factor_graph.alpha = run_request.alpha
    factor_graph.max_iterations = run_request.max_iterations
    factor_graph.run_min_sum()
    return {
        "message": "Min-Sum algorithm executed.",
        "final_assignments": factor_graph.current_best_assignments(),
        "converged": factor_graph.iteration_log[-1]['converged']
    }

# Fetch Iteration Log
@app.get("/log/")
def get_log():
    global factor_graph
    if factor_graph is None:
        raise HTTPException(status_code=400, detail="Factor graph is not initialized.")
    return {"iteration_log": factor_graph.iteration_log}

# Save Iteration Log to File
@app.post("/save_log/")
def save_log(filename: str = "iteration_log.json"):
    global factor_graph
    if factor_graph is None:
        raise HTTPException(status_code=400, detail="Factor graph is not initialized.")
    factor_graph.save_log(filename)
    return {"message": f"Iteration log saved to {filename}"}
