
from fastapi import APIRouter, HTTPException
from ..models import RunRequest

router = APIRouter()

@router.post("/run/")
async def run_algorithm(run_request: RunRequest):
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

@router.get("/log/")
async def get_log():
    global factor_graph
    if factor_graph is None:
        raise HTTPException(status_code=400, detail="Factor graph is not initialized.")
    return {"iteration_log": factor_graph.iteration_log}

@router.post("/save_log/")
async def save_log(filename: str = "iteration_log.json"):
    global factor_graph
    if factor_graph is None:
        raise HTTPException(status_code=400, detail="Factor graph is not initialized.")
    factor_graph.save_log(filename)
    return {"message": f"Iteration log saved to {filename}"}

# Compare this snippet from app/tests/test_algorithm.py:







