
from fastapi import APIRouter, HTTPException
from ..models import FactorGraph

router = APIRouter()

factor_graph = None

@router.post("/initialize/")
async def initialize_graph():
    global factor_graph
    factor_graph = FactorGraph([], [])
    return {"message": "Factor graph initialized successfully."}