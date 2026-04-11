import uuid
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import GraphSpec, RunResponse
from .simulation import run_simulation

app = FastAPI(title="PropFlow API")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store
RUN_STORE: Dict[str, RunResponse] = {}

@app.post("/api/run", response_model=RunResponse)
def create_run(spec: GraphSpec):
    try:
        snapshots = run_simulation(spec)
        run_id = str(uuid.uuid4())
        response = RunResponse(
            run_id=run_id,
            total_steps=len(snapshots),
            snapshots=snapshots
        )
        RUN_STORE[run_id] = response
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/run/{run_id}", response_model=RunResponse)
def get_run(run_id: str):
    if run_id not in RUN_STORE:
        raise HTTPException(status_code=404, detail="Run not found")
    return RUN_STORE[run_id]

@app.get("/api/run/{run_id}/step/{step}", response_model=dict) # Should define SnapshotJSON but saving circular import effort for now
def get_step(run_id: str, step: int):
    if run_id not in RUN_STORE:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run_data = RUN_STORE[run_id]
    
    # Find snapshot with matching step
    # Assuming snapshots are ordered and contiguous for now
    if step < 0 or step >= len(run_data.snapshots):
        raise HTTPException(status_code=404, detail="Step out of range")
        
    return run_data.snapshots[step]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
