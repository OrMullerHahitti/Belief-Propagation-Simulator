#!/bin/bash

# Function to kill child processes on exit
cleanup() {
    echo "Shutting down servers..."
    kill $(jobs -p)
    exit
}

trap cleanup SIGINT

# Start Backend
echo "Starting FastAPI Backend..."
cd src
uv run uvicorn propflow_server.main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Start Frontend
echo "Starting Vite Frontend..."
cd src/propflow_web
npm run dev -- --host &
FRONTEND_PID=$!
cd ../..

echo "Servers running!"
echo "Backend: http://localhost:8000/docs"
echo "Frontend: http://localhost:5173"

wait
