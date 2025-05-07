import os
import sys
import pytest
import logging
import numpy as np
import json
import time
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Now import from utils
from utils.loading_utils import load_pickle

from bp_base.factor_graph import FactorGraph
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.components import Message
from bp_base.bp_engine import BPEngine
from configs.loggers import Logger

log_dir = "test_logs"
logger = Logger(__name__, file=True)


@pytest.fixture
def simple_factor_graph():
    # Use a smaller factor graph for testing
    pickle_path = os.path.join(
        project_root,
        "configs",
        "factor_graphs",
        "factor-graph-random-50-random_intlow1,high1000.3-number2.pkl",  # If this exists, otherwise keep the original
    )
    if not os.path.exists(pickle_path):
        # Fallback to original graph
        pickle_path = os.path.join(
            project_root,
            "configs",
            "factor_graphs",
            "factor-graph-random-20-random_intlow1,high1000.3-number2.pkl",
        )
    
    logger.info(f"Loading factor graph from: {pickle_path}")
    start_time = time.time()
    fg = load_pickle(pickle_path)
    logger.info(f"Graph loaded in {time.time() - start_time:.2f} seconds")
    
    # Log basic graph statistics
    logger.info(f"Graph has {len(fg.variables)} variables and {len(fg.factors)} factors")
    
    return fg


def test_bp_engine_long_run(simple_factor_graph):
    fg = simple_factor_graph
    logger.info("Creating BPEngine...")
    start_time = time.time()
    engine = BPEngine(factor_graph=fg)
    logger.info(f"BPEngine initialized in {time.time() - start_time:.2f} seconds")
    
    # Run just 1 or 2 iterations with timing
    logger.info("Starting BP Engine run (just 1 iteration)...")
    start_time = time.time()
    result_path = engine.run(max_iter=1000, save_json=False, save_csv=True)
    logger.info(f"BP Engine completed 1 iteration in {time.time() - start_time:.2f} seconds")

