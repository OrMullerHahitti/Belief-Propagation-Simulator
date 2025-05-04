import os
import sys
import pytest
import logging
import numpy as np
import json
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

log_dir = 'test_logs'
logger = Logger(__name__, file=True)

@pytest.fixture
def simple_factor_graph():
    pickle_path = os.path.join(project_root, 'configs', 'factor_graphs',
                               'factor-graph-cycle-10-random_intlow1,high1000.4-number2.pkl')
    fg = load_pickle(pickle_path)
    logger.info("Factor graph created with nodes: %s", fg.G.nodes())
    return fg

def test_bp_engine_long_run(simple_factor_graph):
    fg = simple_factor_graph
    engine = BPEngine(factor_graph=fg)
    logger.info("BPEngine initialized.")
    dict = {factor.name:factor.cost_table for factor in fg.factors}
    # Run the engine and save results as JSON
    logger.info(f"\n{dict}\n")

    result_path = engine.run(max_iter=1000, save_json=True)
    logger.info("BP Engine run for 1000 iterations or until convergence.")




