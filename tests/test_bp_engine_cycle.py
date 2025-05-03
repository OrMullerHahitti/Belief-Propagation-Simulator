import os
import sys
import colorlog
import pytest
import logging
import numpy as np

from bp_base.factor_graph import FactorGraph
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.components import Message
from bp_base.bp_engine import BPEngine
from utils.loading_utils import load_pickle, project_root
log_dir = 'test_logs'

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear any existing handlers
if root_logger.handlers:
    root_logger.handlers.clear()

# Create console handler with colored formatting
console_handler = colorlog.StreamHandler(sys.stdout)
console_handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
root_logger.addHandler(console_handler)

# Add file handler
file_handler = logging.FileHandler(os.path.join(log_dir, 'engine_check.log'))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(file_handler)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info("Logging is now set up with colored console output")
@pytest.fixture
def simple_factor_graph():
    pickle_path = os.path.join(project_root, 'configs', 'factor_graphs',
                               'factor-graph-random-10-random_intlow1,high1000.4-number6.pkl')

    # Create two variables and one factor for a minimal test


    # Connect both variables to the factor
    fg= load_pickle(pickle_path)
    logger.info("Factor graph created with nodes: %s", fg.G.nodes())
    return fg

def test_bp_engine_one_cycle(simple_factor_graph):
    fg = simple_factor_graph
    engine = BPEngine(factor_graph=fg)
    logger.info("BPEngine initialized.")

    # Run a single cycle
    cycle = engine.cycle(0)
    logger.info("Cycle executed.")

    # Check that steps exist
    assert len(cycle.steps) > 0
    logger.info("Cycle has %d steps.", len(cycle.steps))

    # Check that messages are present in each step
    for i, step in enumerate(cycle.steps):
        logger.info("Step %d: messages: %s", i, step.messages)
        assert isinstance(step.messages, dict)
        for agent_name, messages in step.messages.items():
            for msg in messages:
                assert isinstance(msg, Message)
                logger.info("Message from %s to %s: %s", msg.sender.name, msg.recipient.name, msg.data)

    # Check beliefs after one cycle
    beliefs = engine.get_beliefs()
    logger.info("Beliefs after one cycle: %s", beliefs)
    assert isinstance(beliefs, dict)
    for var, belief in beliefs.items():
        assert isinstance(belief, (np.ndarray, type(None)))

    # Check assignments
    assignments = engine.assignments
    logger.info("Assignments after one cycle: %s", assignments)
    assert isinstance(assignments, dict)
    for var, assignment in assignments.items():
        assert isinstance(assignment, (int, float, type(None), np.integer, np.floating))
