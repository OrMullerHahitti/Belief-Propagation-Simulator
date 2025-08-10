"""
Search module for implementing search-based algorithms like DSA and MGM.
This module extends the belief propagation framework to support search algorithms.
"""

# Import computators first (they have fewer dependencies)
from .search_computator import (
    SearchComputator,
    DSAComputator,
    MGMComputator,
)

# Conditionally import bp and agents (which need more dependencies)
try:
    from .search_engine import SearchEngine, DSAEngine, MGMEngine
    from .search_agents import (
        SearchVariableAgent,
        extend_variable_agent_for_search,
    )

    __all__ = [
        "SearchComputator",
        "DSAComputator",
        "MGMComputator",
        "SearchEngine",
        "DSAEngine",
        "MGMEngine",
        "SearchVariableAgent",
        "extend_variable_agent_for_search",
    ]
except ImportError:
    # If dependencies aren't available, just provide computators
    __all__ = ["SearchComputator", "DSAComputator", "MGMComputator"]
