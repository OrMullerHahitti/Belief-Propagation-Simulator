# DSA and MGM Search Algorithms

This document describes the DSA (Distributed Stochastic Algorithm) and MGM (Maximum Gain Message) search algorithms implemented in the `search` module.

## Overview

DSA and MGM are local search algorithms for distributed constraint optimization problems (DCOPs). They work by having agents iteratively improve their value assignments to minimize the global cost.

### Key Differences

- **DSA**: Agents make independent, probabilistic decisions simultaneously
- **MGM**: Agents coordinate to ensure only the agent with maximum local gain changes per iteration

## Implementation

### Core Components

1. **Computators**: `DSAComputator` and `MGMComputator` implement the algorithm logic
2. **Engines**: `DSAEngine` and `MGMEngine` orchestrate the search process  
3. **Agents**: `SearchVariableAgent` extends `VariableAgent` with search capabilities

### Usage Example

```python
from search import DSAEngine, MGMEngine, DSAComputator, MGMComputator
from bp_base.factor_graph import FactorGraph
from base_models.agents import VariableAgent, FactorAgent

# Create factor graph (example: two variables with binary constraint)
var1 = VariableAgent("x1", domain=2)
var2 = VariableAgent("x2", domain=2)

def create_cost_table(**kwargs):
    return np.array([[2.0, 1.0], [3.0, 4.0]])  # Prefers x1=0, x2=1

factor = FactorAgent("constraint", domain=2, ct_creation_func=create_cost_table)
factor.connection_number = {"x1": 0, "x2": 1}
factor.initiate_cost_table()

fg = FactorGraph({factor: [var1, var2]})

# Use DSA
dsa_computator = DSAComputator(probability=0.7)
dsa_engine = DSAEngine(factor_graph=fg, computator=dsa_computator)

# Run optimization
results = dsa_engine.run(max_iter=50)
print(f"Best assignment: {results['best_assignment']}")
print(f"Best cost: {results['best_cost']}")

# Use MGM  
mgm_computator = MGMComputator()
mgm_engine = MGMEngine(factor_graph=fg, computator=mgm_computator)

results = mgm_engine.run(max_iter=50)
print(f"Best assignment: {results['best_assignment']}")
print(f"Best cost: {results['best_cost']}")
```

## Algorithm Details

### DSA (Distributed Stochastic Algorithm)

DSA operates in these steps per iteration:

1. **Evaluation**: Each agent evaluates the cost of changing to each possible value
2. **Decision**: If an improvement is found, agent changes with probability `p`
3. **Update**: All agents update their assignments simultaneously

**Parameters**:
- `probability`: Probability of taking an improving move (default: 0.7)

**Characteristics**:
- Simple and distributed
- Can escape local optima due to randomness
- Multiple agents can change simultaneously
- Convergence not guaranteed but often effective in practice

### MGM (Maximum Gain Message)

MGM operates in phases per iteration:

1. **Gain Calculation**: Each agent calculates its maximum possible improvement (gain)
2. **Gain Exchange**: Agents share gains with neighbors
3. **Decision**: Only the agent with maximum gain in its neighborhood changes

**Characteristics**:
- Deterministic (given tie-breaking rules)
- Coordinated decisions prevent conflicts
- Guaranteed monotonic improvement  
- May get stuck in local optima

## Testing

Several test files demonstrate the implementations:

- `tests/test_search_basic.py`: Tests core computator functionality
- `tests/test_search_integration.py`: Tests agent integration and convergence
- `tests/test_search_engines.py`: Tests full engine functionality (requires dependencies)
- `search/demo_search.py`: Interactive demonstration

Run tests with:
```bash
python tests/test_search_basic.py
python tests/test_search_integration.py
python search/demo_search.py
```

## Architecture

### SearchComputator

Base class for search algorithm computators:

```python
class SearchComputator(Computator, ABC):
    @abstractmethod
    def compute_decision(self, agent: Agent, neighbors_values: Dict[str, Any]) -> Any:
        """Compute next value for agent based on neighbors."""
        
    @abstractmethod  
    def evaluate_cost(self, agent: Agent, value: Any, neighbors_values: Dict[str, Any]) -> float:
        """Evaluate local cost of assigning value to agent."""
```

### SearchVariableAgent

Extends `VariableAgent` with search-specific methods:

```python
class SearchVariableAgent(VariableAgent):
    def compute_search_step(self, neighbors_values: Dict[str, Any]) -> Any:
        """Compute next value using search computator."""
        
    def update_assignment(self):
        """Update assignment based on computed decision."""
        
    def get_neighbor_values(self, graph) -> Dict[str, Any]:
        """Get current values of neighboring variables."""
```

### Engines

- **DSAEngine**: Implements DSA algorithm flow with simultaneous decisions
- **MGMEngine**: Implements MGM algorithm with gain coordination phases

Both engines automatically extend variable agents with search capabilities and handle the algorithm-specific coordination logic.

## Performance Considerations

- **DSA**: Fast, simple, good for large problems but may take many iterations
- **MGM**: More complex coordination but often converges faster with guaranteed improvement
- Both scale well to distributed settings
- Memory usage is linear in number of variables and constraints

## Future Extensions

The modular design allows for easy extensions:

- **DSA variants**: Different probability strategies, adaptive probability
- **MGM variants**: k-size coalition, different tie-breaking rules  
- **Hybrid approaches**: Combining DSA and MGM phases
- **Advanced termination**: Convergence detection, cost thresholds

## References

- DSA: Zhang, W., Wang, G., Xing, Z., & Wittenburg, L. (2005). Distributed stochastic search and distributed breakout: properties, comparison and applications to constraint optimization problems in sensor networks.
- MGM: Maheswaran, R. T., Tambe, M., Bowring, E., Pearce, J. P., & Varakantham, P. (2004). Taking DCOP to the real world: efficient complete solutions for distributed multi-event scheduling.