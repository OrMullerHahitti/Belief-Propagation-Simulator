# AGENTS.md

## Agent Architecture in the Belief Propagation Simulator

This document describes the agent-based architecture of our modular belief propagation simulator. The system is designed around autonomous agents that represent nodes in factor graphs and communicate through message passing.

## Table of Contents

1. [Overview](#overview)
2. [Agent Hierarchy](#agent-hierarchy)
3. [Core Agent Types](#core-agent-types)
4. [Message Passing System](#message-passing-system)
5. [Computators](#computators)
6. [Agent Lifecycle](#agent-lifecycle)
7. [Extending Agents](#extending-agents)
8. [Policy Integration](#policy-integration)
9. [Best Practices](#best-practices)
10. [Performance Considerations](#performance-considerations)
11. [Example Usage](#example-usage)

## Overview

The simulator implements a distributed message-passing algorithm where each node in a factor graph is represented by an autonomous agent. These agents:

- Maintain local state (beliefs, assignments, message history)
- Compute and exchange messages with neighbors
- Apply configurable policies for behavior modification
- Support both synchronous and asynchronous execution

The architecture follows the **Single Responsibility Principle** - each agent type has a clear, focused role in the belief propagation process.

## Agent Hierarchy

```
Agent (DCOP_base.py)
├── BPAgent (agents.py) - Abstract BP node
    ├── VariableAgent - Represents variables/decisions
    └── FactorAgent - Represents constraints/factors
```

### Base Classes

#### `Agent` (DCOP_base.py)
The top-level abstract base class for any node in the DCOP problem.

```python
class Agent(ABC):
    def __init__(self, name: str, node_type: str = "general"):
        self.name = name              # Human-readable identifier
        self.type = node_type         # Agent type classification
        self._computator = None       # Message computation strategy
        self.mailer = None           # Message handling system
```

#### `BPAgent` (agents.py) 
Abstract base class for belief propagation nodes with message passing capabilities.

```python
class BPAgent(Agent, ABC):
    def __init__(self, name: str, node_type: str, domain: int):
        super().__init__(name, node_type)
        self.domain = domain          # Size of value domain
        self._history = []           # Message history
        self.mailer = MailHandler(domain)  # Message management
```

## Core Agent Types

### VariableAgent

Represents decision variables in the optimization problem. Each variable:

- Has a discrete domain of possible values
- Maintains beliefs about value preferences
- Computes Q-messages (variable-to-factor messages)
- Makes assignments based on incoming beliefs

```python
class VariableAgent(BPAgent):
    def __init__(self, name: str, domain: int):
        super().__init__(name, "variable", domain)
    
    @property
    def belief(self) -> np.ndarray:
        """Current belief distribution over domain values"""
        
    @property
    def curr_assignment(self) -> int:
        """Current optimal assignment (argmin of beliefs)"""
        
    def compute_messages(self) -> None:
        """Compute Q-messages to factor neighbors"""
```

**Key Responsibilities:**
- Sum incoming R-messages from factors
- Compute outgoing Q-messages (sum of all R-messages except to sender)
- Maintain belief state and current assignment
- Apply variable-specific policies (e.g., damping)

### FactorAgent

Represents constraints or factors in the optimization problem. Each factor:

- Stores a cost/utility table over connected variables
- Computes R-messages (factor-to-variable messages)
- Enforces constraints between variables

```python
class FactorAgent(BPAgent):
    def __init__(self, name: str, domain: int, ct_creation_func: Callable, 
                 param: Dict[str, Any] = None, cost_table: CostTable = None):
        super().__init__(name, "factor", domain)
        self.cost_table = cost_table
        self.connection_number = {}  # Maps variable names to dimensions
        
    def compute_messages(self) -> None:
        """Compute R-messages to variable neighbors"""
        
    def initiate_cost_table(self) -> None:
        """Create cost table based on specified distribution"""
```

**Key Responsibilities:**
- Store and manage multi-dimensional cost tables
- Compute marginal costs for each connected variable
- Handle cost table modifications (splitting, discounting)
- Track variable-to-dimension mappings

## Message Passing System

### Message Structure

```python
class Message:
    def __init__(self, data: np.ndarray, sender: Agent, recipient: Agent):
        self.data = data          # Belief/cost vector
        self.sender = sender      # Source agent
        self.recipient = recipient # Target agent
```

### MailHandler

Manages message routing, deduplication, and optional pruning:

```python
class MailHandler:
    def __init__(self, domain_size: int):
        self._incoming = {}       # Sender -> Message mapping
        self._outgoing = []       # Messages to send
        
    def receive_messages(self, message: Message):
        """Accept incoming message with optional pruning"""
        
    def send(self):
        """Deliver all outgoing messages"""
```

**Features:**
- **Deduplication**: Prevents duplicate messages from same sender
- **Pruning Support**: Optional message filtering based on similarity
- **Synchronization**: Ensures proper message ordering

### Message Types

1. **Q-messages** (Variable → Factor):
   - Sum of all R-messages received except from target factor
   - Represents variable's current belief state

2. **R-messages** (Factor → Variable):
   - Marginal cost for each variable value
   - Computed by minimizing over all other variables

## Computators

Computators implement the core message computation algorithms. They are strategy objects that can be swapped to change the BP variant.

### Base Computator

```python
class BPComputator:
    def __init__(self, reduce_func, combine_func):
        self.reduce_func = reduce_func    # np.min, np.max, etc.
        self.combine_func = combine_func  # np.add, np.multiply, etc.
    
    def compute_Q(self, messages: List[Message]) -> List[Message]:
        """Compute variable-to-factor messages"""
        
    def compute_R(self, cost_table, incoming_messages: List[Message]) -> List[Message]:
        """Compute factor-to-variable messages"""
```

### Available Computators

- **MinSumComputator**: For minimization problems (DCOP)
- **MaxSumComputator**: For maximization problems
- **Custom**: Implement your own reduce/combine functions

## Agent Lifecycle

The agent lifecycle follows a structured pattern within each iteration:

### 1. Message Computation Phase
```python
for var in variable_agents:
    var.compute_messages()     # Compute Q-messages
    apply_policies(var)        # Apply variable policies (damping, etc.)
```

### 2. Message Sending Phase
```python
for var in variable_agents:
    var.mailer.send()          # Send all Q-messages
    var.empty_mailbox()        # Clear for next iteration
```

### 3. Factor Processing Phase
```python
for factor in factor_agents:
    factor.compute_messages()   # Compute R-messages
    factor.mailer.send()       # Send all R-messages
    factor.empty_mailbox()     # Clear for next iteration
```

### 4. State Update Phase
```python
for var in variable_agents:
    belief = var.belief        # Update beliefs
    assignment = var.curr_assignment  # Update assignments
```

## Extending Agents

### Creating Custom Variable Agents

```python
class CustomVariableAgent(VariableAgent):
    def __init__(self, name: str, domain: int, special_param: float):
        super().__init__(name, domain)
        self.special_param = special_param
    
    def compute_messages(self) -> None:
        # Custom message computati