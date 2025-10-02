# PropFlow User Guide
## A Top-Down, Conceptual Introduction to Belief Propagation with PropFlow

---

## Table of Contents

1. [What is PropFlow?](#1-what-is-propflow)
2. [Core Concepts](#2-core-concepts)
3. [Mental Model: How BP Works](#3-mental-model-how-bp-works)
4. [Your First Factor Graph](#4-your-first-factor-graph)
5. [Understanding Messages](#5-understanding-messages)
6. [Running Belief Propagation](#6-running-belief-propagation)
7. [Interpreting Results](#7-interpreting-results)
8. [Policies: Modifying BP Behavior](#8-policies-modifying-bp-behavior)
9. [Comparative Experiments](#9-comparative-experiments)
10. [Advanced Analysis](#10-advanced-analysis)
11. [Common Patterns](#11-common-patterns)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What is PropFlow?

### The Big Picture

PropFlow is a **research toolkit for experimenting with belief propagation** on factor graphs. Think of it as a laboratory where you can:
- Build constraint satisfaction and optimization problems as graphs
- Test different message-passing algorithms
- Compare how variations affect convergence and solution quality
- Visualize and analyze the dynamics of distributed computation

### Who Should Use PropFlow?

- **Researchers** studying distributed optimization algorithms
- **Students** learning about graphical models and message passing
- **Engineers** prototyping multi-agent coordination systems
- **Data scientists** exploring probabilistic inference methods

### What Problems Does It Solve?

PropFlow addresses problems that can be expressed as:
- **Constraint satisfaction**: Finding assignments that satisfy relationships between variables
- **Optimization**: Minimizing (or maximizing) a sum of local cost (or reward) functions
- **Inference**: Computing marginal probabilities in graphical models

**Examples**: Graph coloring, scheduling, resource allocation, MAP inference, distributed sensor networks

---

## 2. Core Concepts

### The Building Blocks

PropFlow operates on **three fundamental concepts**:

#### 2.1 Variables
- Represent **decision points** or **unknowns** in your problem
- Each variable has a **domain**: the set of possible values it can take
- Example: A scheduling variable with domain `{Monday, Tuesday, Wednesday}`

#### 2.2 Factors
- Represent **constraints** or **cost functions** between variables
- Encode which combinations of variable assignments are preferred (low cost) or penalized (high cost)
- Example: "These two meetings can't happen at the same time" → high cost for same-day assignments

#### 2.3 Messages
- **Information packets** exchanged between variables and factors
- Carry **beliefs** about which values are promising based on local information
- The core of how belief propagation works

### The Graph Structure

PropFlow uses a **bipartite factor graph**:
```
Variables ↔ Factors ↔ Variables
   (V)         (F)        (V)

Example:
  V1 ---- F_12 ---- V2
   |               |
  F_13           F_23
   |               |
  V3 --------------+
```

- Variables only connect to factors
- Factors only connect to variables
- Messages flow back and forth along these connections

---

## 3. Mental Model: How BP Works

### The Synchronous Dance

Belief propagation is like a **synchronized conversation** happening across the graph:

#### Phase 1: Variables Speak (Q-Messages)
Each variable looks at messages it received from its factor neighbors and computes new messages to send back:
- "Based on what my factors tell me, here's what I think about each of my possible values"
- These are called **Q-messages** (variable → factor)

#### Phase 2: Factors Listen and Respond (R-Messages)
Each factor receives Q-messages from all its variable neighbors and computes responses:
- "Given what all my variables think, here's my opinion about each value for each of you"
- These are called **R-messages** (factor → variable)

#### Phase 3: Update and Repeat
Variables update their beliefs based on R-messages, and the cycle repeats until:
- Beliefs **converge** (stop changing significantly)
- A **maximum number of iterations** is reached
- Some other **stopping criterion** is met

### Why Does This Work?

In **tree-structured** graphs, BP provably finds the optimal solution. In graphs with **cycles** (loops), BP is a heuristic that often works well in practice but isn't guaranteed to converge or find the optimum.

---

## 4. Your First Factor Graph

### Problem: Two-Variable Constraint

Let's solve a simple problem: "Assign values to X and Y such that they minimize a cost function."

```python
import numpy as np
from propflow import FactorGraph, VariableAgent, FactorAgent

# Step 1: Create variables
X = VariableAgent(name="X", domain=3)  # Can take values 0, 1, or 2
Y = VariableAgent(name="Y", domain=3)  # Can take values 0, 1, or 2

# Step 2: Define a cost table (prefer matching values)
def prefer_matching(num_vars=None, domain_size=None, **kwargs):
    """Cost is 0 when X==Y, higher when they differ"""
    return np.array([
        [0, 5, 10],  # X=0: costs for Y=0,1,2
        [5, 0, 5],   # X=1: costs for Y=0,1,2
        [10, 5, 0]   # X=2: costs for Y=0,1,2
    ])

# Step 3: Create a factor connecting X and Y
F_XY = FactorAgent(name="F_XY", domain=3, ct_creation_func=prefer_matching)

# Step 4: Build the factor graph
graph = FactorGraph(
    variables=[X, Y],
    factors=[F_XY],
    edges={F_XY: [X, Y]}  # F_XY connects to both X and Y
)
```

**What just happened?**
- We created two variables, each with 3 possible values
- We defined a cost table that penalizes mismatched assignments
- We built a graph with one factor connecting the two variables

---

## 5. Understanding Messages

### Message Anatomy

A message is a **vector** with one entry per domain value:

```python
# Example Q-message from X to F_XY
Q_message = [2.5, 1.0, 3.2]
#             ^    ^    ^
#            X=0  X=1  X=2
```

**Interpretation**: Lower values indicate "I think this assignment is better"

### Message Computation

#### Variable → Factor (Q-Messages)
```python
# Pseudocode for variable X computing Q-message to factor F
Q[v] = sum of R-messages from all OTHER factors to X, evaluated at value v
```

#### Factor → Variable (R-Messages)
```python
# Pseudocode for factor F computing R-message to variable X
R[x] = min over all combinations of OTHER variables:
         cost_table[x, y, z, ...] + sum of Q-messages from those variables
```

**Key insight**: Each agent uses information from **neighbors** to update beliefs

---

## 6. Running Belief Propagation

### Basic Execution

```python
from propflow import BPEngine

# Create an engine (default is Min-Sum BP)
engine = BPEngine(factor_graph=graph)

# Run for up to 20 iterations
engine.run(max_iter=20)

# Get results
print(f"Final assignments: {engine.assignments}")
print(f"Final cost: {engine.calculate_global_cost()}")
```

**Output**:
```
Final assignments: {'X': 0, 'Y': 0}  # or any matching pair
Final cost: 0.0
```

### The Engine Lifecycle

Each iteration follows a **6-phase cycle**:

1. **Variable Compute**: Calculate Q-messages
2. **Variable Send**: Dispatch Q-messages to factors
3. **Factor Compute**: Calculate R-messages
4. **Factor Send**: Dispatch R-messages to variables
5. **Bookkeeping**: Update costs, history, snapshots
6. **Convergence Check**: Decide whether to continue

---

## 7. Interpreting Results

### What You Get

After running BP, you can access:

#### 7.1 Assignments
```python
assignments = engine.assignments
# {'X': 1, 'Y': 1}  # Current best guess for each variable
```

#### 7.2 Global Cost
```python
cost = engine.calculate_global_cost()
# 0.0  # Sum of all factor costs under current assignment
```

#### 7.3 Beliefs (Marginals)
```python
beliefs = engine.get_beliefs()
# {'X': [0.01, 0.97, 0.02], 'Y': [0.01, 0.97, 0.02]}
# Normalized confidence for each value
```

#### 7.4 History
```python
costs_over_time = engine.history.costs
# [42.3, 35.1, 28.4, ..., 0.0]  # Cost trajectory
```

### Convergence Indicators

**Good convergence**:
- Cost decreases steadily
- Assignments stabilize
- Beliefs become concentrated (high confidence)

**Poor convergence**:
- Cost oscillates
- Assignments keep changing
- Beliefs remain diffuse

---

## 8. Policies: Modifying BP Behavior

### What are Policies?

Policies **modify the message-passing process** to improve convergence or solution quality. Think of them as "tuning knobs" for BP.

### Common Policies

#### 8.1 Damping
**Problem**: Messages oscillate wildly
**Solution**: Blend old and new messages

```python
from propflow import DampingEngine

engine = DampingEngine(
    factor_graph=graph,
    damping_factor=0.9  # 90% old, 10% new
)
engine.run(max_iter=50)
```

**Effect**: Smoother convergence, reduced oscillations

#### 8.2 Factor Splitting
**Problem**: Large factors dominate message flow
**Solution**: Split factor influence to reduce impact

```python
from propflow import SplitEngine

engine = SplitEngine(
    factor_graph=graph,
    split_factor=0.5  # Reduce factor influence by 50%
)
engine.run(max_iter=50)
```

**Effect**: More balanced information flow

#### 8.3 Cost Reduction
**Problem**: High costs obscure relative differences
**Solution**: Apply one-time discount to all costs

```python
from propflow import CostReductionOnceEngine

engine = CostReductionOnceEngine(
    factor_graph=graph,
    reduction_factor=0.8  # Reduce by 20%
)
engine.run(max_iter=50)
```

**Effect**: Can escape local minima

### Combining Policies

Some engines combine multiple policies:

```python
from propflow import DampingCROnceEngine

engine = DampingCROnceEngine(
    factor_graph=graph,
    damping_factor=0.9,
    reduction_factor=0.8
)
```

---

## 9. Comparative Experiments

### Why Compare?

Different BP variants work better on different problems. PropFlow's `Simulator` makes it easy to compare.

### Setting Up Comparisons

```python
from propflow import Simulator, FGBuilder, BPEngine, DampingEngine
from propflow.configs import CTFactory

# Step 1: Generate test problems
problems = [
    FGBuilder.build_cycle_graph(
        num_vars=10,
        domain_size=5,
        ct_factory=CTFactory.random_int.fn,
        ct_params={"low": 0, "high": 100}
    ) for _ in range(5)  # 5 different random problems
]

# Step 2: Define engine configurations
configs = {
    "Standard BP": {
        "class": BPEngine
    },
    "Damped (0.9)": {
        "class": DampingEngine,
        "damping_factor": 0.9
    },
    "Damped (0.5)": {
        "class": DampingEngine,
        "damping_factor": 0.5
    },
}

# Step 3: Run simulations
simulator = Simulator(configs, log_level="INFO")
results = simulator.run_simulations(problems, max_iter=100)

# Step 4: Visualize results
simulator.plot_results(verbose=True)
```

### Understanding Simulation Results

The simulator produces:
- **Cost trajectories** for each engine on each problem
- **Average performance** across all problems
- **Statistical summaries** (final costs, convergence rates)
- **Comparative plots** showing which approaches work best

---

## 10. Advanced Analysis

### Snapshot Recording

For deep analysis, record every iteration:

```python
from analyzer.snapshot_recorder import EngineSnapshotRecorder
from analyzer.snapshot_visualizer import SnapshotVisualizer

# Wrap engine with recorder
engine = BPEngine(factor_graph=graph)
recorder = EngineSnapshotRecorder(engine)

# Record execution
snapshots = recorder.record_run(max_steps=50)

# Save to disk
recorder.to_json("results/my_run.json")

# Visualize belief trajectories
viz = SnapshotVisualizer.from_object(snapshots)
viz.plot_argmin_per_variable(show=True)
```

### What Snapshots Capture

Each snapshot contains:
- All messages sent that iteration
- Variable assignments
- Global cost
- Neutral message counts (plateaus)
- Message directions and argmins

### Analysis Workflows

1. **Convergence diagnosis**: Plot cost + neutral message ratio over time
2. **Belief tracking**: Visualize how argmins change per variable
3. **Message flow analysis**: Identify bottlenecks or oscillations
4. **Comparative dynamics**: Compare how different engines explore the solution space

---

## 11. Common Patterns

### Pattern 1: Build-Run-Analyze

```python
# Build
graph = FGBuilder.build_random_graph(
    num_vars=20,
    domain_size=10,
    density=0.3,
    ct_factory=CTFactory.gaussian.fn
)

# Run
engine = DampingEngine(factor_graph=graph)
engine.run(max_iter=100)

# Analyze
print(f"Cost: {engine.calculate_global_cost()}")
print(f"Converged: {engine.history.converged}")
```

### Pattern 2: Batch Experimentation

```python
# Generate many problems
problems = [build_problem() for _ in range(50)]

# Test multiple configurations
results = simulator.run_simulations(problems, max_iter=200)

# Export results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("experiment_results.csv")
```

### Pattern 3: Custom Cost Tables

```python
def my_custom_costs(num_vars, domain_size, **kwargs):
    """Define problem-specific costs"""
    table = np.zeros((domain_size,) * num_vars)
    # Fill table with custom logic
    return table

factor = FactorAgent("F", domain=3, ct_creation_func=my_custom_costs)
```

### Pattern 4: Convergence Monitoring

```python
from propflow.snapshots import SnapshotsConfig

snap_config = SnapshotsConfig(
    compute_jacobians=True,
    compute_cycles=True,
    retain_last=10
)

engine = BPEngine(factor_graph=graph, snapshots_config=snap_config)
engine.run(max_iter=100)

# Check for cycles in message flow
latest = engine.latest_snapshot()
if latest.cycles:
    print("Warning: Detected message cycles!")
```

---

## 12. Troubleshooting

### Problem: BP Doesn't Converge

**Symptoms**: Costs oscillate, assignments keep changing

**Solutions**:
1. **Apply damping**: Start with `damping_factor=0.9`
2. **Increase iterations**: Some problems need 100s or 1000s of steps
3. **Try splitting**: Reduce factor dominance
4. **Check graph structure**: Lots of short cycles → harder to converge

### Problem: Poor Solution Quality

**Symptoms**: BP converges, but cost is high

**Solutions**:
1. **Try different engines**: Damping, splitting, cost reduction
2. **Increase iterations**: May be stuck in local minimum
3. **Use local search**: DSA or MGM for refinement
4. **Check problem formulation**: Are cost tables correct?

### Problem: Slow Performance

**Symptoms**: Simulations take too long

**Solutions**:
1. **Reduce problem size**: Fewer variables or smaller domains
2. **Use multiprocessing**: Simulator does this automatically
3. **Simplify cost tables**: Sparse representations when possible
4. **Profile code**: Use built-in performance tools

### Problem: Memory Issues

**Symptoms**: Out of memory errors

**Solutions**:
1. **Disable snapshots**: Only record when needed
2. **Reduce `retain_last`**: Keep fewer snapshots in memory
3. **Process in batches**: Don't load all problems at once
4. **Stream to disk**: Save snapshots incrementally

---

## Quick Reference Card

### Essential Imports
```python
from propflow import (
    FactorGraph,
    VariableAgent,
    FactorAgent,
    BPEngine,
    DampingEngine,
    FGBuilder,
    Simulator,
)
from propflow.configs import CTFactory
from analyzer.snapshot_recorder import EngineSnapshotRecorder
from analyzer.snapshot_visualizer import SnapshotVisualizer
```

### Minimal Example
```python
# Build
v1, v2 = VariableAgent("v1", domain=3), VariableAgent("v2", domain=3)
f = FactorAgent("f", domain=3, ct_creation_func=CTFactory.random_int.fn)
graph = FactorGraph([v1, v2], [f], edges={f: [v1, v2]})

# Run
engine = BPEngine(factor_graph=graph)
engine.run(max_iter=20)

# Results
print(engine.assignments, engine.calculate_global_cost())
```

---

## Next Steps

1. **Try the Jupyter notebook**: `examples/analyzer_complete_tutorial.ipynb`
2. **Read the handbook**: `docs/handbook/` for deployment details
3. **Explore examples**: `examples/` directory
4. **Run tests**: `pytest` to verify installation
5. **Join the community**: GitHub Discussions for questions

---

## Resources

- **GitHub**: [OrMullerHahitti/Belief-Propagation-Simulator](https://github.com/OrMullerHahitti/Belief-Propagation-Simulator)
- **Issues**: Report bugs or request features
- **Examples**: Working code in `examples/` directory
- **API Docs**: Inline docstrings throughout the code
- **Papers**: See README for references on belief propagation

---

**Welcome to PropFlow! Happy experimenting! 🌊**
