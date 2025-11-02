# How to Create Configs

Configs will include:

## Graph Config:
- **Config Type** - e.g., cycle-3, cycle-4, variable loop, factor loop
- **Config Name** - e.g., cycle-3-1, cycle-4-1, etc.
- **Factor Graph**:
 -  [ ] Computator
    - [ ] Type of Computator - e.g., sum, product, min, max
    - [ ] Number of Variables - will also be set from the config type
    - [ ] Number of Factors - will also be set from the config type
    - [ ] Number of Edges - will also be set from the config type
  - [ ] Domain - size of the message domain and subsequent domain of the factors shape
  - [ ] Cost table creator function:
    - Random function from NumPy - e.g., np.random.rand/randint/uniform/normal
    - Parameters for this function (for example: normal - (mean, std), uniform - (low, high))
    - n - number of connections which are set after creating the graph
    - Domain - domain of the graph - which is given in the graph config

> All graphs will include a saver function to save in pickle formats in the directory "saved configs"

## Engine:
- **Type of Engine** - e.g., max sum, min sum, max product
- **Policies for Messages** - damping, message reduction
- **Policies for Factor Graphs (Cost Tables)** - e.g., cost reduction policy (with should apply or when to apply)
- **stopping critiria** - as a class which we will call stopper or decider, will get curr and message and previous message and will decide if we should stop or not
- **Max Iterations** - max iterations for the engine

## Steps for creating new configs:

1. Define your graph type (cycle-3, cycle-4, etc.)
2. Choose a computator type (min-sum, max-sum, etc.)
3. Specify domain size for variables
4. Configure cost table parameters:
   - Select a numpy random function (uniform, normal, randint, etc.)
   - Set parameters for the random function
5. Create a config dictionary with all parameters
6. Use `create_and_save_factor_graph()` to build and save both the config and the graph
7. The graph will be saved as a pickle file in the "saved_configs/factor_graphs" directory

### Example:
```python
# Create a cycle-3 graph with min-sum computator
config = {
    'graph_type': 'cycle-3',
    'graph_name': 'my-cycle-3-graph',
    'computator_type': 'min-sum',
    'domain_size': 4,
    'cost_table_params': {
        'function_type': 'uniform',
        'params': {'low': -10, 'high': 10}
    }
}

# Create and save the graph
graph_path = create_and_save_factor_graph(config)
```
