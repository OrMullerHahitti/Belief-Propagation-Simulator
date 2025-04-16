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
