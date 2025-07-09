# Belief Propagation Simulator - **PropFlow**

## Overview
The **Belief-Propagation-Simulator** is a Python toolkit for building and experimenting with belief propagation algorithms on factor graphs. It was designed for research and education purposes and provides a flexible framework for implementing and testing new policies and engine variants.

## Key Features
- **Belief Propagation Subproblems**: Simulates a variety of belief propagation variants.
- **Factor Graph Support**: Operates on arbitrary factor graphs with built-in tools for graph construction and configuration.
- **Extensible Framework**: Modular policy system designed to support additional inference algorithms.
- **Graph State Persistence**: Save and load graph states efficiently using pickle files.
- **Debugging and Logging**: Integrated logging for debugging and monitoring.

## perlimenery preliminary:
> Results for 3 different variants of Min-Sum - regular, dampened , and using damping + splitting for 30 problems each, 90 simulations overall, with each one running 5000 steps (iterations).
>  using the following parameters : only binary constraints, domain size -10 , density - 0.25 , 50 Variable Nodes (approxematily 306 Factor Nodes), and cost table generated from uniform integer 
> function with [100,200] 

![image](https://github.com/user-attachments/assets/f9b3c0a6-0059-43a2-9eed-c23b6e06c369)


## Installation **not yet published but as you can see package ready!**
Install the package from PyPI:

```bash
pip install propflow
```

To work with the latest development version:

```bash
git clone https://github.com/OrMullerHahitti/Belief-Propagation-Simulator.git
cd Belief-Propagation-Simulator
pip install -e .
```




