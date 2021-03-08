# When Shall I Be Empathetic? The Utility of Empathetic ParameterEstimation in Multi-Agent Interactions

Contributors: Yi Chen, Merry Tanner, Lei Zhang, Sunny Amatya, Yi Ren, Wenlong Zhang

## Overview of the repo

This repo is implemented with the empathetic and non-empathetic agents studied in the 2021 ICRA paper: https://arxiv.org/abs/2011.02047

## Simulation
### Aggressive Empathetic agents with Non-aggressive beliefs
% ![alt text](./plot/movie_E_theta1=a_theta2=a_time_horizon=3.0.gif)
<a href="url"><img src="./plot/movie_E_theta1=a_theta2=a_time_horizon=3.0.gif" align="left" height="48" width="48" ></a>

### Aggressive Non-empathetic agents with Non-aggressive beliefs (closer encounters)
![alt text](./plot/movie_NE_theta1=a_theta2=a_time_horizon=3.0.gif)

## Instruction of reproducing the results <a name="instruction"></a>
In general, the simulation can be conducted by 
running main.py. 
- The agent's parameters can be changed in main.py on line 60, 61.
- The initial belief can be changed in main.py on line 62, 63.
- The initial position of agents can be changed in environment.py, on line 187~190.
- The type of agent can be changed in main.py on line 53.

## Models
Different agent decision models (agent type):
- BVP_empathetic: allows other agent to have misunderstanding of self
- BVP_non_empathetic: assumes other agent knows ego's parameter
- Baseline: no inference/interaction, only plays according to own parameter and what it assumes other agent to be 

### Main.py

Setting of the initial conditions of the simulation are done here. 

### environment.py

The simulation environment is generated here, using the parameters
from main.py. 

### savi_simulation.py

Initial conditions are processed for the simulation here, such as
agent parameters (beta) and action set. The initialization belief table 
is also done here through the function get_initial_belief(). 

### Inference_model.py

Inference is done after observing the state. There are several models
implemented here: bvp, baseline, etc. 
The inference algorithm updates the belief table at each time step using the 
selected model defined in main.py.

### Decision_model.py

Decision model returns an action for each agent, depending on the type
of agent defined in main.py. Models include bvp_empathetic, bvp_non_empathetic,
baseline, etc.

### draw_sim.py

The simulation data are collected and shown using the algorithm here
after the simulation has ended.





