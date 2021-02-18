# When Shall I Be Empathetic? The Utility of Empathetic ParameterEstimation in Multi-Agent Interactions

Contributors: Yi Chen, Merry Tanner, Lei Zhang, Sunny Amatya, Yi Ren, Wenlong Zhang

## Overview of the repo

This repo is implemented with the empathetic and non-empathetic agents studied in the ICRA paper


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

## Models

### BVP_empathetic

### BVP_non_empathetic

### Baseline

## Instruction of reproducing the results <a name="instruction"></a>
In general, the simulation can be conducted by 
running main.py. The motion planning methods can be 
edited by changing $loss\_style$ on line 34 and 38 
in main.py. Aggressiveness ($\theta$) of agents are 
determined on line 129 and 141 of constants.py; 
acceleration ability ($\alpha$) of agents on line 132 
and 144, while $\hat{\alpha}$ on line 133 and 145 of 
the same file. Courtesy weight ($\beta$) is on line 
60 in constants.py. All tables and figures 1-9 of 
the paper can be reproduced by changing these 
parameters and run main.py in the root folder. 
Figure 10 can be reproduced by main.py in 
/turning\_scene.