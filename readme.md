# When Shall I Be Empathetic? The Utility of Empathetic ParameterEstimation in Multi-Agent Interactions

Contributors: Yi Chen, Merry Tanner, Lei Zhang, Sunny Amatya, Yi Ren, Wenlong Zhang

## Overview of the repo

This repo is implemented with the empathetic and non-empathetic agents studied in the 2021 ICRA paper: https://arxiv.org/abs/2011.02047

Home directory contains the simulation code, including parameter estimation, motion planning and other utilities.

Neural network and HJI BVP formulation code can be found in the folder /HJI_Vehicle. Please refer to the readme file.

## Introduction

Human-robot  interactions  (HRI)  can  be  modeledas differential games with incomplete information, where eachagent   holds   private   reward   parameters.   Due   to   the   openchallenge in finding perfect Bayesian equilibria of such games,existing  studies  often  decouple  the  belief  and  physical  dy-namics  by  iterating  between  belief  update  and  motion  plan-ning.  Importantly,  the  robot’s  reward  parameters  are  oftenassumed  to  be  known  to  the  humans,  in  order  to  simplifythe   computation.   We   show   in   this   paper   that   under   thissimplification, the robot performs non-empathetic belief updateabout the humans’ parameters, which causes high safety risksin uncontrolled intersection scenarios. In contrast, we proposea model for empathetic belief update, where the agent updatesthe  joint  probabilities  of  all  agents’  parameter  combinations.The update uses a neural network that approximates the Nashequilibrial  action-values  of  agents.  We  compare  empatheticand  non-empathetic  belief  update  methods  on  a  two-vehicleuncontrolled intersection case with short reaction time. Resultsshow  that  when  both  agents  are  unknowingly  aggressive  (ornon-aggressive),  empathy  is  necessary  for  avoiding  collisionswhen agents have false believes about each others’ parameters.This paper demonstrates the importance of acknowledging theincomplete-information  nature  of  HRI.

## Video presentation

https://youtu.be/fOoF42ORAwk


## Notations
- beta: agent's parameter, composed of theta and lambda.
- theta: agent's reward/intent parameter
- lambda: agent's noise parameter, affects the Boltzmann distribution of the action probability
- empathetic / non-empathetic (E / NE): types of agent on whether they consider other agent's belief on its own parameter


## Simulation
### Aggressive Empathetic agents with Non-aggressive beliefs
<a href="url"><img src="./plot/movie_E_theta1=na_theta2=na_time_horizon=3.0.gif" height="400" width="400" ></a>


### Aggressive Non-empathetic agents with Non-aggressive beliefs (closer encounters)
<a href="url"><img src="./plot/movie_NE_theta1=na_theta2=na_time_horizon=3.0.gif" height="400" width="400" ></a>

## Instruction of reproducing the results <a name="instruction"></a>
In general, the simulation can be conducted by 
running main.py. 
### To generate the baseline simulation, use [none, none] for inference model, [bvp_baseline, bvp_baseline] for decision.
### To generate the empathetic simulation, use [bvp_2, none] for inference model, [bvp_empathetic, bvp_empathetic] for decision.
### To generate the empathetic simulation, use [bvp_2, none] for inference model, [bvp_non_empathetic, bvp_non_empathetic] for decision.
- The agent's parameters can be changed in main.py on line 60, 61.
- The initial belief can be changed in main.py on line 62, 63.
- The initial position of agents can be changed in environment.py, on line 187~190.
- The type of agent (decision) can be changed in main.py on line 53.

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





