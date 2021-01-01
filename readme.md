# When Shall I Be Empathetic? The Utility of Empathetic ParameterEstimation in Multi-Agent Interactions

Contributors: Yi Chen, Merry Tanner, Lei Zhang, Sunny Amatya, Yi Ren, Wenlong Zhang

## Overview of the repo

## Models
 
## Instruction of reproducing the results <a name="instruction"></a>
In general, the simulation can be conducted by running main.py. The motion planning methods can be edited by changing $loss\_style$ on line 34 and 38 in main.py. Aggressiveness ($\theta$) of agents are determined on line 129 and 141 of constants.py; acceleration ability ($\alpha$) of agents on line 132 and 144, while $\hat{\alpha}$ on line 133 and 145 of the same file. Courtesy weight ($\beta$) is on line 60 in constants.py. All tables and figures 1-9 of the paper can be reproduced by changing these parameters and run main.py in the root folder. Figure 10 can be reproduced by main.py in /turning\_scene.