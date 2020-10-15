# HJI_Vehicle
This code is written by Pytorch based on https://github.com/Tenavi/HJB_NN. For our current application, please review the `problem_def.py` from `/examples/vehicle/`. If you want to see all the applied equation in the code, please review `HJI_Equation_Derivation_Rev1`.

#### Software recommendations:

python 3.6.10, scipy version 1.4.1, numpy version 1.14.5

### Main folder:

  * generate.py: generate data by solving BVPs using time-marching
  
  * test_sample.py: output the data set including state `X`, lambda `A`, value cost `V` and time `t` from `train_date.mat`(Please creat a new folder and put this `test_sample.py` and generated `train_data.mat` into your new folder, then run `test_sample.py`. The reason is that required in numpy version`test_sample.py` should be higher  than or equal to `1.15.4`, it is not consistent with required numpy version for current code) 

  * train.py: train NNs to model the value function

  * simulate.py: simulate the closed-loop dynamics of a system and compare with BVP solution.

  * simulate_noise.py: simulate the closed-loop dynamics with a zero-order-hold and measurement noise.

  * predict_value.py: use a NN to predict the value function on a grid.

  * test_time_march.py, test_warm_start.py: test the reliability and speed of time-marching and NN warm start
  
### examples/:

#### *This is the only folder with settings that need to be adjusted*

This folder contains examples of problems each in their own folder. Each of these folders must contain a file called **problem_def.py** which defines the dynamics, optimal control, and various other settings. Data, NN models, and simulation results are all found here. The examples/ folder also contains

  * choose_problem.py: modify this script to tell other scripts which problem to solve_bvp

  * problem_def_template.py: a basic scaffold for how to define problems which these scripts can use

### utilities/:

  * neural_networks.py: auxiliary file which contains classes implementing NNs for predicting initial-time and time-dependent value functions.

  * other.py: other commonly-used utility functions
  
### how to run the code(example:vehicle):
1. run `generate.py` to obtain the `data_train.mat` and `data_val.mat`.

2. Then input `1` when asking `What kind of data? Enter 0 for validation, 1 for training:`. 

3. After BVP solver solution, input `1` when asking `Save data? Enter 0 for no, 1 for yes:`

4. If it asks `Overwrite existing data? Enter 0 for no, 1 for yes:`, please input `1`.

5. Find `train_data.mat` in the `/examples/vehicle/`

6. Run `test_sample.py` in your another new folder to review the data including state `X`, lambda `A`, value cost `V` and time `t`.
