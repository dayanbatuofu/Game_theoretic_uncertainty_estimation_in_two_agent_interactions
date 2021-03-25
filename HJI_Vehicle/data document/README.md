# HJI_Vehicle
Please see paper: https://arxiv.org/pdf/2011.02047.pdf

#### Software recommendations:

python 3.6.10, torch 1.4.0, scipy 1.4.1, numpy 1.14.5, matplotlib 3.2.2

### a_a folder:

  * collect data when both vehicles are aggressive. We set four different time sampling (100, 200, 400 and 800) to generate the train and validation data. 
     Then collect and filter uncoverged solution 

  * Step 1: use generate.py in each HJI_Vechilce_#sample (# represents the sampling number) folder to collect train and validation data

  * Step 2: copy and paste the train and validation data for different sampling, and put them into the HJI_Vechicle_data_collection_a_a/examples/vehicle

  * Step 3: in the HJI_Vechicle_data_collection_a_a folder, run code data_collection.py and data_filter in order to filter the unconverged solutions

  * Step 4: run train.py to train the neural network

  * Step 5: copy and paste the neural network model V_model_a_a.mat from /examples/vehicle/tspan, and put it into the data_summary/examples/vehicle/tspan

  * This is the function description for some main code files:

  * generate.py: generate data by solving BVPs

  * examples/vehicle/problem_def.py: vehicles' dynamic, control input and PMP function definition

  * data_collection.py:  collect best solution from four different time sampling results

  * data_filter.py: filter unconverged solutions after data collection

  * train.py: train neural network to model the value V and co-state

  * utilities/network: vehicles' dynamic, control input and PMP function definition

### na_na folder:

  * collect data when both vehicles are non-aggressive. We set four different time sampling (100, 200, 400 and 800) to generate the train and validation data. 
     Then collect and filter uncoverged solution  

  * Step 1: use generate.py in each HJI_Vechilce_#sample (# represents the sampling number) folder to collect train and validation data

  * Step 2: copy and paste the train and validation data for different sampling, and put them into the HJI_Vechicle_data_collection_na_na/examples/vehicle

  * Step 3: in the HJI_Vechicle_data_collection_na_na folder, run code data_collection.py and data_filter in order to filter the unconverged solutions

  * Step 4: run train.py to train the neural network

  * Step 5: copy and paste the neural network model V_model_na_na.mat from /examples/vehicle/tspan, and put it into the data_summary/examples/vehicle/tspan 

  * This is the function description for some main code files:

  * generate.py: generate data by solving BVPs

  * examples/vehicle/problem_def.py: vehicles' dynamic, control input and PMP function definition

  * data_collection.py:  collect best solution from four different time sampling results

  * data_filter.py: filter unconverged solutions after data collection

  * train.py: train neural network to model the value V and co-state

  * utilities/network: vehicles' dynamic, control input and PMP function definition

### data_summary folder:

  * Ouput the Q based on approximated value V and co-state

  * NN_output.py:  output the Q based on vehicle's different behaviros (aggressive or non-aggressive)

  * sim_draw.py: intersection case simulation

  * trajectory.py: both vehicles' trajecotry with collision area