"""
Use python main.py to execute!
Important files:
1. autonomous_vehicle: process and record agent info
2. inference_model: performs inference and prediction
3. decision_model: returns appropriate action for each agent
4. sim_draw: plots the simulation and results
5. >>> savi_simulation: executes the simulation <<<
Change the DEFAULT values below to change actual model used!
"""
import os
import argparse
import utils
import torch as t
import numpy as np
import csv
from environment import Environment
from savi_simulation import Simulation

parser = argparse.ArgumentParser()
"""
simulation parameters
"""
parser.add_argument('--sim_duration', type=int, default=100)  # time span for simulation
parser.add_argument('--sim_dt', type=int, default=0.05)  # time step in simulation: choices: [0.01, 0.25, 1]
parser.add_argument('--sim_lr', type=float, default=0.1)  # learning rate
parser.add_argument('--sim_nepochs', type=int, default=100)  # number of training epochs
parser.add_argument('--save', type=str, default='./experiment')  # save dir
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

"""
environment parameters
"""
parser.add_argument('--env_name', type=str, choices=['trained_intersection', 'bvp_intersection'],
                    default='bvp_intersection')
# Starting position and velocity is set within the environment

"""
agent model parameters
"""

# choose inference model: use bvp for our experiment, and only for 1st player (i.e. ['bvp', 'none'])
parser.add_argument('--agent_inference', type=str, choices=['none', 'bvp'],
                    default=['bvp', 'none'])

# choose decision model: use the same model for the two agent, bvp_non_empathetic or bvp_empathetic
parser.add_argument('--agent_decision', type=str,
                    choices=['constant_speed', 'bvp_baseline', 'bvp_optimize',
                             'bvp_non_empathetic', 'bvp_empathetic'],
                    default=['bvp_empathetic', 'bvp_empathetic'])

"""
agent parameters (for the proposed s = <x0,p0(β),β†,∆t,l>), for 2 agent case
"""

parser.add_argument('--agent_dt', type=int, default=1)  # time step in planning (NOT IN USE)
parser.add_argument('--agent_intent', type=str, choices=['NA', 'A'], default=['A', 'A'])  # AGENT TRUE PARAM [P1, P2]
parser.add_argument('--agent_noise', type=str, choices=['N', 'NN'], default=['NN', 'NN'])
parser.add_argument('--agent_intent_belief', type=str, choices=['NA', 'A'], default=['A', 'A'])  # AGENT BELIEF
parser.add_argument('--agent_noise_belief', type=str, choices=['N', 'NN'], default=['NN', 'NN'])
parser.add_argument('--belief_weight', type=float, default=0.8)

# parser.add_argument('', type=str, choices=[], default=[])
args = parser.parse_args()


if __name__ == "__main__":
    loss_table = np.empty((6, 6))  # X1 by X2 initial states table
    policy_table_1 = np.empty((6, 6))  # record the policy choice of agent (correctness)
    policy_table_2 = np.empty((6, 6))
    startpos1 = np.empty((6, 6))  # for checking if starting condition is correct
    startpos2 = np.empty((6, 6))

    for i in range(len(loss_table)):  # iterate through rows (agent 1's init states)
        for j in range((len(loss_table[0]))):  # iterate through cols (agent 2's)
            "To run a single initial state, simple change this"
            x1 = 15 + i
            x2 = 15 + j
            startpos1[i][j] = x1
            startpos2[i][j] = x2

            initial_states = [[x1, 18], [x2, 18]]  # x1, v1, x2, v2
            utils.makedirs(args.save)
            logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
            logger.info(args)
            device = t.device('cuda:' + str(args.gpu) if t.cuda.is_available() else 'cpu')
            close_action_set = np.linspace(-5, 10, 31)
            close_action_set = close_action_set.tolist()
            # agent choices
            if args.env_name == 'bvp_intersection':
                sim_par = {"theta": [5, 1],  # NA, A
                           "lambda": [0.1, 0.5],  # N, NN
                           # "action_set": [-5, -3, -1, 0, 2, 4, 6, 8, 10],
                           "action_set": [-5, 0, 3, 7, 10],
                           # "action_set": close_action_set,
                           }
            elif args.env_name == 'trained_intersection':
                sim_par = {"theta": [1, 1000],  # NA, A
                           "lambda": [0.001, 0.005],  # N, NN
                           "action_set": [-8, -4, 0, 4, 8],
                           }
            else:
                sim_par = {"theta": [1, 1000],
                           "lambda": [0.001, 0.005],
                           "action_set": [-8, -4, 0, 4, 8],
                           }

            e = Environment(args.env_name, args.agent_inference, sim_par, initial_states, args.sim_dt, args.agent_intent, args.agent_noise,
                            args.agent_intent_belief,
                            args.agent_noise_belief)
            assert len(args.agent_inference) == e.N_AGENTS and len(args.agent_decision) == e.N_AGENTS

            kwargs = {"env": e,
                      "duration": args.sim_duration,
                      "n_agents": e.N_AGENTS,
                      "inference_type": args.agent_inference,
                      "decision_type": args.agent_decision,
                      "sim_dt": args.sim_dt,
                      "sim_lr": args.sim_lr,
                      "sim_par": sim_par,
                      "sim_nepochs": args.sim_nepochs,
                      "belief_weight": args.belief_weight}
            s = Simulation(**kwargs)
            if args.agent_inference[0] == 'bvp' or args.agent_inference[0] == 'bvp_continuous':
                loss, policy_count = s.run()
                loss_table[i][j] = loss
                policy_table_1[i][j] = policy_count[0]
                policy_table_2[i][j] = policy_count[1]
            else:
                loss = s.run()
                loss_table[i][j] = loss
    print("Loss table result: ", loss_table)
    print("Starting pos for P1: ", startpos1)
    print("Starting pos for P2: ", startpos2)
    print("Policy count for P1: ", policy_table_1)
    print("Policy count for P2: ", policy_table_2)

    "writing to csv file"
    filename = 'experiment/' + 'loss_table_' + str(args.agent_decision[0]) + '_' \
               + str(args.agent_intent[0]) + str(args.agent_intent[1]) + '_' +\
               str(args.agent_intent_belief[0]) + str(args.agent_intent_belief[1]) + '.csv'
    with open(filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for i in range(len(loss_table)):
            csv_writer.writerow(loss_table[i])

    filename2 = 'experiment/' + 'policy_' + str(args.agent_decision[0]) + '_' \
                + str(args.agent_intent[0]) + str(args.agent_intent[1]) + '_' + \
                str(args.agent_intent_belief[0]) + str(args.agent_intent_belief[1]) + '.csv'
    with open(filename2, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for i in range(len(policy_table_1)):
            csv_writer.writerow(policy_table_1[i])
        for i in range(len(policy_table_2)):
            csv_writer.writerow(policy_table_2[i])


    # add analysis stuff here
    # s.postprocess()



