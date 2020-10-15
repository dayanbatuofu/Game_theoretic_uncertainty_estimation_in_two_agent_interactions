"""
Use python main.py to execute!
Important files:
1. autonomous_vehicle: process and record agent info
2. inference_model: performs inference and prediction
3. decision_model: returns appropriate action for each agent
4. sim_draw: plots the simulation and results
5. >>> savi_simulation: executes the simulation <<<
Change the default values below to change actual model used!
"""
import os
import argparse
import utils
import torch as t
from environment import Environment
from savi_simulation import Simulation

parser = argparse.ArgumentParser()
"""
simulation parameters
"""
parser.add_argument('--sim_duration', type=int, default=100)  # time span for simulation
parser.add_argument('--sim_dt', type=int, default=0.1)  # time step in simulation
parser.add_argument('--sim_lr', type=float, default=0.1)  # learning rate
parser.add_argument('--sim_nepochs', type=int, default=100)  # number of training epochs
parser.add_argument('--save', type=str, default='./experiment')  # save dir
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

"""
environment parameters
"""
parser.add_argument('--env_name', type=str, choices=['test_intersection', 'trained_intersection', 'bvp_intersection',
                                                     'lane_change', 'merger'],
                    default='bvp_intersection')

"""
agent model parameters
"""
# choose inference model: none: complete information
parser.add_argument('--agent_inference', type=str, choices=['none', 'test_baseline', 'nfsp_baseline', 'empathetic',
                                                            'bvp_empathetic', 'trained_baseline_2U'],
                    default=['none', 'bvp_empathetic'])  # use only empathetic for our simulation
# choose decision model: complete_information: nash equilibrium with complete information
parser.add_argument('--agent_decision', type=str,
                    choices=['constant_speed', 'baseline', 'baseline2', 'complete_information',
                             'non-empathetic', 'empathetic',
                             'bvp_non-empathetic', 'bvp_empathetic'],
                    default=['bvp_empathetic', 'bvp_empathetic'])

"""
agent parameters (for the proposed s = <x0,p0(β),β†,∆t,l>), for 2 agent case
"""
# TODO: generalize these params
parser.add_argument('--agent_dt', type=int, default=1)  # time step in planning
parser.add_argument('--agent_intent', type=int, choices=[1, 1000], default=[1, 1])
parser.add_argument('--agent_noise', type=float, choices=[0.001, 0.005], default=[0.001, 0.001])
parser.add_argument('--agent_intent_belief', type=int, choices=[1, 1000], default=[1, 1])
parser.add_argument('--agent_noise_belief', type=float, choices=[0.001, 0.005], default=[0.001, 0.001])
parser.add_argument('--belief_weight', type=float, default=0.8)


# TODO: add agent decision args
# parser.add_argument('', type=str, choices=[], default=[])
args = parser.parse_args()


if __name__ == "__main__":
    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    device = t.device('cuda:' + str(args.gpu) if t.cuda.is_available() else 'cpu')
    if args.env_name == 'bvp_intersection':
        sim_par = {"theta": [5, 1],
                   "lambda": [0.001, 0.005],
                   "action_set": [-5, -2, 0, 2, 5],
                   }
    elif args.env_name == 'trained_intersection':
        sim_par = {"theta": [1, 1000],
                   "lambda": [0.001, 0.005],
                   "action_set": [-8, -4, 0, 4, 8],
                   }
    else:
        sim_par = {"theta": [1, 1000],
                   "lambda": [0.001, 0.005],
                   "action_set": [-8, -4, 0, 4, 8],
                   }

    e = Environment(args.env_name, sim_par, args.agent_intent, args.agent_noise, args.agent_intent_belief,
                    args.agent_noise_belief)
    assert len(args.agent_inference) == e.n_agents and len(args.agent_decision) == e.n_agents

    kwargs = {"env": e,
              "duration": args.sim_duration,
              "n_agents": e.n_agents,
              "inference_type": args.agent_inference,
              "decision_type": args.agent_decision,
              "sim_dt": args.sim_dt,
              "sim_lr": args.sim_lr,
              "sim_par": sim_par,
              "sim_nepochs": args.sim_nepochs,
              "belief_weight": args.belief_weight}
    s = Simulation(**kwargs)
    s.run()

    # add analysis stuff here
    # s.postprocess()



