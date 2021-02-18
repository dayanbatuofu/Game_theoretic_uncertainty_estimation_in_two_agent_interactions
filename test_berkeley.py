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
parser.add_argument('--sim_dt', type=int, default=1)  # time step in simulation
parser.add_argument('--sim_lr', type=float, default=0.1)  # learning rate
parser.add_argument('--sim_nepochs', type=int, default=100)  # number of training epochs
parser.add_argument('--save', type=str, default='./experiment')  # save dir
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

"""
environment parameters
"""
parser.add_argument('--env_name', type=str, default='single_agent')

"""
agent parameters
"""
# choose inference model: none: complete information
parser.add_argument('--agent_inference', type=str, choices=['none', 'baseline', 'empathetic'],
                    default=['none', 'baseline'])
# choose decision model: complete_information: nash equilibrium with complete information
parser.add_argument('--agent_decision', type=str,
                    choices=['constant_speed', 'complete_information', 'reactive_point', 'reactive_uncertainty'],
                    default=['constant_speed', 'constant_speed'])
parser.add_argument('--agent_dt', type=int, default=1)  # time step in planning

args = parser.parse_args()


if __name__ == "__main__":
    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    device = t.device('cuda:' + str(args.gpu) if t.cuda.is_available() else 'cpu')

    e = Environment(args.env_name)
    assert len(args.agent_inference) == e.N_AGENTS and len(args.agent_decision) == e.N_AGENTS

    kwargs = {"env": e,
              "duration": args.sim_duration,
              "n_agents": e.N_AGENTS,
              "inference_type": args.agent_inference,
              "decision_type": args.agent_decision,
              "sim_dt": args.sim_dt,
              "sim_lr": args.sim_lr,
              "sim_nepochs": args.sim_nepochs}
    s = Simulation(**kwargs)
    s.run()

    # add analysis stuff here
    # s.postprocess()



