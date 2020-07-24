import os
from math import floor

import matplotlib.pylab as plt
import numpy as np
import seaborn as sb
import torch
from matplotlib import pyplot

from parameters import *
from constants import CONSTANTS as C
from arguments import get_args
from intersection_env import IntersectionEnv
from model import DQN
from set_nfsp_models import get_models

is_rl_enabled = True
# mode = 3

plt.rcParams['figure.figsize'] = (15.0, 15.0) # set default size of plots
font = {'family' : 'sans',
        'weight' : 'normal',
        'size' : 8}
plt.matplotlib.rc('font', **font)
pyplot.locator_params(axis='y', nbins=6)
pyplot.locator_params(axis='x', nbins=10)



def test(env, args, q_set, mode):
    heat_map_path = './heat_maps/' + args.mode + '/mode_' + str(mode) + '/'
    q_path = heat_map_path + 'q_values'
    action_path = heat_map_path + 'best_actions'

    if not os.path.exists(q_path):
        os.makedirs(q_path)

    if not os.path.exists(action_path):
        os.makedirs(action_path)

    if is_rl_enabled:
        # fname = ''
        # if 'inf' in args.mode:
        #     fname = 'runs/inf_0.25/inf_0.25_1400000.pth'
        # elif 'g1' in args.mode:
        #     fname = 'runs/g1_0.25/g1_0.25_1400000.pth'
        # elif 'g2' in args.mode:
        #     fname = 'runs/g2_0.25/g2_0.25_1400000.pth'
        fname = args.log_dir + args.mode + '/model_' + str(args.load_model_index) + '.pth'
        # fname = args.log_dir + args.mode + '/model_' + str(args.load_model_index) + '.pth'
        current_model = DQN(env, args).to(args.device)
        current_model.eval()

        if args.device == torch.device("cpu"):
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        if not os.path.exists(fname):
            raise ValueError("No model saved with name {}".format(fname))

        current_model.load_state_dict(torch.load(fname, map_location))
        position_n = 4.  # number of grid points - 1
        speed_n = 4.
        position_range = [C.Intersection.CAR_2.INITIAL_STATE[0] * 0.5,
                          C.Intersection.CAR_2.INITIAL_STATE[0] * 1.0]
        position_grid = np.arange(position_range[0], position_range[1],
                                  (position_range[1] - position_range[0]) / position_n)
        # print(position_grid)
        speed_grid = []
        for p in position_grid:
            max_speed = np.sqrt((p - 1 - C.CAR_LENGTH * 0.5) * 2. * abs(C.Intersection.MAX_DECELERATION))
            speed_grid.append(np.arange(0.1 * max_speed, 0.5 * max_speed, 0.4 * max_speed / speed_n))

        # print(speed_grid)

        #  define the position-speed grid for the ego car for plotting the Q map
        ego_position_n = 100.  # number of grid points - 1
        ego_speed_n = 100.
        ego_position_range = [- 1 - C.CAR_LENGTH * 0.5, C.Intersection.CAR_1.INITIAL_STATE[0]]
        ego_position_grid = np.arange(ego_position_range[0], ego_position_range[1],
                                      (ego_position_range[1] - ego_position_range[0]) / ego_position_n)
        # print(ego_position_grid)
        ego_speed_range = [0, C.Intersection.VEHICLE_MAX_SPEED]
        ego_speed_grid = np.arange(ego_speed_range[0], ego_speed_range[1],
                                   (ego_speed_range[1] - ego_speed_range[0]) / ego_speed_n)

        for position_other, speed_other in zip(position_grid, speed_grid):
            for speed_o in speed_other:
                ctr = 0
                other_state = [position_other, speed_o]
                # print('other_state: {}'.format([position_other, speed_o]))
                max_Q_grid = np.full((100, 100), 0)
                best_action_grid = np.full((100, 100), 0)
                filled = 0
                minq, maxq = np.inf, -np.inf
                for position in ego_position_grid:
                    for speed in ego_speed_grid:
                        state = [position, speed, position_other, speed_o, mode]  #na, na

                        index_x = (position - ego_position_range[0]) * ego_position_n / (
                                ego_position_range[1] - ego_position_range[0])
                        index_y = (speed - ego_speed_range[0]) * ego_speed_n / (ego_speed_range[1] - ego_speed_range[0])
                        index_x = int(floor(index_x))
                        index_y = int(floor(index_y))

                        # print(state)
                        Q = current_model.forward(torch.FloatTensor(state).to(args.device))
                        Q = Q.tolist()
                        # print('Q: {}'.format(Q))
                        max_q_index = Q.index(max(Q))
                        if max(Q) <= 0:
                            ctr += 1
                        max_Q_grid[index_y][index_x] = max(Q)

                        if minq > max(Q):
                            minq = max(Q)
                        if maxq < max(Q):
                            maxq = max(Q)

                        best_action_grid[index_y][index_x] = max_q_index
                        filled += 1

                print('ctr: {}, filled: {}, minQ: {}, maxQ: {}'.format(ctr, filled, minq, maxq))

                create_heat_map(other_state, max_Q_grid, q_path, 'Max Q-Values', ego_position_n, ego_speed_n,
                                ego_position_grid, ego_speed_grid)
                create_heat_map(other_state, best_action_grid, action_path, 'Best action', ego_position_n, ego_speed_n,
                                ego_position_grid, ego_speed_grid)


def create_heat_map(other_state, grid, path, title, ego_position_n, ego_speed_n, ego_position_grid,  ego_speed_grid):
    fig, ax = plt.subplots()
    sb.heatmap(grid, linewidth=0.5)
    ax.set_xticks(np.arange(ego_position_n))
    ax.set_yticks(np.arange(ego_speed_n))

    ax.set_xticklabels([round(elem, 2) for elem in ego_position_grid])
    ax.set_yticklabels([round(elem, 2) for elem in ego_speed_grid])
    #
    # # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_yticklabels(), rotation=90, ha="right",
    #          rotation_mode="anchor")

    ax.set_title(title)
    fig.tight_layout()
    # plt.show()
    plt.savefig(path + '/fig_' + str(other_state) + '.png')
    plt.close()


if __name__ =='__main__':
    args = get_args()
    action_size = 5
    (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a), \
    (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a) = get_models()
    q_set = [Q_na_na, Q_na_a, Q_a_na, Q_a_a]

    env = IntersectionEnv(control_style_ego, control_style_other,
                                                    time_interval, MAX_TIME / time_interval)

    for mode in range(0, 4):
        test(env, args, q_set, mode)
        print('---')