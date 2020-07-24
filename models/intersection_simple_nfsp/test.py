import os
from math import floor

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pygame as pg
import seaborn as sb
import torch
from matplotlib import pyplot

from arguments import get_args
from common.utils import load_model_two
from intersection_env import IntersectionEnv
from model import DQN, Policy
from parameters import *
from constants import CONSTANTS as C
from tqdm import tqdm

plt.rcParams['figure.figsize'] = (15.0, 15.0)  # set default size of plots
font = {'family': 'sans',
        'weight': 'normal',
        'size': 8}
plt.matplotlib.rc('font', **font)
pyplot.locator_params(axis='y', nbins=6)
pyplot.locator_params(axis='x', nbins=10)

args = get_args()
# print_args(args)
# total_frames = 50000
total_eps = 10000
is_recording = False
recording_freq = 1
is_best_actions = False
is_same_player = False
player_id = 2
heatmap = False

recording_path = './test_recordings/' + args.mode + '/'  # + '_p1_' + str(is_best_actions) + '/'
# print(recording_path)

heat_map_path = './heat_maps/' + args.mode + '/'  # + '_p1_' + str(is_best_actions) + '/'
q_val_path1 = heat_map_path + 'p1_q'
best_action_path1 = heat_map_path + 'p1_a'
avg_action_path1 = heat_map_path + 'p1_avg'

q_val_path2 = heat_map_path + 'p2_q'
best_action_path2 = heat_map_path + 'p2_a'
avg_action_path2 = heat_map_path + 'p2_avg'

if is_recording:

    if not os.path.exists(recording_path):
        os.makedirs(recording_path)

if heatmap:
    if not os.path.exists(heat_map_path):
        os.makedirs(heat_map_path)

    if not os.path.exists(q_val_path1):
        os.makedirs(q_val_path1)

    if not os.path.exists(best_action_path1):
        os.makedirs(best_action_path1)

    if not os.path.exists(avg_action_path1):
        os.makedirs(avg_action_path1)

    if not os.path.exists(q_val_path2):
        os.makedirs(q_val_path2)

    if not os.path.exists(best_action_path2):
        os.makedirs(best_action_path2)

    if not os.path.exists(avg_action_path2):
        os.makedirs(avg_action_path2)


def test(env, args):
    if heatmap:
        analyze(p1_current_model, p1_policy, q_val_path1, best_action_path1, avg_action_path1)
        print('----')
        analyze(p2_current_model, p2_policy, q_val_path2, best_action_path2, avg_action_path2)
        print('---')

    episode_run(env, args)


def analyze(model, policy, q_path, action_path, avg_action_path):
    #  For each of the other car's state, plot a 2D Q map for ego car's position and speed.

    #  the other car's state is descretized as follows
    position_n = 4  # number of grid points - 1
    speed_n = 4
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
    ego_position_n = 100  # number of grid points - 1
    ego_speed_n = 100
    ego_position_range = [-3.0, C.Intersection.CAR_1.INITIAL_STATE[0]]
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
            print('{} '.format([position_other, speed_o]), end='')
            max_Q_grid = np.full((ego_position_n, ego_speed_n), 0)
            best_action_grid = np.full((ego_position_n, ego_speed_n), 0)
            avg_action_grid = np.full((ego_position_n, ego_speed_n), 0)
            filled = 0
            minq, maxq = np.inf, -np.inf
            for position in ego_position_grid:
                for speed in ego_speed_grid:
                    state = [position, speed, position_other, speed_o]

                    index_x = (position - ego_position_range[0]) * ego_position_n / (
                            ego_position_range[1] - ego_position_range[0])
                    index_y = (speed - ego_speed_range[0]) * ego_speed_n / (ego_speed_range[1] - ego_speed_range[0])
                    index_x = int(floor(index_x))
                    index_y = int(floor(index_y))

                    # print(state)
                    Q = model.forward(torch.FloatTensor(state).to(args.device))
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
                    avg_action_grid[index_y][index_x] = policy.act(torch.FloatTensor(state).to(args.device), 1)
                    filled += 1

            print('ctr: {}, filled: {}, minQ: {}, maxQ: {}'.format(ctr, filled, minq, maxq))

            create_heat_map(other_state, max_Q_grid, q_path, 'Max Q-Values', ego_position_n, ego_speed_n,
                            ego_position_grid, ego_speed_grid)
            create_heat_map(other_state, best_action_grid, action_path, 'Best action', ego_position_n, ego_speed_n,
                            ego_position_grid, ego_speed_grid)
            create_heat_map(other_state, avg_action_grid, avg_action_path, 'Avg action', ego_position_n, ego_speed_n,
                            ego_position_grid, ego_speed_grid)
            # return


def episode_run(env, args):
    p1_reward_list = []
    p2_reward_list = []
    length_list = []
    case_0 = 0
    case_1 = 0
    case_2 = 0
    case_3 = 0
    case_4 = 0

    if 'na_na' in args.mode:
        env.ego_car.aggressiveness = 1
        env.other_car.aggressiveness = 1
    elif 'na_a' in args.mode:
        env.ego_car.aggressiveness = 1
        env.other_car.aggressiveness = 1000
    elif 'a_na' in args.mode:
        env.ego_car.aggressiveness = 1000
        env.other_car.aggressiveness = 1
    elif 'a_a' in args.mode:
        env.ego_car.aggressiveness = 1000
        env.other_car.aggressiveness = 1000

    print(env.ego_car.aggressiveness)
    print(env.other_car.aggressiveness)

    states_df = pd.read_pickle('./random.pkl')
    states_test = states_df['states']

    p1_episode_reward = 0
    p2_episode_reward = 0
    episode_length = 0
    episode_number = 0

    state = states_test[episode_number]
    state = env.reset_state(state)
    print(state)
    start_state = state
    frame_idx = 0
    p1_match, p2_match = 0, 0

    # args.policy_type = 1
    with tqdm(total=len(states_test)) as pbar:
        while episode_number != len(states_test):
            frame_idx += 1
            s1 = [state[0], state[1], state[2], state[3]]
            s2 = [state[2], state[3], state[0], state[1]]
            if args.render:
                env.render()
            # if not is_same_player:
            #     if is_best_actions:
            #         p1_action = p1_current_model.best_action(torch.FloatTensor(s1).to(args.device))
            #         p2_action = p2_current_model.best_action(torch.FloatTensor(s2).to(args.device))
            #     else:
            #         # Agents follow average strategy
            #         # p1_action = p1_policy.act(torch.FloatTensor(s1).to(args.device), 1)
            #         # p2_action = p2_policy.act(torch.FloatTensor(s2).to(args.device), 1)
            #         # Agents follow average strategy
            #         p1_action = p1_policy.act(torch.FloatTensor(s1).to(args.device), args.policy_type)
            #         p2_action = p2_policy.act(torch.FloatTensor(s2).to(args.device), args.policy_type)
            # else:
            #     if player_id == 1:
            #         if is_best_actions:
            #             p1_action = p1_current_model.best_action(torch.FloatTensor(s1).to(args.device))
            #             p2_action = p1_current_model.best_action(torch.FloatTensor(s2).to(args.device))
            #         else:
            #             # Agents follow average strategy
            #             p1_action = p1_policy.act(torch.FloatTensor(s1).to(args.device), 1)
            #             p2_action = p1_policy.act(torch.FloatTensor(s2).to(args.device), 1)
            #     else:
            #         if is_best_actions:
            #             p1_action = p2_current_model.best_action(torch.FloatTensor(s1).to(args.device))
            #             p2_action = p2_current_model.best_action(torch.FloatTensor(s2).to(args.device))
            #         else:
            #             # Agents follow average strategy
            #             p1_action = p2_policy.act(torch.FloatTensor(s1).to(args.device), 1)
            #             p2_action = p2_policy.act(torch.FloatTensor(s2).to(args.device), 1)

            p1_b_action = p1_current_model.best_action(torch.FloatTensor(s1).to(args.device))
            p2_b_action = p2_current_model.best_action(torch.FloatTensor(s2).to(args.device))

            # Agents follow average strategy
            p1_action = p1_policy.act(torch.FloatTensor(s1).to(args.device), args.policy_type)
            p2_action = p2_policy.act(torch.FloatTensor(s2).to(args.device), args.policy_type)

            # checks if SL and RL network actions match
            if int(p1_b_action) == int(p1_action):
                p1_match += 1
            if int(p2_b_action) == int(p2_action):
                p2_match += 1

            actions = {"1": p1_action, "2": p2_action}

            next_state, reward, done = env.step(actions)
            if is_recording and episode_number % recording_freq == 0:
                env.render()
                pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))

            state = next_state

            p1_episode_reward += reward[0]
            p2_episode_reward += reward[1]
            episode_length += 1

            if done:
                if env.isCollision:
                    case_0 += 1
                elif not env.ego_car.isReached and env.other_car.isReached:
                    case_1 += 1
                elif not env.other_car.isReached and env.ego_car.isReached:
                    case_2 += 1
                elif not env.other_car.isReached and not env.ego_car.isReached:
                    case_3 += 1
                else:
                    case_4 += 1

                if is_recording and episode_number % recording_freq == 0:
                    env.render()
                    pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))
                    tag_id = -1
                    # if p1_episode_reward <= -1000 or p2_episode_reward <= -1000:
                    if env.isCollision:
                        tag_id = 0
                    elif not env.ego_car.isReached and env.other_car.isReached:
                        tag_id = 1
                    elif not env.other_car.isReached and env.ego_car.isReached:
                        tag_id = 2
                    elif not env.other_car.isReached and not env.ego_car.isReached:
                        tag_id = 3
                    else:
                        tag_id = 4

                    create_movie(tag_id, start_state, episode_number, episode_length, [p1_episode_reward, p2_episode_reward])

                p1_reward_list.append(p1_episode_reward)
                p2_reward_list.append(p2_episode_reward)
                length_list.append(episode_length)
                p1_episode_reward = 0
                p2_episode_reward = 0
                episode_length = 0
                episode_number += 1
                # state = env.reset()

                if episode_number < len(states_test):
                    state = states_test[episode_number]
                state = env.reset_state(state)
                start_state = state
                pbar.update(1)


    # delete extra frames
    if is_recording:
        [os.remove(recording_path + file) for file in os.listdir(recording_path) if ".png" in file]
    # print('0: {}, stop: {}'.format(ctr, ctr1))
    print("Test Result - Length {:.4f} p1/Reward {:.4f} p2/Reward {:.4f}, frames: {}, Episodes: {}".format(
        np.mean(length_list), np.mean(p1_reward_list), np.mean(p2_reward_list), frame_idx, episode_number))
    print(episode_number)
    print('0: {}, 1: {}, 2:{}, 3: {}, 4: {}, tot: {}'.format(case_0, case_1, case_2, case_3, case_4,
                                                             (case_0 + case_1 + case_2 + case_3 + case_4)))
    print('p1_match: {}, p2_match: {}'.format(p1_match/frame_idx, p2_match/frame_idx))

def create_heat_map(other_state, grid, path, title, ego_position_n, ego_speed_n, ego_position_grid, ego_speed_grid):
    fig, ax = plt.subplots()
    sb.heatmap(grid, linewidth=0.5)
    position_range = [C.Intersection.CAR_2.INITIAL_STATE[0] * 0.5,
                      C.Intersection.CAR_2.INITIAL_STATE[0] * 1.0]
    ego_speed_range = [0, C.Intersection.VEHICLE_MAX_SPEED]
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


def create_movie(flag, state, episode_count, episode_step_count, reward):
    img_list = [recording_path + "img" + str(i).zfill(3) + ".png" for i in
                range(episode_step_count)]
    import imageio
    images = []
    for filename in img_list:
        images.append(imageio.imread(filename))
    tag = str(flag) + '_' + str(episode_count) + '_' + str(state) + '_' + str(reward)

    imageio.mimsave(recording_path + 'movie_' + tag + '.gif', images, 'GIF', duration=time_interval)

    # Delete images
    [os.remove(recording_path + file) for file in os.listdir(recording_path) if ".png" in file]
    # print("Simulation video output saved to %s." % recording_path)


if __name__ == "__main__":
    if not args.render:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = IntersectionEnv(control_style_ego, control_style_other,
                          time_interval, MAX_TIME / time_interval)

    # hack to increase max_steps per episode
    # env.max_time_steps = 100

    p1_current_model = DQN(args).to(args.device)
    p2_current_model = DQN(args).to(args.device)
    p1_policy = Policy().to(args.device)
    p2_policy = Policy().to(args.device)
    p1_current_model.eval(), p2_current_model.eval()
    p1_policy.eval(), p2_policy.eval()

    # load_model(models={"p1": p1_current_model, "p2": p2_current_model},
    #            policies={"p1": p1_policy, "p2": p2_policy}, args=args)

    load_model_two(args, p1_current_model, p1_policy, p2_current_model, p2_policy)
    env.args = args
    test(env, args)
