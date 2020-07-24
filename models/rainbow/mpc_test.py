import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import pygame as pg
import torch

from parameters import control_style_ego, control_style_other, time_interval, MAX_TIME
from arguments import get_args
from common.utils import create_movie
from intersection_env_test import IntersectionEnv
from set_nfsp_models import get_models
from constants import CONSTANTS as C

is_recording = False
recording_freq = 1
# mode = 0
total_frames = 50000
is_online_policy = False
total_eps = 10000


def test(env, args, mode):

    pickle_file_name = './mpc_' + str(args.T) + '_beta_' + (args.mode) + '_mode_' + str(mode) + '_0.01.pkl'
    states_test_df = pd.read_pickle('./time_0.01.pkl')
    states_test = list(states_test_df['states'])
    print(pickle_file_name)
    print('mode: {} '.format(mode), end='')
    if is_recording:
        recording_path = 'test_recordings/psuedo_grace_' + args.mode + '/' + str(mode) + '/'
        # print(recording_path)
        if not os.path.exists(recording_path):
            os.makedirs(recording_path)

    # args.mode = '1'
    print('beta: {}'.format(float(args.mode)))
    env.ego_car.aggressiveness = 1
    env.ego_car.gracefulness = float(args.mode)
    choice = -1
    if mode == 0:
        env.other_car.aggressiveness = np.random.choice([1000, 1], 1, p=[0.5, 0.5])[0]
        if env.other_car.aggressiveness == 1:
            choice = np.random.choice([1, 2], 1, p=[0.5, 0.5])[0]
    else:
        env.other_car.aggressiveness = 1000

    all_mpc_trajectories = []
    episode_mpc_trajectory = []


    reward_list, reward_list2 = [], []
    episode_reward, episode_length, episode_count, tr2, frame_idx = 0, 0, 0, 0, 0
    case_0, case_1, case_2, case_3, case_4 = 0, 0, 0, 0, 0
    ego_first = []

    state = states_test[episode_count]
    print(state)
    start_state = state
    env.reset_inference_state(state)

    q_set = get_models()[0]
    # if args.requires_grad:
    #     pos_actions = [torch.tensor(i, dtype=torch.float, device=args.device, requires_grad=True) for i in range(5)]
    # else:
    #     # pos_actions = [torch.tensor(i, dtype=torch.float, device=args.device) for i in range(5)]
    pos_actions = [i for i in range(5)]
    all_list = []
    for t in range(args.T):
        all_list.append(pos_actions)
    # using itertools.product()
    # to compute all possible permutations
    pos_trajectories = list(itertools.product(*all_list))

    # print('choice: {}, aggr:{}'.format(choice, env.other_car.aggressiveness))
    start_time = time.time()
    while episode_count != total_eps:

        frame_idx += 1
        # env.render()
        # print(episode_count)
        p2_state = [state[2], state[3], state[0], state[1]]
        inf_idx = int(state[4])
        p2_theta = env.theta_set[inf_idx][0]
        t1 = (env.ego_car.aggressiveness, p2_theta) # (theta_i, theta_j_hat)
        t1_idx = -1
        if t1[0] == t1[1] and t1[1] == 1 and inf_idx == 0:
            t1_idx = 1
        elif t1[0] == t1[1] and t1[1] == 1 and inf_idx == 1:
            t1_idx = 0
        elif t1[0] == t1[1] and t1[1] == 1000 and inf_idx == 4:
            t1_idx = 5
        elif t1[0] == t1[1] and t1[1] == 1000 and inf_idx == 5:
            t1_idx = 4
        else:
            t1_idx = env.theta_set.index(t1)
        # id-based on ego policy
        policy_idx = t1_idx

        action2, action1_inf = None, None
        if policy_idx == 0:
            action1_inf = policy_na_na.act(torch.FloatTensor(state[:-1]).to(args.device))
        elif policy_idx == 1:
            action1_inf = policy_na_na_2.act(torch.FloatTensor(state[:-1]).to(args.device))
        elif policy_idx == 2:
            action1_inf = policy_na_a.act(torch.FloatTensor(state[:-1]).to(args.device))

        if mode == 0:
            if env.other_car.aggressiveness == 1:
                if choice == 1:
                    action2 = policy_na_na.act(torch.FloatTensor(p2_state).to(args.device))
                else:
                    action2 = policy_na_na_2.act(torch.FloatTensor(p2_state).to(args.device))
            else:
                action2 = policy_a_na.act(torch.FloatTensor(p2_state).to(args.device))
        else:
            action2 = policy_a_na.act(torch.FloatTensor(p2_state).to(args.device))

        # if one car crosses no notion of being graceful
        if (state[0] + C.CAR_LENGTH * 0.5 + 1. <= 0.) or (state[2] + C.CAR_LENGTH * 0.5 + 1. <= 0):
            action1 = action1_inf
        else:
            action1 = env.mpc_psuedo_batch(args, state, q_set, pos_trajectories, T=args.T)
            # print(action1)
            if action1 is None:
                action1 = action1_inf

        actions = {"1": action1, "2": action2}
        episode_mpc_trajectory.append(action1)
        # print(actions)
        # print('psuedo-act:{}, action2:{}, act1-eq:{}\n'.format(action1, action2, action1_eq))

        if is_recording and episode_count % recording_freq == 0:
            env.render()
            pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))
        next_state, reward, done = env.step_inference(actions, q_set)

        state = next_state
        episode_reward += reward[0]
        tr2 += reward[1]
        episode_length += 1

        if done:
            ego_first.append(env.ego_crossed_first)
            # if env.ego_car.gracefulness > 0:
            #     if not env.ego_crossed_first:
            #         print(env.ego_crossed_first, episode_count)
            # else:
            #     if env.ego_crossed_first:
            #         print(env.ego_crossed_first, episode_count)
            # print([episode_count, episode_reward, tr2, choice, env.other_car.aggressiveness])
            if is_recording and episode_count % recording_freq == 0:
                env.render()
                pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))
            if env.isCollision:
                print(start_state)
                print([episode_count, episode_reward, tr2, choice, env.other_car.aggressiveness])
                print('--------------------')
                case_0 += 1
            elif not env.ego_car.isReached and env.other_car.isReached:
                case_1 += 1
            elif not env.other_car.isReached and env.ego_car.isReached:
                print(start_state)
                print([episode_count, episode_reward, tr2, choice, env.other_car.aggressiveness])
                print('--------------------')
                case_2 += 1
            elif not env.other_car.isReached and not env.ego_car.isReached:
                case_3 += 1
            else:
                case_4 += 1

            if is_recording and episode_count % recording_freq == 0:
                env.render()
                pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))

                # if tr2 < 0 or episode_reward < 0:
                if env.isCollision:
                    create_movie(0, start_state, recording_path, episode_count, episode_length + 1,
                                 [episode_reward, tr2], env.other_car.aggressiveness)
                elif not env.ego_car.isReached and env.other_car.isReached:
                    create_movie(1, start_state, recording_path, episode_count, episode_length + 1,
                                 [episode_reward, tr2],
                                 env.other_car.aggressiveness)
                elif not env.other_car.isReached and env.ego_car.isReached:
                    create_movie(2, start_state, recording_path, episode_count, episode_length + 1,
                                 [episode_reward, tr2],
                                 env.other_car.aggressiveness)
                elif not env.other_car.isReached and not env.ego_car.isReached:
                    create_movie(3, start_state, recording_path, episode_count, episode_length + 1,
                                 [episode_reward, tr2],
                                 env.other_car.aggressiveness)
                else:
                    create_movie(4, start_state, recording_path, episode_count, episode_length + 1,
                                 [episode_reward, tr2],
                                 env.other_car.aggressiveness)
            if is_recording:
                [os.remove(recording_path + file) for file in os.listdir(recording_path) if ".png" in file]
            reward_list.append(episode_reward)
            reward_list2.append(tr2)
            episode_reward, episode_length, tr2 = 0, 0, 0
            episode_count += 1

            if mode == 0:
                env.other_car.aggressiveness = np.random.choice([1000, 1], 1, p=[0.5, 0.5])[0]
                if env.other_car.aggressiveness == 1:
                    choice = np.random.choice([1, 2], 1, p=[0.5, 0.5])[0]
            else:
                env.other_car.aggressiveness = 1000

            if episode_count < total_eps:
                state = states_test[episode_count]
            start_state = state
            state = env.reset_inference_state(state)
            all_mpc_trajectories.append(episode_mpc_trajectory)
            episode_mpc_trajectory = []
            # print('choice: {}, aggr:{}'.format(choice, env.other_car.aggressiveness))
            # sys.exit()

            # print('--------------------------')
    end_time = time.time()
    print('Total test_time: {}'.format(end_time-start_time))
    print('#frames: {}'.format(frame_idx))
    if is_recording:
        # Delete images
        [os.remove(recording_path + file) for file in os.listdir(recording_path) if ".png" in file]
    print('Avg rewards: P1: {}, {}'.format(round(np.mean(reward_list), 2), round(np.mean(reward_list2), 2)))
    print('0: {}, 1: {}, 2:{}, 3: {}, 4: {}, tot: {}'.format(case_0, case_1, case_2, case_3, case_4,
                                                             (case_0 + case_1 + case_2 + case_3 + case_4)))

    # print(all_mpc_trajectories)
    # save all mpc trajectories
    fname = 'mpc_traj_T_' + str(args.T) + '_beta_' + str(env.ego_car.gracefulness) + '_' + str(mode) + '.csv'
    import csv
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_mpc_trajectories)

    df = {'start_states': states_test, 'p1_reward': reward_list, 'p2_reward': reward_list2, 'ego_first': ego_first}
    pandas_df = pd.DataFrame(df)
    pandas_df.to_pickle(pickle_file_name)



if __name__ == '__main__':
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    args = get_args()
    action_size = 5
    env = IntersectionEnv(control_style_ego, control_style_other,
                          time_interval, MAX_TIME / time_interval)
    env.args = args
    (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()[1]
    for mode in range(2):
        test(env, args, mode)