import itertools
import os
import sys

import numpy as np
import pandas as pd
import pygame as pg
import torch
import random

from parameters import control_style_ego, control_style_other, time_interval, MAX_TIME
from arguments import get_args
from common.utils import create_movie
from intersection_env_test import IntersectionEnv
from model import DQN
from set_nfsp_models import get_models
from constants import CONSTANTS as C

is_recording = True
recording_freq = 1
save_logs = False

def get_mpc_action(env, state, q_set, pos_trajectories):
   #  print(state)
    inf_idx = int(state[4])
    p2_theta = env.theta_set[inf_idx][0]
    t1 = (env.ego_car.aggressiveness, p2_theta)  # (theta_i, theta_j_hat)
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

    if (state[0] + C.CAR_LENGTH * 0.5 + 1. <= 0.) or (state[2] + C.CAR_LENGTH * 0.5 + 1. <= 0):
        action1 = action1_inf
    else:
        action1 = env.mpc_psuedo_batch(args, state, q_set, pos_trajectories, T=args.T)
        # print(action1)
        if action1 is None:
            action1 = action1_inf
    return action1

def get_online_action(state):
    inf_mode = state[4]  # (theta_j, theta_i)
    if inf_mode == 0:  # na, na
        action1 = policy_na_na_2.act(torch.FloatTensor(state[:-1]).to(args.device))
    elif inf_mode == 1:  # na, na
        action1 = policy_na_na.act(torch.FloatTensor(state[:-1]).to(args.device))
    elif inf_mode == 2:  # na, a
        # choosing conservative (na, na, 2)
        action1 = policy_na_na_2.act(torch.FloatTensor(state[:-1]).to(args.device))
    elif inf_mode == 3:  # a, na
        action1 = policy_na_a.act(torch.FloatTensor(state[:-1]).to(args.device))
    elif inf_mode == 4:  # a, a
        action1 = policy_na_a.act(torch.FloatTensor(state[:-1]).to(args.device))
    elif inf_mode == 5:  # a, a
        action1 = policy_na_a.act(torch.FloatTensor(state[:-1]).to(args.device))
    return action1

def test(env, args, mode):
    print('mode: {}'.format(mode[3]))
    dir = './start_slice/'

    recording_path = 'test_recordings/' + args.mode + '/' + mode[4] + '/'
    env.recording_path = recording_path
    env.args.render = True
    env.is_recording = is_recording

    if not os.path.exists(recording_path):
        os.makedirs(recording_path)

    states_file = dir + mode[4] + '.pkl'
    p1_reward = 0
    episode_length = 0
    reward_list, reward_list2, reward_list3, reward_list4 = [], [], [], []
    episode_count = 0
    p2_reward, p1_reward_self, p2_reward_self = 0, 0, 0
    frame_idx = 0
    case_0, case_1, case_2, case_3, case_4 = 0, 0, 0, 0, 0

    env.ego_car.aggressiveness = mode[0]
    env.other_car.aggressiveness = mode[1]

    states_test_df = pd.read_pickle(states_file)
    states_test = list(states_test_df['bad_states'])
    p1_r = list(states_test_df['r1'])
    p2_r = list(states_test_df['r2'])
    print('file: {}, found: {}'.format(states_file, len(states_test)))

    # # filtered_states = random.sample(states_test, 10)
    # # print(filtered_states)
    # # sys.exit()
    # # print(len(states_test))
    # filtered_states = []
    # for i in range(len(states_test)):
    #     if p1_r[i] < -15.0:
    #         filtered_states.append(states_test[i])
    #         # print(states_test[i], p1_r[i], p2_r[i])
    #
    # states_test = filtered_states[:10]
    # print(len(filtered_states))

    # sys.exit()
    is_online = False
    is_rl_enabled = False
    is_mpc_enabled = False

    q_set = get_models()[0]
    pos_actions = [i for i in range(5)]
    all_list = []
    for t in range(args.T):
        all_list.append(pos_actions)
    pos_trajectories = list(itertools.product(*all_list))

    if len(states_test) > 0:

        if args.mode == 'online':
            is_online = True
        elif args.mode == 'mpc':
            env.ego_car.gracefulness = 1
            is_mpc_enabled = True
        else:
            is_rl_enabled = True

        if is_rl_enabled:
            fname = args.log_dir + args.mode + '/model_' + str(args.load_model_index) + '.pth'
            if 'inf' in args.mode:
                env.ego_car.gracefulness = 0
            elif 'g1' in args.mode:
                env.ego_car.gracefulness = 1
            elif 'g2' in args.mode:
                env.ego_car.gracefulness = 1000

            print(env.ego_car.gracefulness)
            current_model = DQN(env, args).to(args.device)
            current_model.eval()
            if args.device == torch.device("cpu"):
                map_location = lambda storage, loc: storage
            else:
                map_location = None

            if not os.path.exists(fname):
                raise ValueError("No model saved with name {}".format(fname))

            current_model.load_state_dict(torch.load(fname, map_location))

        state = states_test[episode_count]
        state = env.reset_inference_state(state)
        start_state = state

        while episode_count != len(states_test):
            frame_idx += 1
            p2_state = [state[2], state[3], state[0], state[1]]
            action1 = -1.0 # should give error
            if is_rl_enabled:
                action1 = current_model.best_act(torch.FloatTensor(state[:-1]).to(args.device))
            elif is_online:
                action1 = get_online_action(state)
            elif is_mpc_enabled:
                action1 = get_mpc_action(env, state, q_set, pos_trajectories)

            action2 = mode[2].act(torch.FloatTensor(p2_state).to(args.device))

            # env.render()
            # pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))

            actions = {"1": action1, "2": action2}

            # next_state, reward, done = env.step(actions)
            # print(next_state, reward, done)
            next_state, reward, done = env.step_inference(actions, q_set)
            # print(next_state, reward, done)
            # print('------')
            state = next_state
            p1_reward += reward[0]
            p2_reward += reward[1]
            p1_reward_self += reward[2]
            p2_reward_self += reward[3]
            episode_length += 1

            if done:

                # env.render()
                # pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))

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

                if is_recording and episode_count % recording_freq == 0:
                    # env.render()
                    # pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))
                    # if p2_reward < 0 or p1_reward < 0:
                    tag_id = -1
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

                    create_movie(tag_id, start_state, recording_path, episode_count, env.total_frames,
                                 [p1_reward, p2_reward], env.other_car.aggressiveness)

                reward_list.append(p1_reward)
                reward_list2.append(p2_reward)
                reward_list3.append(p1_reward_self)
                reward_list4.append(p2_reward_self)
                p1_reward, episode_length = 0, 0
                p2_reward, p1_reward_self, p2_reward_self = 0, 0, 0
                episode_count += 1

                if episode_count < len(states_test):
                    state = states_test[episode_count]
                    state = env.reset_inference_state(state)
                    start_state = state
                else:
                    break

        print('#frames: {}'.format(frame_idx))
        # remove extra frames
        [os.remove(recording_path + file) for file in os.listdir(recording_path) if ".png" in file]
        print('Avg rewards: P1: {}, {}'.format(round(np.mean(reward_list), 2), round(np.mean(reward_list2), 2)))
        print('0: {}, 1: {}, 2:{}, 3: {}, 4: {}, tot: {}'.format(case_0, case_1, case_2, case_3, case_4,
                                                                 (case_0 + case_1 + case_2 + case_3 + case_4)))


if __name__ == '__main__':
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    args = get_args()
    action_size = 5
    env = IntersectionEnv(control_style_ego, control_style_other,
                          time_interval, MAX_TIME / time_interval)
    env.args = args

    (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()[1]

    # 'scenario' : [ego_aggressiveness, human_aggressiveness, human_policy, 'scenario_tag', 'states_file']
    modes = {'a1': [1, 1, policy_na_na, 'a1', 'inf_mpc_a1_ne'] ,
             'a2': [1, 1, policy_na_na_2, 'a2', 'inf_mpc_a2_ne'],
             'b1': [1, 1e3, policy_a_na, 'b1', 'inf_mpc_b1_ne']}

    for mode in modes.values():
        test(env, args, mode)
