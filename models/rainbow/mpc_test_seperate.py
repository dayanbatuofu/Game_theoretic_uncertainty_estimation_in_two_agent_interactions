import itertools
import logging
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
save_logs = True

def test(env, args, mode):
    pickle_file_name = './start_slice_new/mpc_' + str(args.T) + '_beta_' + (args.mode) + \
                       '_mode_' + mode[3] + '_' + str(script_ts) + '.pkl'
    logger.info(pickle_file_name)
    # logger.info('mode: {} '.format(mode), end='')
    if is_recording:
        env.is_recording = True
        env.args.render = True
        recording_path = 'test_recordings/psuedo_grace_' + args.mode + '/' + mode[3] + '/'
        env.recording_path = recording_path
        # logger.info(recording_path)
        if not os.path.exists(recording_path):
            os.makedirs(recording_path)

    # args.mode = '1'
    logger.info('beta: {}'.format(float(args.mode)))
    env.ego_car.aggressiveness = 1
    env.ego_car.gracefulness = float(args.mode)
    env.other_car.aggressiveness = mode[1]

    all_mpc_trajectories = []
    episode_mpc_trajectory = []
    # states_test_df = pd.read_pickle('./start_new/mpc_inf_a1_ne.pkl')
    # states_test = list(states_test_df['bad_states'])
    # pickle_file_name = './start_new/'
    # states_test = states_test[:1]

    reward_list, reward_list2 = [], []
    episode_reward, episode_length, episode_count, tr2, frame_idx = 0, 0, 0, 0, 0
    case_0, case_1, case_2, case_3, case_4 = 0, 0, 0, 0, 0
    ego_first = []

    ego_position_n = 100.  # number of grid points - 1
    ego_speed_n = 100.
    ego_position_range = [25.0, 45.0]
    ego_position_grid = np.arange(ego_position_range[0], ego_position_range[1],
                                  (ego_position_range[1] - ego_position_range[0]) / ego_position_n)
    # logger.info(ego_position_grid)
    ego_speed_range = [5.0, 15.0]
    ego_speed_grid = np.arange(ego_speed_range[0], ego_speed_range[1],
                               (ego_speed_range[1] - ego_speed_range[0]) / ego_speed_n)

    states_test1 = []
    for position in ego_position_grid:
        for speed in ego_speed_grid:
            states_test1.append([position - 0.1, speed, position, speed, 0.0])
    #
    # states_test1 = []
    # pos = 30.5
    # while len(states_test1) < 2000:
    #     # states_test1.append([14.94843262,  5.14828587, pos,  8.246211251235321, 2.0])
    #     states_test1.append([pos, 10.2, pos, 10.2, 2.0])
        # pos = pos - 0.005
    #
    states_test = states_test1
    # states_test_df = pd.read_pickle('./random_slice_5.pkl')
    # states_test = list(states_test_df['start_states'])
    # logger.info(len(states_test))

    # states_test = states_test1[:1]
    logger.info('Test_states: {}'.format(len(states_test)))
    state = states_test[episode_count]
    logger.info(state)
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

    log_df = pd.DataFrame(columns=['start_state', 'p1_reward', 'p2_reward'])


    # logger.info('choice: {}, aggr:{}'.format(choice, env.other_car.aggressiveness))
    start_time = time.time()
    episode_step = 0
    while episode_count != total_eps:
        episode_step += 1
        frame_idx += 1
        if args.render:
            # logger.info('render...')
            env.render()
        # logger.info(episode_count)
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

        action2 = mode[2].act(torch.FloatTensor(p2_state).to(args.device))
        max_q_index = (get_models()[0])[0].act(torch.FloatTensor(p2_state).to(args.device))

        # if one car crosses no notion of being graceful
        if (state[0] + C.CAR_LENGTH * 0.5 + 1. <= 0.) or (state[2] + C.CAR_LENGTH * 0.5 + 1. <= 0):
            # logger.info('crossed an intersection!!')
            action1 = action1_inf
        else:
            # if episode_step > 8:
            action1 = env.mpc_psuedo_batch(args, state, q_set, pos_trajectories, T=args.T)
            # else:
            #     action1 = None
            # logger.info(action1)
            if action1 is None:
                action1 = action1_inf

        actions = {"1": action1, "2": action2}
        episode_mpc_trajectory.append(action1)
        # logger.info(actions)
        # logger.info('psuedo-act:{}, action2:{}, act1-eq:{}\n'.format(action1, action2, action1_eq))

        # if is_recording and episode_count % recording_freq == 0:
        #     env.render()
        #     pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))
        # if int(action2) < 2 and (state[2] + C.CAR_LENGTH * 0.5 + 1. > 0.):
        #     logger.info(actions, max_q_index, env.frame)
        # logger.info(action2, end=' ')
        next_state, reward, done = env.step_inference(actions, q_set)
        # logger.info(state, actions)

        state = next_state
        episode_reward += reward[0]
        tr2 += reward[1]
        episode_length += 1

        if done:
            # logger.info([episode_count, episode_reward, tr2])
            if episode_count % 100 == 0:
                logger.info(episode_count)
            ego_first.append(env.ego_crossed_first)
            if env.isCollision:
                logger.info(start_state)
                logger.info([episode_count, episode_reward, tr2, mode[3], env.other_car.aggressiveness])
                logger.info('--------------------')
                case_0 += 1
            elif not env.ego_car.isReached and env.other_car.isReached:
                case_1 += 1
            elif not env.other_car.isReached and env.ego_car.isReached:
                logger.info(start_state)
                logger.info([episode_count, episode_reward, tr2, mode[3], env.other_car.aggressiveness])
                logger.info('--------------------')
                case_2 += 1
            elif not env.other_car.isReached and not env.ego_car.isReached:
                case_3 += 1
            else:
                case_4 += 1

            if is_recording and episode_count % recording_freq == 0:
                # env.render()
                # pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))

                # if tr2 < 0 or episode_reward < 0:
                if env.isCollision:
                    logger.info('collision for: {}'.format(start_state))
                    create_movie(0, start_state, recording_path, episode_count, env.total_frames,
                                 [episode_reward, tr2], env.other_car.aggressiveness)
                elif not env.ego_car.isReached and env.other_car.isReached:
                    create_movie(1, start_state, recording_path, episode_count, env.total_frames,
                                 [episode_reward, tr2],
                                 env.other_car.aggressiveness)
                elif not env.other_car.isReached and env.ego_car.isReached:
                    create_movie(2, start_state, recording_path, episode_count, env.total_frames,
                                 [episode_reward, tr2],
                                 env.other_car.aggressiveness)
                elif not env.other_car.isReached and not env.ego_car.isReached:
                    create_movie(3, start_state, recording_path, episode_count, env.total_frames,
                                 [episode_reward, tr2],
                                 env.other_car.aggressiveness)
                else:
                    create_movie(4, start_state, recording_path, episode_count, env.total_frames,
                                 [episode_reward, tr2],
                                 env.other_car.aggressiveness)
            if is_recording:
                [os.remove(recording_path + file) for file in os.listdir(recording_path) if ".png" in file]
            reward_list.append(episode_reward)
            reward_list2.append(tr2)
            # add to the log
            log_df = log_df.append({'start_state': start_state, 'p1_reward': episode_reward, 'p2_reward': tr2}, ignore_index=True)   #  [[start_state, episode_reward, tr2]])
            # logger.info(episode_count)
            log_df.to_csv('mpc_' + str(args.T) + '_beta_' + (args.mode) + '_mode_' + mode[3] + '_' + str(script_ts) + '.csv')
            log_df.to_pickle('mpc_' + str(args.T) + '_beta_' + (args.mode) + '_mode_' + mode[3] + '_' + str(script_ts) + '.pkl')

            episode_reward, episode_length, tr2 = 0, 0, 0
            env.total_frames = 0
            episode_count += 1
            episode_step = 0

            if episode_count < total_eps:
                state = states_test[episode_count]
            start_state = state
            # logger.info('here: {}'.format(state))
            state = env.reset_inference_state(state)
            all_mpc_trajectories.append(episode_mpc_trajectory)
            episode_mpc_trajectory = []
            # logger.info('choice: {}, aggr:{}'.format(choice, env.other_car.aggressiveness))
            # sys.exit()

            # logger.info('--------------------------')
    end_time = time.time()
    logger.info('Total test_time: {}'.format(end_time-start_time))
    logger.info('#frames: {}'.format(frame_idx))
    if is_recording:
        # Delete images
        [os.remove(recording_path + file) for file in os.listdir(recording_path) if ".png" in file]
    logger.info('Avg rewards: P1: {}, {}'.format(round(np.mean(reward_list), 2), round(np.mean(reward_list2), 2)))
    logger.info('0: {}, 1: {}, 2:{}, 3: {}, 4: {}, tot: {}'.format(case_0, case_1, case_2, case_3, case_4,
                                                             (case_0 + case_1 + case_2 + case_3 + case_4)))

    # logger.info(all_mpc_trajectories)
    # save all mpc trajectories
    if save_logs:
        # fname = 'mpc_traj_T_' + str(args.T) + '_beta_' + str(args.mode) + '_' + mode[3] + '_' + str(script_ts) \
        #         + '.csv'
        # import csv
        # with open(fname, "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(all_mpc_trajectories)

        df = {'start_states': states_test, 'p1_reward': reward_list, 'p2_reward': reward_list2, 'ego_first': ego_first}
        pandas_df = pd.DataFrame(df)
        pandas_df.to_pickle(pickle_file_name)



if __name__ == '__main__':
    args = get_args()
    action_size = 5

    if not args.render:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    script_ts = int(time.time())
    env = IntersectionEnv(control_style_ego, control_style_other,
                          time_interval, MAX_TIME / time_interval)
    env.args = args
    logger = logging.getLogger('mpc_policy')
    hdlr = logging.FileHandler('mpc_policy_' + str(script_ts) + '.log')
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    formatter = logging.Formatter('%(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()[1]

    modes = {'a1': [1, 1, policy_na_na, 'a1'], 'a2': [1, 1, policy_na_na_2, 'a2'],
             'b1': [1, 1e3, policy_a_na, 'b1']}

    # modes = {'b1': [1, 1e3, policy_a_na, 'b1']}

    for mode in modes.values():
        env = IntersectionEnv(control_style_ego, control_style_other,
                              time_interval, MAX_TIME / time_interval)
        env.args = args
        test(env, args, mode)