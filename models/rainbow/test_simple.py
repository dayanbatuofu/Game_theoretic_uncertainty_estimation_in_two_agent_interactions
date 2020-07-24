import logging
import os

import numpy as np
import pygame as pg
import torch

from parameters import control_style_ego, control_style_other, time_interval, MAX_TIME
from arguments import get_args
from common.utils import create_movie
from intersection_env_test import IntersectionEnv
from model import DQN
from set_nfsp_models import get_models
import pandas as pd

is_recording = False
recording_freq = 1
is_rl_enabled = False

def test(env, args, mode):
    logging.info('mode: {} \r'.format(mode[4]))
    if is_recording:
        recording_path = 'test_recordings/' + args.mode + '/' + mode[4] + '/'
        # print(recording_path)
        if not os.path.exists(recording_path):
            os.makedirs(recording_path)

    if is_rl_enabled:
        fname = args.model_dir + 'model.pth'
        current_model = DQN(env, args).to(args.device)
        current_model.eval()

        if args.device == torch.device("cpu"):
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        if not os.path.exists(fname):
            raise ValueError("No model saved with name {}".format(fname))

        current_model.load_state_dict(torch.load(fname, map_location))

    env.ego_car.aggressiveness = mode[0]
    env.other_car.aggressiveness = mode[1]

    p1_reward = 0
    episode_length = 0
    reward1_list, reward2_list= [], []
    episode_count = 0
    p2_reward = 0
    frame_idx = 0
    case_0, case_1, case_2, case_3, case_4 = 0, 0, 0, 0, 0
    inf = []
    inf_success = 0
    inf_failed = 0

    states_test_df = pd.read_pickle('./random.pkl')
    states_test = list(states_test_df['states'])

    state = states_test[episode_count]
    state = env.reset_state(state)
    start_state = state

    while episode_count != len(states_test):
        frame_idx += 1
        p2_state = [state[2], state[3], state[0], state[1]]

        if not is_rl_enabled:
            action1 = mode[2].act(torch.FloatTensor(state).to(args.device))
        else:
            action1 = current_model.best_act(torch.FloatTensor(state).to(args.device))

        action2 = mode[3].act(torch.FloatTensor(p2_state).to(args.device))

        actions = {"1": action1, "2": action2}

        if is_recording and episode_count % recording_freq == 0:
            env.render()
            pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))
        next_state, reward, done = env.step(actions)

        reward1 = reward[0]
        state = next_state
        p1_reward += reward1
        p2_reward += reward[1]
        episode_length += 1

        if done:
            if is_recording and episode_count % recording_freq == 0:
                env.render()
                pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))

            # logging.debug('EP: {}, inf: {}, r:{}'.format(episode_count, inf, [p1_reward, p2_reward]))
            if env.isCollision:
                case_0 += 1
                # logging.debug('case-0, EP: {}, start_state:{}, inf: {}, r:{}'.format(episode_count, start_state, inf, [p1_reward, p2_reward]))
            elif not env.ego_car.isReached and env.other_car.isReached:
                case_1 += 1
                # logging.debug('case-1, EP: {}, start_state:{}, inf: {}, r:{}'.format(episode_count, start_state, inf, [p1_reward, p2_reward]))
            elif not env.other_car.isReached and env.ego_car.isReached:
                case_2 += 1
                # logging.debug('case-2, EP: {}, start_state:{}, inf: {}, r:{}'.format(episode_count, start_state, inf, [p1_reward, p2_reward]))
            elif not env.other_car.isReached and not env.ego_car.isReached:
                case_3 += 1
                # logging.debug('case-3, EP: {}, start_state:{}, inf: {}, r:{}'.format(episode_count, start_state, inf, [p1_reward, p2_reward]))
            else:
                case_4 += 1
                # logging.debug('case-4, EP: {}, start_state:{}, inf: {}, r:{}'.format(episode_count, start_state, inf, [p1_reward, p2_reward]))

            if is_recording and episode_count % recording_freq == 0:
                env.render()
                pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))
                # if p1_reward < 0 or p2_reward < 0:
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

            # print("episode reward: p1: {}".format(p1_reward))
            # print(len(states_test))
            reward1_list.append(p1_reward)
            reward2_list.append(p2_reward)
            p1_reward = 0
            episode_length = 0
            p2_reward = 0
            env.inference_actions = []
            env.inference_states = []
            inf = []
            episode_count += 1

            # state = env.reset()
            if episode_count < len(states_test):
                state = states_test[episode_count]
                state = env.reset_state(state)
                start_state = state

            # print('--------------------------')
    logging.info('#frames: {}'.format(frame_idx))
    if is_recording:
        # Delete images
        [os.remove(recording_path + file) for file in os.listdir(recording_path) if ".png" in file]

    logging.info('Avg rewards: P1: {}, {}'.format(round(np.mean(reward1_list), 2), round(np.mean(reward2_list), 2)))
    logging.info('0: {}, 1: {}, 2:{}, 3: {}, 4: {}, tot: {}'.format(case_0, case_1, case_2, case_3, case_4,
                                                             (case_0 + case_1 + case_2 + case_3 + case_4)))
    # save the metrics to file


if __name__ == '__main__':
    args = get_args()
    action_size = 5
    if not args.render:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()[1]


    log_level = logging.DEBUG #logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    modes = {'a1': [1, 1, policy_na_na, policy_na_na, 'a1'], 'a2': [1, 1, policy_na_na, policy_na_na_2, 'a2'],
             'a3': [1, 1, policy_na_na_2, policy_na_na, 'a3'], 'a4': [1, 1, policy_na_na_2, policy_na_na_2, 'a4'],
             'b1': [1, 1e3, policy_na_na, policy_a_na, 'b1'], 'b2': [1, 1e3, policy_na_na_2, policy_a_na, 'b2'],
             'c1': [1, 1, policy_na_a, policy_na_na, 'c1'], 'c2': [1, 1, policy_na_a, policy_na_na_2, 'c2'],
             'd1': [1, 1e3, policy_na_a, policy_a_na, 'd1']}

    # modes = {'c1': [1, 1, policy_na_a, policy_na_na, 'c1']}

    env = IntersectionEnv(control_style_ego, control_style_other,
                          time_interval, MAX_TIME / time_interval)
    env.args = args
    states_test = []

    for mode in modes.values():
        # print(mode)
        # print(type(mode))
        test(env, args, mode)
