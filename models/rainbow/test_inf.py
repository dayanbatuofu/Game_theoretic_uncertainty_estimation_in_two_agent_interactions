import logging
import os

import numpy as np
import pandas as pd
import pygame as pg
import torch

from parameters import control_style_ego, control_style_other, time_interval, MAX_TIME
from arguments import get_args
from common.utils import create_movie
from intersection_env_test import IntersectionEnv
from model import DQN
from set_nfsp_models import get_models

is_recording = False
recording_freq = 1
save_logs = True
is_empathetic = False

def test(env, args, q_set, mode):
    logging.info('mode: {} \r'.format(mode[3]))

    pickle_file_name = './'

    if is_recording:
        recording_path = 'test_recordings/' + args.mode + '/' + mode[3] + '/'
        # print(recording_path)

        if not os.path.exists(recording_path):
            os.makedirs(recording_path)

    is_rl_enabled = True
    if args.mode == 'n_inf':
        is_rl_enabled = False
        pickle_file_name += 'online'

    if is_rl_enabled:
        # fname = args.log_dir + args.mode + '/' + args.mode + '_' + str(args.load_model_index) + '.pth'
        fname = args.log_dir + args.mode + '/model_' + str(args.load_model_index) + '.pth'
        # fname = args.log_dir + args.mode + '/model.pth'
        # print(fname)
        if 'inf' in args.mode:
            pickle_file_name += 'neutral'
            env.ego_car.gracefulness = 0
        elif 'g1' in args.mode:
            pickle_file_name += 'g1'
            env.ego_car.gracefulness = 1
        elif 'g2' in args.mode:
            pickle_file_name += 'g2'
            env.ego_car.gracefulness = 1000
        # print(fname)
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

    env.ego_car.aggressiveness = mode[0]
    env.other_car.aggressiveness = mode[1]

    if save_logs:
        pickle_file_name += '_' + mode[3] + '.pkl'
        print(pickle_file_name)

    states_test_df = pd.read_pickle('./time_0.01.pkl')
    states_test = list(states_test_df['states'])

    reward1_list, reward2_list, reward3_list, reward4_list = [], [], [], []
    p1_reward, episode_length, episode_count, inf_success, inf_failed = 0, 0, 0, 0, 0
    p2_reward, p1_reward_self, p2_reward_self = 0, 0, 0
    frame_idx = 0
    case_0, case_1, case_2, case_3, case_4 = 0, 0, 0, 0, 0

    inf = []
    state = states_test[episode_count]

    start_state = state

    state = env.reset_inference_state(state)

    while episode_count != len(states_test):
        frame_idx += 1
        p1_state = [state[0], state[1], state[2], state[3]]
        p2_state = [state[2], state[3], state[0], state[1]]
        action1 = None
        inf.append(state[4])

        if env.theta_set[int(state[4])][0] == mode[1]:
            inf_success += 1
        else:
            inf_failed += 1

        if not is_rl_enabled:
            # state[4] = 1
            inf_mode = state[4]  # (theta_j, theta_i)
            if inf_mode == 0:  # na, na
                action1 = policy_na_na_2.act(torch.FloatTensor(p1_state).to(args.device))
            elif inf_mode == 1:  # na, na
                action1 = policy_na_na.act(torch.FloatTensor(p1_state).to(args.device))
            elif inf_mode == 2:  # na, a
                # choosing conservative (na, na, 2)
                if not is_empathetic:
                    action1 = policy_na_na_2.act(torch.FloatTensor(p1_state).to(args.device))
                else:
                    action1 = policy_a_na.act(torch.FloatTensor(p1_state).to(args.device))
            elif inf_mode == 3:  # a, na
                action1 = policy_na_a.act(torch.FloatTensor(p1_state).to(args.device))
            elif inf_mode == 4:  # a, a
                if not is_empathetic:
                    action1 = policy_na_a.act(torch.FloatTensor(p1_state).to(args.device))
                else:
                    action1 = policy_a_a_2.act(torch.FloatTensor(p1_state).to(args.device))
            elif inf_mode == 5:  # a, a
                if not is_empathetic:
                    action1 = policy_na_a.act(torch.FloatTensor(p1_state).to(args.device))
                else:
                    action1 = policy_a_a.act(torch.FloatTensor(p1_state).to(args.device))
        else:
            action1 = current_model.best_act(torch.FloatTensor(state).to(args.device))

        action2 = mode[2].act(torch.FloatTensor(p2_state).to(args.device))
        # env.render()
        actions = {"1": action1, "2": action2}
        if is_recording and episode_count % recording_freq == 0:
            env.render()
            pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))
        next_state, reward, done = env.step_inference(actions, q_set)

        reward1 = reward[0]

        state = next_state
        p1_reward += reward1
        p2_reward += reward[1]
        p1_reward_self += reward[2]
        p2_reward_self += reward[3]
        episode_length += 1

        if done:
            if is_recording and episode_count % recording_freq == 0:
                env.render()
                pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))

            # logging.debug('EP: {}, inf: {}, r:{}'.format(episode_count, inf, [p1_reward, p2_reward]))

            if env.isCollision:
                case_0 += 1
                # logging.debug('case-0, EP: {}, inf: {}, r:{}'.format(episode_count, inf, [p1_reward, p2_reward]))
            elif not env.ego_car.isReached and env.other_car.isReached:
                case_1 += 1
                # logging.debug('case-1, EP: {}, inf: {}, r:{}'.format(episode_count, inf, [p1_reward, p2_reward]))
            elif not env.other_car.isReached and env.ego_car.isReached:
                case_2 += 1
                # logging.debug('case-2, EP: {}, inf: {}, r:{}'.format(episode_count, inf, [p1_reward, p2_reward]))
            elif not env.other_car.isReached and not env.ego_car.isReached:
                case_3 += 1
                # logging.debug('case-3, EP: {}, inf: {}, r:{}'.format(episode_count, inf, [p1_reward, p2_reward]))
            else:
                case_4 += 1
                # logging.debug('case-4, EP: {}, inf: {}, r:{}'.format(episode_count, inf, [p1_reward, p2_reward]))

            if is_recording and episode_count % recording_freq == 0:
                env.render()
                pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))
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

            reward1_list.append(p1_reward)
            reward2_list.append(p2_reward)
            reward3_list.append(p1_reward_self)
            reward4_list.append(p2_reward_self)

            p1_reward, episode_length = 0, 0
            p2_reward, p1_reward_self, p2_reward_self = 0, 0, 0
            episode_count += 1

            inf = []

            if episode_count < len(states_test):
                state = states_test[episode_count]
                # state[4] = 1
            start_state = state

            state = env.reset_inference_state(state)

            # print('--------------------------')

    if save_logs:
        df = {'start_states': states_test, 'p1_reward': reward3_list, 'p2_reward': reward4_list}
        pandas_df = pd.DataFrame(df)
        pandas_df.to_pickle(pickle_file_name)

    logging.info('#frames: {}'.format(frame_idx))
    if is_recording:
        # Delete images
        [os.remove(recording_path + file) for file in os.listdir(recording_path) if ".png" in file]
    logging.info(
        'Inferred Avg rewards: P1: {}, {}'.format(round(np.mean(reward1_list), 2), round(np.mean(reward2_list), 2)))
    logging.info(
        'Inferred Avg rewards: P1: {}, {}'.format(round(np.mean(reward3_list), 2), round(np.mean(reward4_list), 2)))

    logging.info('0: {}, 1: {}, 2:{}, 3: {}, 4: {}, tot: {}'.format(case_0, case_1, case_2, case_3, case_4,
                                                                    (case_0 + case_1 + case_2 + case_3 + case_4)))
    logging.info('inf_success: {}, inf_failed: {}, %-success: {}'.format(inf_success, inf_failed,
                                                                         (inf_success * 100) / (
                                                                                     inf_success + inf_failed)))


if __name__ == '__main__':
    args = get_args()
    action_size = 5
    if not args.render:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()[1]

    q_set = get_models()[0]
    log_level = logging.DEBUG  # logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    modes = {'a1': [1, 1, policy_na_na, 'a1'], 'a2': [1, 1, policy_na_na_2, 'a2'],
             'b1': [1, 1e3, policy_a_na, 'b1']}

    # modes = {'b1': [1, 1e3, policy_a_na, 'b1']}

    for mode in modes.values():
        env = IntersectionEnv(control_style_ego, control_style_other,
                              time_interval, MAX_TIME / time_interval)
        env.args = args
        test(env, args, q_set, mode)
