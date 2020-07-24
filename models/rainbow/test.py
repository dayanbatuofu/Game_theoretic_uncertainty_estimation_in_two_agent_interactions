import os
import sys

import numpy as np
import pandas as pd
import pygame as pg
import torch

from parameters import control_style_ego, control_style_other, time_interval, MAX_TIME
from arguments import get_args
from common.utils import create_movie
from intersection_env import IntersectionEnv
from model import DQN
from set_nfsp_models import get_models
from constants import CONSTANTS as C

is_recording = False
recording_freq = 1
total_eps = 10000
save_logs = False


def test(env, args, mode):
    print('mode: {} \r'.format(mode[3]))

    states_test_df = pd.read_pickle('./random.pkl')
    states_test = list(states_test_df['states'])
    pickle_file_name = './start_slice/'
    # states_test = states_test[:1]

    # ego_position_n = 100.  # number of grid points - 1
    # ego_speed_n = 100.
    # ego_position_range = [25.0, 45.0]
    # ego_position_grid = np.arange(ego_position_range[0], ego_position_range[1],
    #                               (ego_position_range[1] - ego_position_range[0]) / ego_position_n)
    # # print(ego_position_grid)
    # ego_speed_range = [5.0, 15.0]
    # ego_speed_grid = np.arange(ego_speed_range[0], ego_speed_range[1],
    #                            (ego_speed_range[1] - ego_speed_range[0]) / ego_speed_n)
    #
    # states_test1 = []
    # for position in ego_position_grid:
    #     for speed in ego_speed_grid:
    #         states_test1.append([position - 0.1, speed, position, speed, 0.0])
    #
    # states_test = states_test1
    # print('Test_states: {}'.format(len(states_test)))

    # states_test1 = [[41.5,  5.6, 41.6,  5.6,  2.]]

    # for position_other, speed_other in zip(ego_position_grid, ego_speed_grid):
    #     states_test1.append([position_other-0.01, speed_other, position_other, speed_other])

    # states_test1 = []
    # pos = 30.5
    # while len(states_test1) < 2000:
    #     # states_test1.append([14.125, 17.2,   19.5,    8.2  , 2.0])
    #     states_test1.append([pos, 10.2, pos, 10.2, 2.0])
    #     # pos = pos - 0.005
    # states_test = states_test1

    # states_test_df = pd.read_pickle('./random_slice_5.pkl')
    # states_test = list(states_test_df['start_states'])
    # print(len(states_test))

    is_rl_enabled = True

    if is_recording:
        env.is_recording = True
        env.args.render = True
        recording_path = 'test_recordings/' + args.mode + '/' + mode[3] + '/'
        env.recording_path = recording_path
        if not os.path.exists(recording_path):
            os.makedirs(recording_path)

    if is_rl_enabled:
        # fname = args.log_dir + args.mode + '/' + args.mode + '_' + str(args.load_model_index) + '.pth'
        fname = args.log_dir + args.mode + '/model_' + str(args.load_model_index) + '.pth'
        # fname = args.log_dir + args.mode + '/model.pth'
        # print(fname)
        if 'neutral' in args.mode:
            pickle_file_name += 'neutral_ne'
            env.ego_car.gracefulness = 0
        elif 'g1' in args.mode:
            pickle_file_name += 'g1_ne'
            env.ego_car.gracefulness = 1
        elif 'g2' in args.mode:
            pickle_file_name += 'g2_ne'
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

        # this hack needed for models saved directly using DataParallel
        if torch.cuda.device_count() > 1:
            # original saved file with DataParallel
            state_dict = torch.load(fname, map_location)
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            current_model.load_state_dict(new_state_dict)
        else:
            current_model.load_state_dict(torch.load(fname, map_location))

    env.ego_car.aggressiveness = mode[0]
    env.other_car.aggressiveness = mode[1]
    if save_logs:
        pickle_file_name += '_' + mode[3] + '.pkl'
        print(pickle_file_name)

    # states_test_df = pd.read_pickle('./states_test.pkl')
    # states_test = list(states_test_df['states_test'])

    # state = [state[0], state[1], state[2], state[3], theta_mode]

    p1_reward = 0
    episode_length = 0
    reward1_list, reward2_list, reward3_list, reward4_list = [], [], [], []
    episode_count = 0
    p2_reward, p1_reward_self, p2_reward_self = 0, 0, 0
    frame_idx = 0
    case_0, case_1, case_2, case_3, case_4 = 0, 0, 0, 0, 0

    state = states_test[episode_count]
    state = env.reset_state(state)
    print(state)
    start_state = state
    ego_first_states = []
    human_decelerated = False

    while episode_count != len(states_test):
        # if episode_count % 1000 == 0:
        #     print(episode_count)

        frame_idx += 1
        p1_state = [state[0], state[1], state[2], state[3]]
        p2_state = [state[2], state[3], state[0], state[1]]

        action1 = current_model.best_act(torch.FloatTensor(state).to(args.device))
        action2 = mode[2].act(torch.FloatTensor(p2_state).to(args.device))

        if args.render:
            env.render()

        actions = {"1": action1, "2": action2}
        # print(actions, end= ' ')
        # print(action2, end=' ')

        # if int(action2) < 2 and (state[2] + C.CAR_LENGTH * 0.5 + 1. > 0.):
        #     # print(actions, max_q_index, env.frame)
        #     human_decelerated = True
        # if is_recording and episode_count % recording_freq == 0:
        #     # env.render()
        #     pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))
        next_state, reward, done = env.step(actions)
        # print(state, actions)
        reward1 = reward[0]

        state = next_state
        p1_reward += reward1
        p2_reward += reward[1]
        p1_reward_self += reward[2]
        p2_reward_self += reward[3]
        episode_length += 1

        if done:
            # print([start_state, episode_count, p1_reward, p2_reward])
            if env.ego_crossed_first and human_decelerated:
                # print(start_state, episode_count)
                ego_first_states.append(start_state)

            # if is_recording and episode_count % recording_freq == 0:
            #     env.render()
            #     pg.image.save(env.renderer.screen, "%simg%03d.png" % (recording_path, episode_length))

            if env.isCollision:
                case_0 += 1
                # print('collision for state: {}'.format(start_state))
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

            reward1_list.append(p1_reward)
            reward2_list.append(p2_reward)
            reward3_list.append(p1_reward_self)
            reward4_list.append(p2_reward_self)
            p1_reward, episode_length = 0, 0
            env.total_frames = 0
            p2_reward, p1_reward_self, p2_reward_self = 0, 0, 0
            episode_count += 1
            human_decelerated = False

            if episode_count < len(states_test):
                state = states_test[episode_count]
            state = env.reset_state(state)
            start_state = state
            # print(state, end='')
            # sys.exit()
            # print('--------------------------')
    if save_logs:
        df = {'start_states': states_test, 'p1_reward': reward3_list, 'p2_reward': reward4_list}
        pandas_df = pd.DataFrame(df)
        pandas_df.to_pickle(pickle_file_name)

        # df1 = {'states': ego_first_states}
        # pd.DataFrame(df1).to_pickle('./start_0.1_0.6/ego_first_states.pkl')

    print('#frames: {}'.format(frame_idx))
    if is_recording:
        # Delete images
        [os.remove(recording_path + file) for file in os.listdir(recording_path) if ".png" in file]
    print('Avg rewards: P1: {}, {}'.format(round(np.mean(reward1_list), 4), round(np.mean(reward2_list), 4)))
    print('0: {}, 1: {}, 2:{}, 3: {}, 4: {}, tot: {}'.format(case_0, case_1, case_2, case_3, case_4,
                                                             (case_0 + case_1 + case_2 + case_3 + case_4)))


if __name__ == '__main__':
    # Uncomment to run on server with display

    args = get_args()
    action_size = 5
    if not args.render:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()[1]

    # q_set = [Q_na_na, Q_na_a, Q_a_na, Q_a_a]

    modes = {'a1': [1, 1, policy_na_na, 'a1'], 'a2': [1, 1, policy_na_na_2, 'a2'],
             'b1': [1, 1e3, policy_a_na, 'b1']}

    # modes = {'a2': [1, 1, policy_na_na_2, 'a2']}

    # modes = {'b1': [1, 1e3, policy_a_na, 'b1']}

    for mode in modes.values():
        env = IntersectionEnv(control_style_ego, control_style_other,
                              time_interval, MAX_TIME / time_interval)
        # env.max_time_steps = 100
        env.args = args
        test(env, args, mode)
