import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from intersection_simple_nfsp.common.utils import epsilon_scheduler, \
    update_target, print_log, save_model_two, set_global_seeds
from intersection_simple_nfsp.model import DQN, Policy
from intersection_simple_nfsp.storage import ReplayBuffer, ReservoirBuffer
import logging
from tqdm import tqdm


def train(env, args, writer, log_path):
    logger = logging.getLogger(args.mode)
    hdlr = logging.FileHandler(log_path + 'app.log')
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    formatter = logging.Formatter('%(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    # Sets the aggressiveness based on the mode
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
    logger.info(env.ego_car.aggressiveness)
    logger.info(env.other_car.aggressiveness)

    # set_global_seeds(args.seed)

    # RL Model for Player 1
    p1_current_model = DQN(args).to(args.device)
    p1_target_model = DQN(args).to(args.device)

    # RL Model for Player 2
    p2_current_model = DQN(args).to(args.device)
    p2_target_model = DQN(args).to(args.device)

    # SL Model for Player 1, 2
    p1_policy = Policy().to(args.device)
    p2_policy = Policy().to(args.device)

    # load_model_two(args, p1_current_model, p1_policy, p2_current_model, p2_policy)
    update_target(p1_current_model, p1_target_model)
    update_target(p2_current_model, p2_target_model)
    p1_target_model.eval()
    p2_target_model.eval()

    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)

    # Replay Buffer for Reinforcement Learning - Best Response
    p1_replay_buffer = ReplayBuffer(args.rl_size)
    p2_replay_buffer = ReplayBuffer(args.rl_size)

    # Reservoir Buffer for Supervised Learning - Average Strategy
    # TODO(Aiden): How to set buffer size of SL?
    p1_reservoir_buffer = ReservoirBuffer(args.sl_size)
    p2_reservoir_buffer = ReservoirBuffer(args.sl_size)

    # Deque data structure for multi-step learning
    p1_state_deque = deque(maxlen=args.multi_step)
    p1_reward_deque = deque(maxlen=args.multi_step)
    p1_action_deque = deque(maxlen=args.multi_step)

    p2_state_deque = deque(maxlen=args.multi_step)
    p2_reward_deque = deque(maxlen=args.multi_step)
    p2_action_deque = deque(maxlen=args.multi_step)

    # Combination of optimizers 1-works better
    case = args.opt_comb
    print('train_d case: {}'.format(case))

    if case == 1:
        p1_rl_optimizer = optim.SGD(p1_current_model.parameters(), lr=args.lr1)
        p2_rl_optimizer = optim.SGD(p2_current_model.parameters(), lr=args.lr2)
        p1_sl_optimizer = optim.Adam(p1_policy.parameters(), lr=args.lr3)
        p2_sl_optimizer = optim.Adam(p2_policy.parameters(), lr=args.lr4)
    elif case == 2:
        p1_rl_optimizer = optim.Adam(p1_current_model.parameters(), lr=args.lr1)
        p2_rl_optimizer = optim.Adam(p2_current_model.parameters(), lr=args.lr2)
        p1_sl_optimizer = optim.SGD(p1_policy.parameters(), lr=args.lr3)
        p2_sl_optimizer = optim.SGD(p2_policy.parameters(), lr=args.lr4)
    elif case == 3:
        p1_rl_optimizer = optim.SGD(p1_current_model.parameters(), lr=args.lr1)
        p2_rl_optimizer = optim.SGD(p2_current_model.parameters(), lr=args.lr2)
        p1_sl_optimizer = optim.SGD(p1_policy.parameters(), lr=args.lr3)
        p2_sl_optimizer = optim.SGD(p2_policy.parameters(), lr=args.lr4)
    else:
        p1_rl_optimizer = optim.Adam(p1_current_model.parameters(), lr=args.lr1)
        p2_rl_optimizer = optim.Adam(p2_current_model.parameters(), lr=args.lr2)
        p1_sl_optimizer = optim.Adam(p1_policy.parameters(), lr=args.lr3)
        p2_sl_optimizer = optim.Adam(p2_policy.parameters(), lr=args.lr4)

    # Logging
    length_list = []
    p1_reward_list, p1_rl_loss_list, p1_sl_loss_list = [], [], []
    p2_reward_list, p2_rl_loss_list, p2_sl_loss_list = [], [], []
    p1_episode_reward, p2_episode_reward = 0, 0
    tag_interval_length = 0
    prev_time = time.time()
    prev_frame = 1
    episode_count = 0
    episode_step_count = 0
    # Main Loop

    combined_state = env.reset()
    # if random.random() > args.eta:
    #     is_best_response1 = False
    # else:
    #     is_best_response1 = True

    if random.random() > args.eta:
        is_best_response = False
    else:
        is_best_response = True

    if random.random() > args.eta:
        is_best_response2 = False
    else:
        is_best_response2 = True

    p1_state = np.array([combined_state[0], combined_state[1]])
    p2_state = np.array([combined_state[2], combined_state[3]])

    # combined_state = [p1_state[0], p1_state[1], p2_state[0], p2_state[1]]
    # with tqdm(total=(args.max_frames)) as pbar:
    for frame_idx in range(1, args.max_frames + 1):
        episode_step_count += 1

        # Action should be decided by a combination of Best Response and Average Strategy
        s1 = [p1_state[0], p1_state[1], p2_state[0], p2_state[1]]
        s2 = [p2_state[0], p2_state[1], p1_state[0], p1_state[1]]

        assert not (torch.isnan(torch.FloatTensor(s1).to(args.device)).any() or torch.isinf(
            torch.FloatTensor(s1).to(args.device)).any())
        assert not (torch.isnan(torch.FloatTensor(s2).to(args.device)).any() or torch.isinf(
            torch.FloatTensor(s2).to(args.device)).any())

        if not is_best_response:
            p1_action = p1_policy.act(torch.FloatTensor(s1).to(args.device), args.policy_type)
            # p2_action = p2_policy.act(torch.FloatTensor(s2).to(args.device), args.policy_type)
        else:
            epsilon = epsilon_by_frame(frame_idx)
            p1_action = p1_current_model.act(torch.FloatTensor(s1).to(args.device), epsilon)
            # p2_action = p2_current_model.act(torch.FloatTensor(s2).to(args.device), epsilon)

        if not is_best_response2:
            p2_action = p2_policy.act(torch.FloatTensor(s2).to(args.device), args.policy_type)
        else:
            epsilon = epsilon_by_frame(frame_idx)
            p2_action = p2_current_model.act(torch.FloatTensor(s2).to(args.device), epsilon)

        actions = {"1": p1_action, "2": p2_action}

        combined_next_state, reward, done = env.step(actions)

        p1_next_state = np.array([combined_next_state[0], combined_next_state[1]])
        p2_next_state = np.array([combined_next_state[2], combined_next_state[3]])

        s1_next = [p1_next_state[0], p1_next_state[1], p2_next_state[0], p2_next_state[1]]
        s2_next = [p2_next_state[0], p2_next_state[1], p1_next_state[0], p1_next_state[1]]
        # Save current state, reward, action to deque for multi-step learning
        p1_state_deque.append(s1)
        p2_state_deque.append(s2)

        p1_reward = reward[0]  # - 1 if args.negative else reward[0]
        p2_reward = reward[1]  # - 1 if args.negative else reward[1]

        # assert not (p1_reward > 0 or p2_reward > 0)

        p1_reward_deque.append(p1_reward)
        p2_reward_deque.append(p2_reward)

        p1_action_deque.append(p1_action)
        p2_action_deque.append(p2_action)

        # Store (state, action, reward, next_state) to Replay Buffer for Reinforcement Learning
        if len(p1_state_deque) == args.multi_step or done:
            n_reward = multi_step_reward(p1_reward_deque, args.gamma)
            n_state = p1_state_deque[0]
            n_action = p1_action_deque[0]
            p1_replay_buffer.push(n_state, n_action, n_reward, s1_next, np.float32(done))

            n_reward = multi_step_reward(p2_reward_deque, args.gamma)
            n_state = p2_state_deque[0]
            n_action = p2_action_deque[0]
            p2_replay_buffer.push(n_state, n_action, n_reward, s2_next, np.float32(done))

        if is_best_response:
            p1_reservoir_buffer.push(s1, p1_action)
            # p2_reservoir_buffer.push(s2, p2_action)

        if is_best_response2:
            p2_reservoir_buffer.push(s2, p2_action)

        (p1_state, p2_state) = (p1_next_state, p2_next_state)

        # Logging
        p1_episode_reward += p1_reward
        p2_episode_reward += p2_reward
        tag_interval_length += 1
        # pbar.update(1)

        # Episode done. Reset environment and clear logging records
        if done:
            length_list.append(tag_interval_length)
            episode_step_count = 0
            combined_state = env.reset()

            if random.random() > args.eta:
                is_best_response = False
            else:
                is_best_response = True

            if random.random() > args.eta:
                is_best_response2 = False
            else:
                is_best_response2 = True

            p1_state = np.array([combined_state[0], combined_state[1]])
            p2_state = np.array([combined_state[2], combined_state[3]])

            p1_reward_list.append(p1_episode_reward)
            p2_reward_list.append(p2_episode_reward)

            assert not (torch.isnan(torch.FloatTensor([p1_episode_reward]).to(args.device)).any() or torch.isinf(
                torch.FloatTensor([p1_episode_reward]).to(args.device)).any())
            assert not (torch.isnan(torch.FloatTensor([p2_episode_reward]).to(args.device)).any() or torch.isinf(
                torch.FloatTensor([p2_episode_reward]).to(args.device)).any())

            writer.add_scalar("p1/episode_reward", p1_episode_reward, frame_idx)
            writer.add_scalar("p2/episode_reward", p2_episode_reward, frame_idx)
            writer.add_scalar("data/episode_length", tag_interval_length, frame_idx)

            p1_episode_reward, p2_episode_reward, tag_interval_length = 0, 0, 0
            p1_state_deque.clear(), p2_state_deque.clear()
            p1_reward_deque.clear(), p2_reward_deque.clear()
            p1_action_deque.clear(), p2_action_deque.clear()
            episode_count += 1

        # print('p1_rep:{}, p1_res: {}, p2_rep: {}, p2_res: {}'.format(len(p1_replay_buffer), len(p1_reservoir_buffer), len(p2_replay_buffer), len(p2_reservoir_buffer)))

        if (len(p1_replay_buffer) > args.rl_start and
                len(p1_reservoir_buffer) > args.sl_start and
                frame_idx % args.train_freq == 0):
            # print("Computing losses!")

            # Update Best Response with Reinforcement Learning
            p1_rl_loss = compute_rl_loss(p1_current_model, p1_target_model, p1_replay_buffer, p1_rl_optimizer, args)
            p1_rl_loss_list.append(p1_rl_loss.item())
            assert not (torch.isnan(torch.FloatTensor([p1_rl_loss.item()]).to(args.device)).any() or
                        torch.isinf(
                            torch.FloatTensor([p1_rl_loss.item()]).to(args.device)).any()), "loss: {}".format(
                p1_rl_loss.item())
            writer.add_scalar("p1/rl_loss", p1_rl_loss.item(), frame_idx)

            # Update Average Strategy with Supervised Learning
            p1_sl_loss = compute_sl_loss(p1_policy, p1_reservoir_buffer, p1_sl_optimizer, args)
            p1_sl_loss_list.append(p1_sl_loss.item())
            assert not (torch.isnan(torch.FloatTensor([p1_sl_loss.item()]).to(args.device)).any() or
                        torch.isinf(
                            torch.FloatTensor([p1_sl_loss.item()]).to(args.device)).any()), "loss: {}".format(
                p1_sl_loss.item())
            writer.add_scalar("p1/sl_loss", p1_sl_loss.item(), frame_idx)

            p2_rl_loss = compute_rl_loss(p2_current_model, p2_target_model, p2_replay_buffer, p2_rl_optimizer, args)
            p2_rl_loss_list.append(p2_rl_loss.item())
            assert not (torch.isnan(torch.FloatTensor([p2_rl_loss.item()]).to(args.device)).any() or
                        torch.isinf(
                            torch.FloatTensor([p2_rl_loss.item()]).to(args.device)).any()), "loss: {}".format(
                p2_rl_loss.item())
            writer.add_scalar("p2/rl_loss", p2_rl_loss.item(), frame_idx)

            p2_sl_loss = compute_sl_loss(p2_policy, p2_reservoir_buffer, p2_sl_optimizer, args)
            p2_sl_loss_list.append(p2_sl_loss.item())
            assert not (torch.isnan(torch.FloatTensor([p2_sl_loss.item()]).to(args.device)).any() or
                        torch.isinf(
                            torch.FloatTensor([p2_sl_loss.item()]).to(args.device)).any()), "loss: {}".format(
                p2_sl_loss.item())
            writer.add_scalar("p2/sl_loss", p2_sl_loss.item(), frame_idx)

        if frame_idx % args.update_target == 0:
            update_target(p1_current_model, p1_target_model)
            update_target(p2_current_model, p2_target_model)

        # Logging and Saving models
        if frame_idx % args.evaluation_interval == 0:
            print_log(frame_idx, prev_frame, prev_time, (p1_reward_list, p2_reward_list), length_list,
                      (p1_rl_loss_list, p2_rl_loss_list), (p1_sl_loss_list, p2_sl_loss_list), logger)
            p1_reward_list.clear(), p2_reward_list.clear(), length_list.clear()
            p1_rl_loss_list.clear(), p2_rl_loss_list.clear()
            p1_sl_loss_list.clear(), p2_sl_loss_list.clear()
            prev_frame = frame_idx
            prev_time = time.time()

        if frame_idx % args.model_save_freq == 0:
            save_model_two(args, log_path, models={"p1": p1_current_model, "p2": p2_current_model},
                           policies={"p1": p1_policy, "p2": p2_policy}, itr=frame_idx)


def compute_sl_loss(policy, reservoir_buffer, optimizer, args):
    state, action = reservoir_buffer.sample(args.batch_size_sl)

    state = torch.FloatTensor(state).to(args.device)
    action = torch.LongTensor(action).to(args.device)

    probs = policy(state)
    probs_with_actions = probs.gather(1, action.unsqueeze(1))

    log_probs = probs_with_actions.log()
    loss = -1 * log_probs.mean()

    # loss = -1 * probs_with_actions.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def compute_rl_loss(current_model, target_model, replay_buffer, optimizer, args):
    state, action, reward, next_state, done = replay_buffer.sample(args.batch_size_rl)
    weights = torch.ones(args.batch_size_rl)

    state = torch.FloatTensor(state).to(args.device)
    next_state = torch.FloatTensor(next_state).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    # print('action: {}'.format(action))
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    # Q-Learning with target network
    q_values = current_model(state)
    target_next_q_values = target_model(next_state)
    next_q_value = target_next_q_values.max(1)[0]

    # next_q_values = current_model(next_state)
    # next_actions = next_q_values.max(1)[1].unsqueeze(1)
    # next_q_value = target_next_q_values.gather(1, next_actions).squeeze(1)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    expected_q_value = reward + (args.gamma ** args.multi_step) * next_q_value * (1 - done)

    loss = F.mse_loss(q_value, expected_q_value.detach())

    # loss = (loss * weights).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret
