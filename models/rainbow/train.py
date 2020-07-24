import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchcontrib.optim import SWA
from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from common.utils import epsilon_scheduler, beta_scheduler, update_target, print_log
from model import DQN
from set_nfsp_models import get_models


# from rainbow.run_test import run_test

def train(env, args, writer, log_dir):
    current_model = DQN(env, args).to(args.device)
    target_model = DQN(env, args).to(args.device)
    if args.gpu_count > 0 and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        current_model = nn.DataParallel(current_model).to(args.device)
        target_model = nn.DataParallel(target_model).to(args.device)

        # current_model.to(args.device)
        # target_model.to(args.device)
    
    choice = -1

    if 'neutral' in args.mode:
        env.ego_car.gracefulness = 0
    elif 'g1' in args.mode:
        env.ego_car.gracefulness = 1
    elif 'g2' in args.mode:
        env.ego_car.gracefulness = 1000

    print(env.ego_car.gracefulness)
    # fname = args.log_dir + args.mode + '/model'
    fname = log_dir + 'model'
    print('model: {}'.format(fname))

    if args.noisy:
        current_model.update_noisy_modules()
        target_model.update_noisy_modules()

    # if args.load_model and os.path.isfile(args.load_model):
    #     load_model(current_model, args)

    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)
    beta_by_frame = beta_scheduler(args.beta_start, args.beta_frames)

    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args.buffer_size, args.alpha)
    else:
        replay_buffer = ReplayBuffer(args.buffer_size)

    (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2) = get_models()[1]

    q_set = get_models()[0]

    state_deque = deque(maxlen=args.multi_step)
    reward_deque = deque(maxlen=args.multi_step)
    action_deque = deque(maxlen=args.multi_step)

    if args.nesterov:
        optimizer = optim.SGD(current_model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    else:
        optimizer = optim.Adam(current_model.parameters(), lr=args.lr)

    # optimizer = optim.Adam(current_model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=0.01)
    # optimizer = SWA(base_optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)

    p2_r_list, reward_list, reward_1_list, p1_r_list, length_list, loss_list = [], [], [], [], [], []
    episode_reward, episode_reward_1, p2_reward, p1_reward, episode_length, episode_count = 0, 0, 0, 0, 0, 0

    prev_time = time.time()
    prev_frame = 1

    # state = env.reset()
    env.other_car.aggressiveness = np.random.choice([1000, 1], 1, p=[0.5, 0.5])[0]
    if env.other_car.aggressiveness == 1:
        choice = np.random.choice([1, 2], 1, p=[0.5, 0.5])[0]

    state = env.reset_inference()
    # frame_idx = 0
    for frame_idx in range(1, args.max_frames + 1):
        if args.render:
            env.render()

        if args.noisy:
            current_model.sample_noise()
            target_model.sample_noise()

        epsilon = epsilon_by_frame(frame_idx)
        p2_state = [state[2], state[3], state[0], state[1]]
        if args.gpu_count > 0 and torch.cuda.device_count() > 1:
            action1 = current_model.module.act(torch.FloatTensor(state).to(args.device), epsilon)
        else:
            action1 = current_model.act(torch.FloatTensor(state).to(args.device), epsilon)

        if env.other_car.aggressiveness == 1000:
            action2 = policy_a_na.act(torch.FloatTensor(p2_state).to(args.device))
        elif env.other_car.aggressiveness == 1:

            if choice == 1:
                action2 = policy_na_na.act(torch.FloatTensor(p2_state).to(args.device))
            else:
                action2 = policy_na_na_2.act(torch.FloatTensor(p2_state).to(args.device))

        actions = {"1": action1, "2": action2}

        next_state, reward, done = env.step_inference(actions, q_set)

        p1_reward += reward[2]
        p2_reward += reward[3]

        state_deque.append(state)
        reward_deque.append(reward[0])
        action_deque.append(action1)

        if len(state_deque) == args.multi_step or done:
            n_reward = multi_step_reward(reward_deque, args.gamma)
            n_state = state_deque[0]
            n_action = action_deque[0]
            replay_buffer.push(n_state, n_action, n_reward, next_state, np.float32(done))

        state = next_state
        episode_reward += reward[0]
        episode_reward_1 += reward[1]
        episode_length += 1

        if done:
            state = env.reset_inference()
            # state = env.reset()
            env.other_car.aggressiveness = np.random.choice([1000, 1], 1, p=[0.5, 0.5])[0]
            if env.other_car.aggressiveness == 1:
                choice = np.random.choice([1, 2], 1, p=[0.5, 0.5])[0]

            reward_list.append(episode_reward)
            reward_1_list.append(episode_reward_1)
            p2_r_list.append(p2_reward)
            p1_r_list.append(p1_reward)
            episode_count += 1

            length_list.append(episode_length)
            writer.add_scalar("data/p1_reward", episode_reward, frame_idx)
            writer.add_scalar("data/episode_length", episode_length, frame_idx)
            writer.add_scalar("data/p2_reward", p2_reward, frame_idx)
            p2_reward, p1_reward, episode_reward, episode_reward_1, episode_length = 0, 0, 0, 0, 0
            state_deque.clear()
            reward_deque.clear()
            action_deque.clear()

        if len(replay_buffer) > args.learning_start and frame_idx % args.train_freq == 0:
            beta = beta_by_frame(frame_idx)
            loss = compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta)
            loss_list.append(loss.item())
            writer.add_scalar("data/loss", loss.item(), frame_idx)

        if frame_idx % args.update_target == 0:
            update_target(current_model, target_model)

        if frame_idx % args.evaluation_interval == 0:
            print_log(frame_idx, prev_frame, prev_time, reward_list, reward_1_list, length_list, loss_list)
            writer.add_scalar("data/mean_p1", np.mean(reward_list), frame_idx)
            writer.add_scalar("data/mean_p2", np.mean(reward_1_list), frame_idx)

            if frame_idx / args.evaluation_interval >= 5:
                if np.mean(p1_r_list) > -10 and np.mean(p2_r_list) > -10:
                    torch.save(current_model.state_dict(), fname + '_' + str(frame_idx) + '.pth')

                # if np.mean(reward_list) > -250 and np.mean(reward_1_list) > -250:
                #     torch.save(current_model.state_dict(), fname + '_' + str(frame_idx) + '_i.pth')

            p1_r_list.clear(), p2_r_list.clear(), reward_list.clear(), reward_1_list.clear(), length_list.clear(), loss_list.clear()
            prev_frame = frame_idx
            prev_time = time.time()

        try:
            state_dict = current_model.module.state_dict()
        except AttributeError:
            state_dict = current_model.state_dict()

        if frame_idx % args.save_freq == 0:
            torch.save(state_dict, fname + '_' + str(frame_idx) + '.pth')
        
        # save anyway
        torch.save(state_dict, fname + '_' + str(frame_idx) + '.pth')

    # save_model(current_model, args)
    # torch.save(current_model.state_dict(), fname + str(frame_idx) + '.pth')


def compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta=None):
    """
    Calculate loss and optimize for non-c51 algorithm
    """
    if args.prioritized_replay:
        state, action, reward, next_state, done, weights, indices = replay_buffer.sample(args.batch_size, beta)
    else:
        state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
        weights = torch.ones(args.batch_size)

    state = torch.FloatTensor(np.float32(state)).to(args.device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    if not args.c51:
        q_values = current_model(state)
        target_next_q_values = target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        if args.double:
            next_q_values = current_model(next_state)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            next_q_value = target_next_q_values.gather(1, next_actions).squeeze(1)
        else:
            next_q_value = target_next_q_values.max(1)[0]

        expected_q_value = reward + (args.gamma ** args.multi_step) * next_q_value * (1 - done)

        # loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
        # if args.prioritized_replay:
        #     prios = torch.abs(loss) + 1e-5
        # loss = (loss * weights).mean()
        loss = F.mse_loss(q_value, expected_q_value.detach())

    else:
        q_dist = current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(args.batch_size, 1, args.num_atoms)
        q_dist = q_dist.gather(1, action).squeeze(1)
        q_dist.data.clamp_(0.01, 0.99)

        target_dist = projection_distribution(current_model, target_model, next_state, reward, done,
                                              target_model.support, target_model.offset, args)

        loss = - (target_dist * q_dist.log()).sum(1)
        if args.prioritized_replay:
            prios = torch.abs(loss) + 1e-6
        loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    if args.prioritized_replay:
        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()
    return loss


def projection_distribution(current_model, target_model, next_state, reward, done, support, offset, args):
    delta_z = float(args.Vmax - args.Vmin) / (args.num_atoms - 1)

    target_next_q_dist = target_model(next_state)

    if args.double:
        next_q_dist = current_model(next_state)
        next_action = (next_q_dist * support).sum(2).max(1)[1]
    else:
        next_action = (target_next_q_dist * support).sum(2).max(1)[1]

    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(target_next_q_dist.size(0), 1,
                                                               target_next_q_dist.size(2))
    target_next_q_dist = target_next_q_dist.gather(1, next_action).squeeze(1)

    reward = reward.unsqueeze(1).expand_as(target_next_q_dist)
    done = done.unsqueeze(1).expand_as(target_next_q_dist)
    support = support.unsqueeze(0).expand_as(target_next_q_dist)

    Tz = reward + args.gamma * support * (1 - done)
    Tz = Tz.clamp(min=args.Vmin, max=args.Vmax)
    b = (Tz - args.Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    target_dist = target_next_q_dist.clone().zero_()
    target_dist.view(-1).index_add_(0, (l + offset).view(-1), (target_next_q_dist * (u.float() - b)).view(-1))
    target_dist.view(-1).index_add_(0, (u + offset).view(-1), (target_next_q_dist * (b - l.float())).view(-1))

    return target_dist


def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret