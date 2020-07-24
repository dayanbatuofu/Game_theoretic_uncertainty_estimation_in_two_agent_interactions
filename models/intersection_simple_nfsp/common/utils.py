import datetime
import math
import os
import pathlib
import random
import time

import numpy as np
import torch


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def switch_target(current_model, target_model):
    temp = target_model.state_dict()
    target_model.load_state_dict(current_model.state_dict())
    current_model.load_state_dict(temp)


def epsilon_scheduler(eps_start, eps_final, eps_decay):
    def function(frame_idx):
        return eps_final + (eps_start - eps_final) * math.exp(-1. * frame_idx / eps_decay)

    return function


def create_log_dir(args):
    log_dir = ""
    log_dir = log_dir + "{}-".format(args.env)
    if args.negative:
        log_dir = log_dir + "negative-"
    if args.multi_step != 1:
        log_dir = log_dir + "{}-step-".format(args.multi_step)
    if args.dueling:
        log_dir = log_dir + "dueling-"
    log_dir = log_dir + "dqn-{}".format(args.save_model)

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = log_dir + '-' + now
    log_dir = os.path.join("runs", log_dir)
    return log_dir


def print_log(frame, prev_frame, prev_time, rewards, length_list, rl_losses, sl_losses, logger):
    fps = (frame - prev_frame) / (time.time() - prev_time)
    # print(rewards[0])
    p1_avg_reward, p2_avg_reward = (np.mean(rewards[0]), np.mean(rewards[1]))
    if len(rl_losses[0]) != 0:
        p1_avg_rl_loss, p2_avg_rl_loss = (np.mean(rl_losses[0]), np.mean(rl_losses[1]))
        p1_avg_sl_loss, p2_avg_sl_loss = (np.mean(sl_losses[0]), np.mean(sl_losses[1]))
    else:
        p1_avg_rl_loss, p2_avg_rl_loss = 0., 0.
        p1_avg_sl_loss, p2_avg_sl_loss = 0., 0.

    avg_length = np.mean(length_list)

    # print("Frame: {:<8} FPS: {:.2f} Avg. Tagging Interval Length: {:.2f}".format(frame, fps, avg_length))
    # print("Player 1 Avg. Reward: {:.2f} Avg. RL/SL Loss: {:.2f}/{:.2f}".format(
    #     p1_avg_reward, p1_avg_rl_loss, p1_avg_sl_loss))
    # print("Player 2 Avg. Reward: {:.2f} Avg. RL/SL Loss: {:.2f}/{:.2f}".format(
    #     p2_avg_reward, p2_avg_rl_loss, p2_avg_sl_loss))
    logger.info("Frame: {:<8} FPS: {:.2f} Avg. Tagging Interval Length: {:.2f}".format(frame, fps, avg_length))
    logger.info("Player 1 Avg. Reward: {:.4f} Avg. RL/SL Loss: {:.4f}/{:.4f}".format(
        p1_avg_reward, p1_avg_rl_loss, p1_avg_sl_loss))
    logger.info("Player 2 Avg. Reward: {:.4f} Avg. RL/SL Loss: {:.4f}/{:.4f}".format(
        p2_avg_reward, p2_avg_rl_loss, p2_avg_sl_loss))

def print_log_other(frame, prev_frame, prev_time, rewards, length_list, rl_losses, sl_losses):
    fps = (frame - prev_frame) / (time.time() - prev_time)
    # print(rewards[0])
    p1_avg_reward, p2_avg_reward = (np.mean(rewards[0]), np.mean(rewards[1]))
    if len(rl_losses) != 0:
        p2_avg_rl_loss = np.mean(rl_losses)
        p2_avg_sl_loss = np.mean(sl_losses)
    else:
        p1_avg_rl_loss, p2_avg_rl_loss = 0., 0.
        p1_avg_sl_loss, p2_avg_sl_loss = 0., 0.

    avg_length = np.mean(length_list)

    print("Frame: {:<8} FPS: {:.2f} Avg. Tagging Interval Length: {:.2f}".format(frame, fps, avg_length))
    print("Player 1 Avg. Reward: {:.2f} ".format(
        p1_avg_reward))
    print("Player 2 Avg. Reward: {:.2f} Avg. RL/SL Loss: {:.2f}/{:.2f}".format(
        p2_avg_reward, p2_avg_rl_loss, p2_avg_sl_loss))


def print_log_episodes(episode_count, rewards, length_list, rl_losses, sl_losses):
    p1_avg_reward, p2_avg_reward = (np.mean(rewards[0]), np.mean(rewards[1]))
    if len(rl_losses[0]) != 0:
        p1_avg_rl_loss, p2_avg_rl_loss = (np.mean(rl_losses[0]), np.mean(rl_losses[1]))
        p1_avg_sl_loss, p2_avg_sl_loss = (np.mean(sl_losses[0]), np.mean(sl_losses[1]))
    else:
        p1_avg_rl_loss, p2_avg_rl_loss = 0., 0.
        p1_avg_sl_loss, p2_avg_sl_loss = 0., 0.

    avg_length = np.mean(length_list)

    print("Episode: {}, Avg. Tagging Interval Length: {:.2f}".format(episode_count, avg_length))
    print("Player 1 Avg. Reward: {:.2f} Avg. RL/SL Loss: {:.2f}/{:.2f}".format(
        p1_avg_reward, p1_avg_rl_loss, p1_avg_sl_loss))
    print("Player 2 Avg. Reward: {:.2f} Avg. RL/SL Loss: {:.2f}/{:.2f}".format(
        p2_avg_reward, p2_avg_rl_loss, p2_avg_sl_loss))


def print_args(args):
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))


def save_model(models, policies, args):
    fname = ""
    fname += "{}-".format(args.env)
    if args.negative:
        fname += "negative-"
    if args.multi_step != 1:
        fname += "{}-step-".format(args.multi_step)
    if args.dueling:
        fname += "dueling-"

    fname += "dqn-{}".format(args.save_model)

    fname = os.path.join("models", fname)

    pathlib.Path('models').mkdir(exist_ok=True)
    torch.save({
        'p1_model': models['p1'].state_dict(),
        'p2_model': models['p2'].state_dict(),
        'p1_policy': policies['p1'].state_dict(),
        'p2_policy': policies['p2'].state_dict(),
    }, fname + 'shallow-256-10L_final' + '.pth')


def load_model(models, policies, args):
    if args.load_model is not None:
        fname = os.path.join("models", args.load_model)
        fname += ".pth"
    else:
        fname = ""
        fname += "{}-".format(args.env)
        if args.negative:
            fname += "negative-"
        if args.multi_step != 1:
            fname += "{}-step-".format(args.multi_step)
        if args.dueling:
            fname += "dueling-"
        fname += "dqn-{}.pth".format(args.save_model)
        fname = os.path.join("models", fname)

    fname = 'models/pygame-dqn-modelshallow-256-10L_final.pth'
    print(fname)

    # Hack to load models saved with GPU
    if args.device == torch.device("cpu"):
        map_location = lambda storage, loc: storage
    else:
        map_location = None

    if not os.path.exists(fname):
        raise ValueError("No model saved with name {}".format(fname))

    checkpoint = torch.load(fname, map_location)
    models['p1'].load_state_dict(checkpoint['p1_model'])
    models['p2'].load_state_dict(checkpoint['p2_model'])
    policies['p1'].load_state_dict(checkpoint['p1_policy'])
    policies['p2'].load_state_dict(checkpoint['p2_policy'])


def save_model_two(args, log_path, models, policies, itr):
    # fname = args.model_dir + args.mode + '/'
    fname = log_path


    # pathlib.Path(fname).mkdir(exist_ok=True)
    if not os.path.exists(fname):
        os.makedirs(fname)

    torch.save({
        'p1_model': models['p1'].state_dict(),
        # 'p2_model': models['p2'].state_dict(),
        'p1_policy': policies['p1'].state_dict(),
        # 'p2_policy': policies['p2'].state_dict(),
    }, fname + 'player_1_{}.pth'.format(itr))

    torch.save({
        'p2_model': models['p2'].state_dict(),
        'p2_policy': policies['p2'].state_dict(),
    }, fname + 'player_2_{}.pth'.format(itr))


def load_model_two(args, p1_current_model, p1_policy, p2_current_model, p2_policy):
    model_num = args.load_model_index  # args.max_frames

    fname1 = args.model_dir + args.mode + '/player_1_' + str(model_num) + '.pth'
    fname2 = args.model_dir + args.mode + '/player_2_' + str(model_num) + '.pth'

    print(fname1)
    print(fname2)
    # Hack to load models saved with GPU
    if args.device == torch.device("cpu"):
        map_location = lambda storage, loc: storage
    else:
        map_location = None

    if not os.path.exists(fname1):
        raise ValueError("No model saved with name {}".format(fname1))

    checkpoint = torch.load(fname1, map_location)
    p1_current_model.load_state_dict(checkpoint['p1_model'])
    p1_policy.load_state_dict(checkpoint['p1_policy'])

    if args.device == torch.device("cpu"):
        map_location = lambda storage, loc: storage
    else:
        map_location = None

    if not os.path.exists(fname2):
        raise ValueError("No model saved with name {}".format(fname2))

    checkpoint = torch.load(fname2, map_location)
    p2_current_model.load_state_dict(checkpoint['p2_model'])
    p2_policy.load_state_dict(checkpoint['p2_policy'])


def load_one_model(fname, args, model, policy, num):
    fname = 'models/' + fname
    print(fname)

    # Hack to load models saved with GPU
    if args.device == torch.device("cpu"):
        map_location = lambda storage, loc: storage
    else:
        map_location = None

    if not os.path.exists(fname):
        raise ValueError("No model saved with name {}".format(fname))

    checkpoint = torch.load(fname, map_location)
    model.load_state_dict(checkpoint['p' + str(num) + '_model'])
    policy.load_state_dict(checkpoint['p' + str(num) + '_policy'])


def save_model_one(model, policy, num, behavior):
    fname = 'models/'

    pathlib.Path('models').mkdir(exist_ok=True)
    torch.save({
        'p' + str(num) + '_model': model.state_dict(),
        'p' + str(num) + '_policy': policy.state_dict(),
    }, fname + 'player_' + str(num) + '_' + behavior + '.pth')


def set_global_seeds(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass

    np.random.seed(seed)
    random.seed(seed)
