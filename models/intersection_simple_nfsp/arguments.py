import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='DQN')

    # Logging arguments
    parser.add_argument('--mode', type=str, default='',
                        help='Mode')

    # Path to save the training graph
    parser.add_argument('--log-dir', type=str, default='runs/',
                        help='Directory to save the log files')
    # Path to save the models
    parser.add_argument('--model-dir', type=str, default='runs/',
                        help='Directory to save models')

    parser.add_argument('--load-model-index', type=int, default=5000000,
                        help='Loading index for the model')

    parser.add_argument('--model-save-freq', type=int, default=100000,
                        help='Freq to save the model')

    # Basic Arguments
    parser.add_argument('--seed', type=int, default=1122,
                        help='Random seed')
    parser.add_argument('--batch-size-rl', type=int, default=32,
                        help='Batch size RL')
    parser.add_argument('--batch-size-sl', type=int, default=32,
                        help='Batch size SL')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Training Arguments
    parser.add_argument('--policy-type', type=int, default=0,
                        help='Policy type- 0 for Categorial and 1 for Argmax')
    parser.add_argument('--max-frames', type=int, default=5000000,
                        help='Number of frames to train')
    parser.add_argument('--rl-size', type=int, default=2000000,
                        help='Maximum memory buffer size')
    parser.add_argument('--sl-size', type=int, default=20000000,
                        help='Maximum memory buffer size')

    parser.add_argument('--update-target', type=int, default=1000,
                        help='Interval of target network update')
    # parser.add_argument('--switch-target', type=int, default=3500,
    #                     help='Interval of network switch')
    parser.add_argument('--train-freq', type=int, default=1,
                        help='Number of steps between optimization step')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='Anticipatory Parameter for NFSP')
    parser.add_argument('--rl-start', type=int, default=10000,
                        help='How many steps of the model to collect transitions for before RL starts')
    parser.add_argument('--sl-start', type=int, default=1000,
                        help='How many steps of the model to collect transitions for before SL starts')

    parser.add_argument('--opt-comb', type=int, default=4,
                        help='Optimizer combination for RL and SL networks')

    # Algorithm Arguments
    parser.add_argument('--dueling', action='store_true',
                        help='Enable Dueling Network')
    parser.add_argument('--multi-step', type=int, default=1,
                        help='N-Step Learning')

    # Environment Arguments
    parser.add_argument('--env', type=str, default='pygame',
                        help='Environment Name')
    parser.add_argument('--negative', action='store_true', default=False,
                        help='Give negative(-1) reward for not done.')

    # Evaluation Arguments
    parser.add_argument('--load-model', type=str, default=None,
                        help='Pretrained model name to load (state dict)')
    parser.add_argument('--save-model', type=str, default='model',
                        help='Pretrained model name to save (state dict)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate only')
    parser.add_argument('--render', action='store_true',
                        help='Render evaluation agent')
    parser.add_argument('--evaluation-interval', type=int, default=10000,
                        help='Frames for evaluation interval')

    # Optimization Arguments
    # RL network learning rates
    parser.add_argument('--lr1', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lr2', type=float, default=1e-3,
                        help='Learning rate')

    # SL network learning rates
    parser.add_argument('--lr3', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--lr4', type=float, default=1e-5,
                        help='Learning rate')

    parser.add_argument('--eps-start', type=float, default=1.0,
                        help='Start value of epsilon')
    parser.add_argument('--eps-final', type=float, default=0.01,
                        help='Final value of epsilon')
    parser.add_argument('--eps-decay', type=int, default=30000,
                        help='Adjustment parameter for epsilon')

    # Enable to penalize agents for deceleration
    parser.add_argument('--acc-loss', action='store_true',
                        help='Enable acc_loss')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args
