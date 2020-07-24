import json

from tensorboardX import SummaryWriter

from parameters import *
from arguments import get_args
from common.utils import print_args
from intersection_env import IntersectionEnv
from train import train
from train_simple import train_simple
import time


def main():
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    args = get_args()
    print_args(args)
    log_dir = args.log_dir + args.mode + '_' + str(int(time.time())) + '/'
    # log_dir = args.log_dir + args.mode + '/'
    print(log_dir)

    writer = SummaryWriter(log_dir)
    with open(log_dir + 'args.json', 'w', encoding='utf-8') as f:
        json.dump(str(vars(args)), f, ensure_ascii=False, indent=4)

    env = IntersectionEnv(control_style_ego, control_style_other,
                          time_interval, MAX_TIME / time_interval)
    env.args = args

    # Use train if intent is encoded in state; else train_simple
    if args.encoding:
        train(env, args, writer, log_dir)
    else:
        "use this for our case"
        train_simple(env, args, writer, log_dir)

if __name__ == "__main__":
    main()
