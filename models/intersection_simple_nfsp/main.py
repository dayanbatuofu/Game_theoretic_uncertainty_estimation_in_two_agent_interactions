from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json

from tensorboardX import SummaryWriter

from arguments import get_args
from common.utils import print_args
from intersection_env import IntersectionEnv
from parameters import *
# from train import train
from train_d import train
import time

def main():
    # Uncomment to run on server without display
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    try:
        args = get_args()
        print_args(args)
        # log-dir path set to write logs
        log_dir = args.log_dir + args.mode + '_' + str(int(time.time())) + '/'
        print(log_dir)

        # create a writer for logs
        writer = SummaryWriter(log_dir)
        # save the parameters used during train to a json file
        with open(log_dir + 'args.json', 'w', encoding='utf-8') as f:
            json.dump(str(vars(args)), f, ensure_ascii=False, indent=4)

        env = IntersectionEnv(control_style_ego, control_style_other,
                              time_interval, MAX_TIME / time_interval)
        env.args = args
        train(env, args, writer, log_dir)
    finally:
        pass


if __name__ == "__main__":
    main()
