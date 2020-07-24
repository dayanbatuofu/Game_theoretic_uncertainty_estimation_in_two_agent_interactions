from parameters import control_style_ego, control_style_other, time_interval, MAX_TIME
from intersection_env_test import IntersectionEnv

import pandas as pd
import numpy as np

# Helper script to create test states
if __name__ == '__main__':
    epsilon = 0.1
    fname1 = './random_select' + str(epsilon) + '_1e6.pkl'
    fname2 = './random.pkl'
    env = IntersectionEnv(control_style_ego, control_style_other,
                              time_interval, MAX_TIME / time_interval)
    total_eps = 1e6

    test_states = set()
    while len(test_states) != total_eps:
        # s = env.reset_inference()
        # print(s)
        # test_states.add(tuple(s))
        state = []
        state.append(np.random.uniform(25, 45))
        # self.ego_car.state[0] = 3.0
        state.append(np.random.uniform(5, 15))
        state.append(np.random.uniform(25, 45))
        state.append(np.random.uniform(5, 15))

        test_states.add((state[0], state[1], state[2], state[3], 2.0))
        # test_states.append([s[2]-epsilon, s[3], s[2], s[3], s[4]])

    df = {'states': list(test_states)}
    print(len(test_states))

    pandas_df = pd.DataFrame(df)
    pandas_df.to_pickle(fname1)

