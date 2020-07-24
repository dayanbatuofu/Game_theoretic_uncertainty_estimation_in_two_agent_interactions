from math import floor

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sb
from matplotlib import pyplot
import numpy as np
from constants import CONSTANTS as C
import logging

plt.rcParams['figure.figsize'] = (15.0, 15.0) # set default size of plots
font = {'family' : 'sans',
        'weight' : 'normal',
        'size' : 8}
plt.matplotlib.rc('font', **font)
pyplot.locator_params(axis='y', nbins=6)
pyplot.locator_params(axis='x', nbins=10)
# logging.getLogger('matplotlib.font_manager').disabled = True
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def draw_grid(f1, f2, grid, mode):
    print(f1, f2)
    f1 = f1.split('/')
    f2 = f2.split('/')
    ft1 = f1[2].split('_')
    ft2 = f2[2].split('_')
    fname = f1[0]+ '/' + f1[1]+ '/' + ft1[0] + '_' + ft2[0] + '_' + mode
    # print(fname)
    fig, ax = plt.subplots()
    sb.heatmap(grid, linewidth=0.5)

    ax.set_xticks(np.arange(ego_position_n))
    ax.set_yticks(np.arange(ego_speed_n))

    ax.set_xticklabels([round(elem, 2) for elem in ego_position_grid])
    ax.set_yticklabels([round(elem, 2) for elem in ego_speed_grid])
    #
    # # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    ax.set_title('Loss')
    fig.tight_layout()
    # plt.show()
    if not no_col:
        plt.savefig(fname + '.png')
    else:
        plt.savefig(fname + '_no_col.png')
    plt.close()


def get_heat_map(grid, fname):
    fig, ax = plt.subplots()
    sb.heatmap(grid, linewidth=0.5)

    ax.set_xticks(np.arange(ego_position_n))
    ax.set_yticks(np.arange(ego_speed_n))

    ax.set_xticklabels([round(elem, 2) for elem in ego_position_grid])
    ax.set_yticklabels([round(elem, 2) for elem in ego_speed_grid])
    #
    # # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    ax.set_title('Loss')
    fig.tight_layout()
    # plt.show()
    if not no_col:
        plt.savefig(fname + '.png')
    else:
        plt.savefig(fname + '_no_col.png')
    plt.close()

def compute(f1, f2, mode):
    f1_i = f1
    f2_i = f2
    f1 = dir + f1 +mode + '.pkl'
    f2 = dir + f2 +mode + '.pkl'
    bs1, bs2, bs3 = [], [], []
    et1, et2, et3 = [], [], []
    f1_r1, f1_r2, f2_r1, f2_r2, f3_r1, f3_r2 = [], [], [], [], [], []

    f1_df = pd.read_pickle(f1)
    f2_df = pd.read_pickle(f2)
    ss1 = f1_df['start_states']
    f1_p1_reward, f1_p2_reward = f1_df['p1_reward'], f1_df['p2_reward']
    f2_p1_reward, f2_p2_reward = f2_df['p1_reward'], f2_df['p2_reward']
    grid1 = np.full((100, 100), 0)

    for i in range(len(f1_p1_reward)):
        # print(ss1[i])
        if not no_col:
            if f1_p2_reward[i] < f2_p2_reward[i]:
                bs1.append(ss1[i])
                et1.append(i)
                f1_r1.append(f1_p2_reward[i])
                f1_r2.append(f2_p2_reward[i])

                position = ss1[i][0]
                speed = ss1[i][1]

                index_x = (position - ego_position_range[0]) * ego_position_n / (
                        ego_position_range[1] - ego_position_range[0])
                index_y = (speed - ego_speed_range[0]) * ego_speed_n / (ego_speed_range[1] -
                                                                        ego_speed_range[0])
                index_x = int(floor(index_x))
                index_y = int(floor(index_y))
                # print(ss1[i], index_x, index_y)
                grid1[index_y][index_x] = f1_p2_reward[i] - f2_p2_reward[i]
        else:
            if f1_p2_reward[i] < f2_p2_reward[i] and f1_p2_reward[i] > -1000:
            # if f1_p2_reward[i] > -1000 and f2_p2_reward[i] > -1000:
                bs1.append(ss1[i])
                et1.append(i)
                f1_r1.append(f1_p2_reward[i])
                f1_r2.append(f2_p2_reward[i])

                position = ss1[i][0]
                speed = ss1[i][1]

                index_x = (position - ego_position_range[0]) * ego_position_n / (
                        ego_position_range[1] - ego_position_range[0])
                index_y = (speed - ego_speed_range[0]) * ego_speed_n / (ego_speed_range[1] -
                                                                        ego_speed_range[0])
                index_x = int(floor(index_x))
                index_y = int(floor(index_y))
                # print(ss1[i], index_x, index_y)
                grid1[index_y][index_x] = f1_p2_reward[i] - f2_p2_reward[i]
                # print(ss1[i], f1_p2_reward[i], f2_p2_reward[i])

    print('Found: {}'.format(len(bs1)))
    draw_grid(f1, f2, grid1, mode)
    df1 = {'bad_states': bs1, 'episode_tag': et1, 'r1': f1_r1, 'r2': f1_r2}
    pandas_df = pd.DataFrame(df1)
    f1 = f1.split('/')
    f2 = f2.split('/')
    ft1 = f1[2].split('_')
    ft2 = f2[2].split('_')
    fname = f1[0] + '/' + f1[1] + '/' + ft1[0] + '_' + ft2[0] + '_' + mode
    if no_col:
        if not is_ne:
            pandas_df.to_pickle(fname + '.pkl')
        else:
            pandas_df.to_pickle(fname + '_ne.pkl')
    else:
        if not is_ne:
            pandas_df.to_pickle(fname + '_no_col.pkl')
        else:
            pandas_df.to_pickle(fname + '_ne_no_col.pkl')

    return grid1


def loss_grid(f, mode):
    f_ori = f
    f = dir+ f +mode + '.pkl'
    f1_df = pd.read_pickle(f)
    ss1 = f1_df['start_states']
    f1_p1_reward, f1_p2_reward = f1_df['p1_reward'], f1_df['p2_reward']
    grid1 = np.full((100, 100), 0)
    grid2 = np.full((100, 100), 0)
    for i in range(len(f1_p1_reward)):
        position = ss1[i][0]
        speed = ss1[i][1]

        index_x = (position - ego_position_range[0]) * ego_position_n / (
                ego_position_range[1] - ego_position_range[0])
        index_y = (speed - ego_speed_range[0]) * ego_speed_n / (ego_speed_range[1] -
                                                                ego_speed_range[0])
        index_x = int(floor(index_x))
        index_y = int(floor(index_y))
        # print(ss1[i], index_x, index_y)

        if not no_col:
            grid1[index_y][index_x] = f1_p2_reward[i]
            grid2[index_y][index_x] = f1_p1_reward[i]
        else:
            if f1_p2_reward[i] > -1000:
                grid1[index_y][index_x] = f1_p2_reward[i]
            if f1_p1_reward[i] > -1000:
                grid2[index_y][index_x] = f1_p1_reward[i]

        if f1_p2_reward[i] < -15:
            print(f1_p2_reward[i], ss1[i])


    get_heat_map(grid1, dir +  f_ori + mode + '_p2_loss')
    # get_heat_map(grid2, dir + f_ori+ mode + '_p1_loss')

if __name__ == '__main__':
    no_col = True

    ego_position_n = 100.  # number of grid points - 1
    ego_speed_n = 100.
    ego_position_range = [25.0, 45.0]
    ego_position_grid = np.arange(ego_position_range[0], ego_position_range[1],
                                  (ego_position_range[1] - ego_position_range[0]) / ego_position_n)
    # print(ego_position_grid)
    ego_speed_range = [5.0, 15.0]
    ego_speed_grid = np.arange(ego_speed_range[0], ego_speed_range[1],
                               (ego_speed_range[1] - ego_speed_range[0]) / ego_speed_n)

    is_ne = True
    log_level = logging.DEBUG  # logging.INFO
    logging.basicConfig(format='%(message)s', level=log_level)

    dir = './start_slice/'
    if not is_ne:
        f1 = 'online_'
        f2 = 'neutral_'
        f3 = 'g2_'
        f4 = 'mpc_2_beta_1_mode_'
    else:
        f1 = 'online_'
        f2 = 'neutral_ne_'
        f3 = 'g2_ne_'
        f4 = 'mpc_2_beta_1_mode_'

    modes = ['a1'] #, 'a2', 'b1']

    for mode in modes:
        bs1, bs2, bs3 = [], [], []
        et1, et2, et3 = [], [], []
        f1_r1, f1_r2, f2_r1, f2_r2, f3_r1, f3_r2 = [], [], [], [], [], []


        # loss_grid(f2, mode)
        # loss_grid(f3, mode)
        # loss_grid(f4, mode)
        compute(f2, f3, mode)
        compute(f3, f2, mode)
        compute(f2, f4, mode)
        compute(f4, f2, mode)


        if not is_ne:
            f1 = 'online_'
            f2 = 'neutral_'
            f3 = 'g2_'
            f4 = 'mpc_2_beta_1_mode_'
        else:
            f1 = 'online_'
            f2 = 'neutral_ne_'
            f3 = 'g2_ne_'
            f4 = 'mpc_2_beta_1_mode_'

