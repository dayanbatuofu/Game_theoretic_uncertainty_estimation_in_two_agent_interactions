import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

from HJI_Vehicle_costate.examples.choose_problem import system, problem

#____________________________________________________________________________________________________

sym = False
baselineAA = False
baselineNANA = True
baselineNAA = False
baselineANA = False
bvpAA = False
bvpNANA = False
bvpANA = False
bvpNAA = False
empAAbeliefNANA = False
empNANAbeliefAA = False
empAAbeliefAA = False
empNANAbeliefNANA = False
nonempAAbeliefNANA = False
nonempNANAbeliefAA = False
nonempAAbeliefAA = False
nonempNANAbeliefNANA = False

#________________________________________________________________________________________

if baselineAA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_baseline_a_a.mat')
    # data = scipy.io.loadmat('examples/' + system + '/time_nodes=100/data_discrete_verified_a_a.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_discrete_verified_a_a.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_a_a_NN.mat')
    index = 1
    title = 'Neural Network $\Theta^{*}=(a,a)$'
    special = 0
    theta1 = 1
    theta2 = 1
if baselineNANA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_baseline_na_na.mat')
    # data = scipy.io.loadmat('examples/' + system + '/time_nodes=100/data_discrete_verified_na_na.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_discrete_verified_na_na.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_na_na_NN.mat')
    index = 1
    title = 'Neural Network $\Theta^{*}=(na,na)$'
    special = 0
    theta1 = 5
    theta2 = 5
if baselineNAA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_baseline_na_a.mat')
    # data = scipy.io.loadmat('examples/' + system + '/time_nodes=100/data_discrete_verified_na_na.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_discrete_verified_na_na.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_na_na_NN.mat')
    index = 1
    title = 'Neural Network $\Theta^{*}=(na,a)$'
    special = 0
    theta1 = 5
    theta2 = 1
if baselineANA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_baseline_a_na.mat')
    # data = scipy.io.loadmat('examples/' + system + '/time_nodes=100/data_discrete_verified_na_na.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_discrete_verified_na_na.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_na_na_NN.mat')
    index = 1
    title = 'Neural Network $\Theta^{*}=(a,na)$'
    special = 0
    theta1 = 1
    theta2 = 5
if bvpAA is True:
    # data = scipy.io.loadmat('examples/' + system + '/data_BVP_a_a.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_discrete_val_a_a.mat')
    data = scipy.io.loadmat('examples/' + system + '/data_val_a_a.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_discrete_verified_a_a.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_discrete_verified_aa.mat')
    # data = scipy.io.loadmat('examples/' + system + '/time_nodes=100/data_discrete_test_a_a.mat')
    index = 2
    title = 'BVP $\Theta^{*}=(a,a)$'
    special = 0
    theta1 = 1
    theta2 = 1
if bvpNANA is True:
    # data = scipy.io.loadmat('examples/' + system + '/data_BVP_na_na.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_discrete_val_na_na.mat')
    data = scipy.io.loadmat('examples/' + system + '/data_val_na_na.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_discrete_verified_na_na.mat')
    # data = scipy.io.loadmat('examples/' + system + '/data_discrete_verified_nana.mat')
    # data = scipy.io.loadmat('examples/' + system + '/time_nodes=100/data_discrete_test_na_na.mat')
    index = 2
    title = 'BVP $\Theta^{*}=(na,na)$'
    special = 0
    theta1 = 5
    theta2 = 5
if bvpANA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_val_a_na.mat')
    index = 2
    title = 'BVP $\Theta^{*}=(a,na)$'
    special = 0
    theta1 = 1
    theta2 = 5
if bvpNAA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_val_na_a.mat')
    index = 2
    title = 'BVP $\Theta^{*}=(na,a)$'
    special = 0
    theta1 = 5
    theta2 = 1
if empAAbeliefAA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_E_a_a_belief_a_a.mat')
    index = 1
    title = '$(e,e), \Theta^{*}=(a,a), P_0=P_0^{a}$'
    special = 1
    theta1 = 1
    theta2 = 1
if empAAbeliefNANA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_E_a_a_belief_na_na.mat')
    index = 1
    title = '$(e,e), \Theta^{*}=(a,a), P_0=P_0^{na}$'
    special = 1
    theta1 = 1
    theta2 = 1
if empNANAbeliefAA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_E_na_na_belief_a_a.mat')
    index = 1
    title = '$(e,e), \Theta^{*}=(na,na), P_0=P_0^{a}$'
    special = 1
    theta1 = 5
    theta2 = 5
if empNANAbeliefNANA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_E_na_na_belief_na_na flipped.mat')
    index = 1
    title = '$(e,e), \Theta^{*}=(na,na), P_0=P_0^{na}$'
    special = 1
    theta1 = 5
    theta2 = 5
if nonempAAbeliefAA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_NE_a_a_belief_a_a.mat')
    index = 1
    title = '$(ne,ne), \Theta^{*}=(a,a), P_0=P_0^{a}$'
    special = 1
    theta1 = 1
    theta2 = 1
if nonempAAbeliefNANA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_NE_a_a_belief_na_na.mat')
    index = 1
    title = '$(ne,ne), \Theta^{*}=(a,a), P_0=P_0^{na}$'
    special = 1
    theta1 = 1
    theta2 = 1
if nonempNANAbeliefAA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_NE_na_na_belief_a_a.mat')
    index = 1
    title = '$(ne,ne), \Theta^{*}=(na,na), P_0=P_0^{a}$'
    special = 1
    theta1 = 5
    theta2 = 5
if nonempNANAbeliefNANA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_NE_na_na_belief_na_na.mat')
    index = 1
    title = '$(ne,ne), \Theta^{*}=(na,na), P_0=P_0^{na}$'
    special = 1
    theta1 = 5
    theta2 = 5

#____________________________________________________________________________________________________

font = {'family': 'normal', 'weight': 'normal', 'size': 16}

plt.rc('font', **font)

X = data['X']
# V = data['V']
t = data['t']

data.update({'t0': data['t']})
idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]

fig, axs = plt.subplots(1, 1, figsize=(6, 6))

for n in range(1, len(idx0) + 1):
# for n in range(1, 53):

    if sym is False:
        if n == len(idx0):
            x1 = X[0, idx0[n - 1]:]
            x2 = X[index, idx0[n - 1]:]
            # V1 = V[0, idx0[n - 1]:]
            # V2 = V[1, idx0[n - 1]:]
            # axs.plot(x1, x2)
            axs.plot(x1, x2, '-k')
        else:
            x1 = X[0, idx0[n - 1]: idx0[n]]
            x2 = X[index, idx0[n - 1]: idx0[n]]
            # V1 = V[0, idx0[n - 1]: idx0[n]]
            # V2 = V[1, idx0[n - 1]: idx0[n]]
            # axs.plot(x1, x2)
            axs.plot(x1, x2, '-k')

    if sym is True:
        x1 = X[0, idx0[n - 1]: idx0[n]]
        x2 = X[index, idx0[n - 1]: idx0[n]]
        # V1 = V[0, idx0[n - 1]: idx0[n]]
        # V2 = V[1, idx0[n - 1]: idx0[n]]
        if special == 1:
            # Plot 1
            axs.plot(x1, x2, '-k')
            # Flip Data
            tempx = x2
            x2 = x1
            x1 = tempx
            tempV = V2
            V2 = V1
            V1 = tempV
            # Plot 1
            axs.plot(x1, x2, '-k')
        else:
            if x2[0] > x1[0]:
                # Plot 1
                axs.plot(x1, x2, '-k')
                # Flip Data
                tempx = x2
                x2 = x1
                x1 = tempx
                tempV = V2
                V2 = V1
                V1 = tempV
                # Plot 1
                axs.plot(x1, x2, '-k')
            elif x2[0] - x1[0] > -0.2:
                axs.plot(x1, x2, '-k')
                # Flip Data
                tempx = x2
                x2 = x1
                x1 = tempx
                tempV = V2
                V2 = V1
                V1 = tempV
                # Plot 1
                axs.plot(x1, x2, '-k')

train1 = patches.Rectangle((35 - theta1 * 0.75, 35 - theta2 * 0.75), 3 + theta1 * 0.75 + 0.75,
                            3 + theta2 * 0.75 + 0.75, linewidth=1, edgecolor='k', facecolor='none')
start1 = patches.Rectangle((15, 15), 5, 5, linewidth=0.5, edgecolor='k', facecolor='none')
intersection1 = patches.Rectangle((34.25, 34.25), 4.5, 4.5, linewidth=1, edgecolor='grey', facecolor='grey')
axs.add_patch(intersection1)
axs.add_patch(train1)
axs.add_patch(start1)
axs.set_xlim(15, 40)
axs.set_xlabel('d1')
axs.set_ylim(15, 40)
axs.set_ylabel('d2')

p1 = [0, 35 - theta1 * 0.75]
p2 = [0, 35 - theta2 * 0.75]
# axs.plot(p1, p2, '-r')

axs.set_title(title)

plt.show()