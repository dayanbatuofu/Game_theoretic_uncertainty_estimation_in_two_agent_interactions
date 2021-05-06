import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

from examples.choose_problem import system, problem

sym = False
empAAbeliefNANA = False
empNANAbeliefAA = False

nonempAAbeliefNANA = True
nonempNANAbeliefAA = False

#________________________________________________________________________________________

if empAAbeliefNANA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_E_a_a_belief_na_na policy.mat')
    index = 1
    title = '$(e,e), \Theta^{*}=(1,1), P_0=P_0^{na}$'
    special = 1
    theta1 = 1
    theta2 = 1
    feel = 'e'
    true = 'a'
    belief = 'na'
if empNANAbeliefAA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_E_na_na_belief_a_a policy.mat')
    index = 1
    title = '$(e,e), \Theta^{*}=(5,5), P_0=P_0^{a}$'
    special = 1
    theta1 = 5
    theta2 = 5
    feel = 'e'
    true = 'na'
    belief = 'a'
if nonempAAbeliefNANA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_NE_a_a_belief_na_na policy.mat')
    index = 1
    title = '$(ne,ne), \Theta^{*}=(1,1), P_0=P_0^{na}$'
    special = 1
    theta1 = 1
    theta2 = 1
    feel = 'ne'
    true = 'a'
    belief = 'na'
if nonempNANAbeliefAA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_NE_na_na_belief_a_a policy.mat')
    index = 1
    title = '$(ne,ne), \Theta^{*}=(5,5), P_0=P_0^{a}$'
    special = 1
    theta1 = 5
    theta2 = 5
    feel = 'ne'
    true = 'na'
    belief = 'a'

#____________________________________________________________________________________________________

font = {'family' : 'normal','weight' : 'normal','size'   : 14}

plt.rc('font', **font)

X = data['X']
V = data['V']
T = data['t']

data.update({'t0': data['t']})
idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]


fig, axs = plt.subplots(1,2, figsize=(10,4))
norm1 = plt.Normalize(0, 1)
norm2 = plt.Normalize(0, 1)

if sym is True:
    norm3 = plt.Normalize(0, 1)
    norm4 = plt.Normalize(0, 1)

for n in range(1, len(idx0)):

    x1 = X[0, idx0[n-1]: idx0[n]]
    x2 = X[1, idx0[n-1]: idx0[n]]
    V1 = V[0, idx0[n-1]: idx0[n]]
    V2 = V[1, idx0[n-1]: idx0[n]]

    if sym is False:
        #Set Up Plot
        points = np.array([x1,x2]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize/Plot Value 1 ColorBar
        lc1 = LineCollection(segments,cmap='PuOr', norm=norm1)
        lc1.set_array(V1)
        lc1.set_linewidth(2)
        line1 = axs[0].add_collection(lc1)

        # Normalize/Plot Value 2 ColorBar
        lc2 = LineCollection(segments,cmap='PuOr', norm=norm2)
        lc2.set_array(V2)
        lc2.set_linewidth(2)
        line2 = axs[1].add_collection(lc2)

    if sym is True:

        # Set Up Plot
        points = np.array([x1,x2]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize/Plot Value 1 ColorBar
        lc1 = LineCollection(segments,cmap='PuOr', norm=norm1)
        lc1.set_array(V1)
        lc1.set_linewidth(2)
        line1 = axs[0].add_collection(lc1)

        # Normalize/Plot Value 2 ColorBar
        lc2 = LineCollection(segments,cmap='PuOr', norm=norm2)
        lc2.set_array(V2)
        lc2.set_linewidth(2)
        line2 = axs[1].add_collection(lc2)

        # Now Plot Mirror

        # Flip Data
        tempx = x2
        x2 = x1
        x1 = tempx
        tempV = V2
        V2 = V1
        V1 = tempV

        # Set Up Plot
        points = np.array([x1, x2]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize/Plot Value 1 ColorBar
        lc3 = LineCollection(segments, cmap='PuOr', norm=norm3)
        lc3.set_array(V1)
        lc3.set_linewidth(2)
        line3 = axs[0].add_collection(lc3)

        # Normalize/Plot Value 2 ColorBar
        lc4 = LineCollection(segments, cmap='PuOr', norm=norm4)
        lc4.set_array(V2)
        lc4.set_linewidth(2)
        line4 = axs[1].add_collection(lc4)

# Configure Plot
train1 = patches.Rectangle((35 - theta1*0.75,35 - theta2*0.75), 3+theta1*0.75+0.75, 3+theta2*0.75+0.75, linewidth=1, edgecolor = 'k', facecolor='none')
start1 = patches.Rectangle((15,15), 5, 5, linewidth=0.5, edgecolor='k', facecolor='none')
intersection1 = patches.Rectangle((34.25,34.25), 4.5, 4.5, linewidth=1, edgecolor = 'grey', facecolor='grey')
axs[0].add_patch(intersection1)
axs[0].add_patch(train1)
axs[0].add_patch(start1)
axs[0].set_xlim(15,40)
axs[0].set_xlabel('d1')
axs[0].set_ylim(15,40)
axs[0].set_ylabel('d2')
if sym is False:
    fig.colorbar(line1, ax=axs[0])
if sym is True:
    fig.colorbar(line3, ax=axs[0])
axs[0].set_title(f"$V_1$ ({feel}, {feel}), " + '$\Theta^{*}=$'+f"({true},{true}),"+' $P_0=P_0^{'+f"{belief}"+'}$')
train2 = patches.Rectangle((35 - theta1*0.75,35 - theta2*0.75), 3+theta1*0.75+0.75, 3+theta2*0.75+0.75, linewidth=1, edgecolor = 'k', facecolor='none')
start2 = patches.Rectangle((15,15), 5, 5, linewidth=0.5, edgecolor='k', facecolor='none')
intersection2 = patches.Rectangle((34.25,34.25), 4.5, 4.5, linewidth=1, edgecolor = 'grey', facecolor='grey')
axs[1].add_patch(intersection2)
axs[1].add_patch(train2)
axs[1].add_patch(start2)
axs[1].set_xlim(15,40)
axs[1].set_xlabel('d1')
axs[1].set_ylim(15,40)
axs[1].set_ylabel('d2')
if sym is False:
    fig.colorbar(line2, ax=axs[1])
if sym is True:
    fig.colorbar(line4, ax=axs[1])
axs[1].set_title(f"$V_2$ ({feel}, {feel}), " + '$\Theta^{*}=$'+f"({true},{true}),"+' $P_0=P_0^{'+f"{belief}"+'}$')

plt.show()