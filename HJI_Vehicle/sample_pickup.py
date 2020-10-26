import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from examples.choose_problem import system, problem

train_data = scipy.io.loadmat('examples/' + system + '/data_train.mat')

X = train_data['X']
V = train_data['V']
T = train_data['t']

train_data.update({'t0': train_data['t']})
idx0 = np.nonzero(np.equal(train_data.pop('t0'), 0.))[1]

x1 = X[0, idx0[0]: idx0[1]]
x2 = X[2, idx0[0]: idx0[1]]
V1 = V[0, idx0[0]: idx0[1]]
V2 = V[1, idx0[0]: idx0[1]]
t = T[0, idx0[0]: idx0[1]]

fig, ax = plt.subplots()
# ax = fig.gca()
ax.plot(x1, x2, label='trajectory for car1 and car2')
rect = patches.Rectangle((31.25, 31.25), 7.5, 7.5, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
ax.set_xlabel('Trajectory x1')
ax.set_ylabel('Trajectory x2')
ax.xaxis.set_ticks(np.arange(0, 50, 2))
ax.yaxis.set_ticks(np.arange(0, 50, 2))
ax.title.set_text('Vehicle position variation')
ax.legend(loc='upper left')
ax.axis('equal')
plt.show()








