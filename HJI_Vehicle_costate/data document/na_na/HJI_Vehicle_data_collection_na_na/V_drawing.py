import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import io
from examples.choose_problem import system
from utilities.other import int_input

data_type = int_input('What kind of data for value do you want to plot? Enter 0 for validation, 1 for training:')
if data_type:
    train_data = scipy.io.loadmat('examples/' + system + '/data_train_update.mat')
    data_number = 89
else:
    train_data = scipy.io.loadmat('examples/' + system + '/data_val_update.mat')
    data_number = 26

train_data.update({'t0': train_data['t']})
idx0 = np.nonzero(np.equal(train_data.pop('t0'), 0.))[1]

t = train_data['t']
V = train_data['V']

length = len(idx0)

plt.figure(1)
for i in range(length):
    if i == data_number:  # 26 for validation data, 89 for train data
        plt.plot(t[0, idx0[i]:], V[0, idx0[i]:], label='V1')
    else:
        plt.plot(t[0, idx0[i]: idx0[i + 1]], V[0, idx0[i]: idx0[i + 1]], label='V1')

plt.xlabel("Time t")
plt.ylabel("V1")
plt.title('BVP $V_1$ vs Time $t$ $\Theta^{*}=(na,na)$')
plt.show()

plt.figure(2)
for i in range(length):
    if i == data_number:  # 26 for validation data, 89 for train data
        plt.plot(t[0, idx0[i]:], V[1, idx0[i]:], label='V2')
    else:
        plt.plot(t[0, idx0[i]: idx0[i + 1]], V[1, idx0[i]: idx0[i + 1]], label='V2')

plt.xlabel("Time t")
plt.ylabel("V2")
plt.title('BVP $V_2$ vs Time $t$ $\Theta^{*}=(na,na)$')
plt.show()
