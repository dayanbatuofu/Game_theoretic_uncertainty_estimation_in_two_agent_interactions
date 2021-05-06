'''
This file is to generate the data only including the initial information for different trajectory
'''
import numpy as np
import scipy
from scipy import io
from examples.choose_problem import system

train_data = scipy.io.loadmat('examples/' + system + '/data_train_100.mat')
# train_data = scipy.io.loadmat('examples/' + system + '/data_val_100.mat')

train_data.update({'t0': train_data['t']})

for data in [train_data]:
    idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]
    data.update({'X': data['X'][:, idx0],
                 'A': data['A'][:, idx0],
                 'V': data['V'][:, idx0],
                 't': data['t'][:, idx0]})

x1 = train_data['X'][0, :]
v1 = train_data['X'][1, :]
x2 = train_data['X'][2, :]
v2 = train_data['X'][3, :]

V1 = train_data['V'][0, :]
V2 = train_data['V'][1, :]

save_path = 'examples/' + system + '/data_train_new_100.mat'
# save_path = 'examples/' + system + '/data_val_new_100.mat'
scipy.io.savemat(save_path, train_data)



