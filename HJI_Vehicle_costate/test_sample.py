# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:45:14 2020

@author: dell
"""
import pandas as pd
import scipy
from scipy import io
# from examples.choose_problem import system, problem, config
#
# load_path = save_path = 'examples/' + system + '/data_' + data_type + '.mat'

train_data = scipy.io.loadmat('data_train.mat')
train_data_X = train_data['X']
dfdata_X = pd.DataFrame(train_data_X)
datapath1 = './train_data_X.csv'
dfdata_X.to_csv(datapath1, index=False)

train_data_V = train_data['V']
dfdata_V = pd.DataFrame(train_data_V)
datapath2 = './train_data_V.csv'
dfdata_V.to_csv(datapath2, index=False)

train_data_t = train_data['t']
dfdata_t = pd.DataFrame(train_data_t)
datapath3 = './train_data_t.csv'
dfdata_t.to_csv(datapath3, index=False)

train_data_A = train_data['A']
dfdata_A = pd.DataFrame(train_data_A)
datapath4 = './train_data_A.csv'
dfdata_A.to_csv(datapath4, index=False)





