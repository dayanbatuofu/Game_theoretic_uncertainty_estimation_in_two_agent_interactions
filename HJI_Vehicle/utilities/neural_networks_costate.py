# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:53:47 2020

@author: dell
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import time
import matplotlib.pyplot as plt
#from lbfgsnew import LBFGSNew
#from sdlbfgs import SdLBFGS
from LBFGS import FullBatchLBFGS
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FC_network(nn.Module):
    def __init__(self, layers, parameters):
        super(FC_network, self).__init__()
        self.feature1 = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.Tanh(),
            nn.Linear(layers[1], layers[2]),
            nn.Tanh(),
            nn.Linear(layers[2], layers[3]),
            nn.Tanh(),
            nn.Linear(layers[3], layers[4])
        ).to(device)
        # self.boundary = collision_upper
        # print('------successfully FC initialization--------')

        if parameters is None:
            # the first NN initialization (weight and bias)
            nn.init.xavier_normal_(self.feature1[0].weight)
            nn.init.zeros_(self.feature1[0].bias)
            nn.init.xavier_normal_(self.feature1[2].weight)
            nn.init.zeros_(self.feature1[2].bias)
            nn.init.xavier_normal_(self.feature1[4].weight)
            nn.init.zeros_(self.feature1[4].bias)
            nn.init.xavier_normal_(self.feature1[6].weight)
            nn.init.zeros_(self.feature1[6].bias)

        else:
            # the first NN initialization (weight and bias)
            self.feature1[0].weight = nn.Parameter(torch.tensor(parameters['weights'][0]).float())
            self.feature1[0].bias = nn.Parameter(torch.tensor(parameters['biases'][0].flatten()).float())
            self.feature1[2].weight = nn.Parameter(torch.tensor(parameters['weights'][1]).float())
            self.feature1[2].bias = nn.Parameter(torch.tensor(parameters['biases'][1].flatten()).float())
            self.feature1[4].weight = nn.Parameter(torch.tensor(parameters['weights'][2]).float())
            self.feature1[4].bias = nn.Parameter(torch.tensor(parameters['biases'][2].flatten()).float())
            self.feature1[6].weight = nn.Parameter(torch.tensor(parameters['weights'][3]).float())
            self.feature1[6].bias = nn.Parameter(torch.tensor(parameters['biases'][3].flatten()).float())

    def forward(self, x):
        x = self.feature1(x)
        return x

class HJI_network:
    def __init__(self, problem, scaling, config, parameters=None):
        '''Class implementing a NN for modeling time-dependent value functions.
        problem: instance of a problem class
        scaling: dictionary with 8 components:
            'lb' and 'ub',
                the lower and upper bounds of the input data, prior to scaling
            'A_lb' and 'A_ub',
                the lower and upper bounds of the gradient data, prior to scaling
            'U_lb' and 'U_ub',
                the lower and upper bounds of the control data, prior to scaling
            'V_min', and 'V_max',
                the lower and upper bounds of the output data, prior to scaling
        config: config_NN instance
        parameters: dict of weights and biases with pre-trained weights and biases'''

        self.lb = torch.tensor(scaling['lb'], requires_grad=True, dtype=torch.float32).to(device)
        self.ub = torch.tensor(scaling['ub'], requires_grad=True, dtype=torch.float32).to(device)
        self.A_lb = torch.tensor(scaling['A_lb'], requires_grad=True, dtype=torch.float32).to(device)
        self.A_ub = torch.tensor(scaling['A_ub'], requires_grad=True, dtype=torch.float32).to(device)

        self.problem = problem
        self.config = config
        self.layers = config.layers

        self.t1 = config.t1
        self.N_states = problem.N_states

        # Initializes the neural network
        self.FC_network = FC_network(config.layers, parameters)

        # load the NN

    def export_model(self):
        '''Returns a list of weights and biases to save model parameters.'''
        parm = {}

        for name, parameters in self.FC_network.named_parameters():
            parm[name] = parameters.detach().cpu().numpy()

        weights, biases = [], []

        for num in range(0, len(self.layers) + 2, 2):
            weights.append(parm['feature1.{}.weight'.format(num)])
            biases.append(parm['feature1.{}.bias'.format(num)].flatten())  # change the bias from 2-D to 1-D
        return weights, biases

    def make_eval_graph(self, t, X):
        '''Builds the NN computational graph.'''

        # (N_states, ?) matrix of linearly rescaled input values
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.
        X = torch.cat((X, 2.0 * t / self.t1 - 1.), dim=0)
        # It could keep the same dimension as the reference
        A = self.FC_network(X.T).T
        A_descaled = ((self.A_ub - self.A_lb) * (A + 1.) / 2. + self.A_lb)  # consider the scale range [-1,1]

        return A, A_descaled

    def train(self, train_data, val_data, EPISODE=1000, LR=0.1):  # updated
        '''Implements training with L-BFGS.'''
        train_data.update({
            'A_scaled': 2. * (train_data['A'] - self.A_lb.detach().cpu().numpy()) / (
                    self.A_ub.detach().cpu().numpy() - self.A_lb.detach().cpu().numpy()) - 1.})

        t_train = torch.tensor(train_data['t'], requires_grad=True, dtype=torch.float32).to(device)
        X_train = torch.tensor(train_data['X'], requires_grad=True, dtype=torch.float32).to(device)
        A_train = torch.tensor(train_data['A'], requires_grad=True, dtype=torch.float32).to(device)

        A_scaled_train = torch.tensor(train_data['A_scaled'], requires_grad=True, dtype=torch.float32).to(device)

        # ----------------------------------------------------------------------
        train_err = []
        val_err = []
        iternum = 0

        # Full batch
        # interpolate = True
        # max_ls = 1000
        #
        # optimizer = FullBatchLBFGS(self.FC_network.parameters(), lr=LR, history_size=10, debug=True)
        #
        # def current_loss(t_train, X_train, A_train, A_scaled_train):
        #     total_loss, _, _ = self.total_loss(t_train, X_train, A_train, A_scaled_train)
        #     optimizer.zero_grad()
        #
        #     total_loss.backward(retain_graph=True)
        #     return total_loss
        #
        # obj = current_loss(t_train, X_train, A_train, A_scaled_train)
        #
        # for _ in range(EPISODE):
        #     def closure():
        #         total_loss, MAE_train, _ = self.total_loss(t_train, X_train, A_train, A_scaled_train)
        #         optimizer.zero_grad()
        #         total_loss.backward(retain_graph=True)
        #         print(iternum, MAE_train.detach().cpu().numpy(), total_loss.detach().cpu().numpy(), end='\r')
        #         return total_loss
        #
        #     options = {'closure': closure, 'current_loss': obj, 'eta': 2, 'max_ls': max_ls, 'interpolate': interpolate,
        #                'inplace': False, 'damping': True}
        #
        #     optimizer.step(options)
        #     iternum += 1
        #
        # loss_A_train, MAE_train, A_train_descaled = self.total_loss(t_train, X_train, A_train, A_scaled_train)

        # Random single batch
        optimizer = torch.optim.Adam(self.FC_network.parameters(), lr=LR)

        for _ in range(EPISODE):
            batch_list = list(range(t_train.shape[1]))
            batch = random.sample(batch_list, len(batch_list))
            for i in batch:
                t_sample = t_train[0, i].reshape(-1, 1)
                X_sample = X_train[:, i].reshape(-1, 1)
                A_sample = A_train[:, i].reshape(-1, 1)
                A_scaled_sample = A_scaled_train[:, i].reshape(-1, 1)
                total_loss, MAE_train, _ = self.total_loss(t_sample, X_sample, A_sample, A_scaled_sample)

                optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                optimizer.step()

            iternum += 1
            total_loss, MAE_train, _ = self.total_loss(t_train, X_train, A_train, A_scaled_train)
            print(iternum, MAE_train.detach().cpu().numpy(), total_loss.detach().cpu().numpy(), end='\r')

        loss_A_train, MAE_train, A_train_descaled = self.total_loss(t_train, X_train, A_train, A_scaled_train)

        # batch_list = list(range(t_train.shape[1]))
        # batch = random.sample(batch_list, len(batch_list))
        #
        # for i in batch:
        #     t_sample = t_train[0, i].reshape(-1, 1)
        #     X_sample = X_train[:, i].reshape(-1, 1)
        #     A_sample = A_train[:, i].reshape(-1, 1)
        #     A_scaled_sample = A_scaled_train[:, i].reshape(-1, 1)
        #
        #     for _ in range(EPISODE):
        #         total_loss, MAE_train, _ = self.total_loss(t_sample, X_sample, A_sample, A_scaled_sample)
        #
        #         optimizer.zero_grad()
        #         total_loss.backward(retain_graph=True)
        #         optimizer.step()
        #
        #     total_loss, MAE_train, _ = self.total_loss(t_sample, X_sample, A_sample, A_scaled_sample)
        #     print(MAE_train.detach().cpu().numpy(), total_loss.detach().cpu().numpy())
        #
        #     iternum += 1
        #     total_loss, MAE_train, _ = self.total_loss(t_train, X_train, A_train, A_scaled_train)
        #     print(iternum, MAE_train.detach().cpu().numpy(), total_loss.detach().cpu().numpy())
        #     print('')
        #
        # loss_A_train, MAE_train, A_train_descaled = self.total_loss(t_train, X_train, A_train, A_scaled_train)

        train_err.append(MAE_train)

        print('')
        print('loss_A = %1.1e' % (loss_A_train))

        t_val = torch.tensor(val_data.pop('t'), requires_grad=True, dtype=torch.float32).to(device)
        X_val = torch.tensor(val_data.pop('X'), requires_grad=True, dtype=torch.float32).to(device)
        A_val = torch.tensor(val_data.pop('A'), requires_grad=True, dtype=torch.float32).to(device)

        A_val_scaled, A_val_descaled = self.make_eval_graph(t_val, X_val)
        # A_train_scaled, A_train_descaled = self.make_eval_graph(self.X_train)

        # Error metrics
        MAE_val = torch.mean(
            torch.abs(A_val_descaled - A_val).to(device)).to(device) / torch.mean(
            torch.abs(A_val).to(device)
        ).to(device)  # Costate

        val_err.append(MAE_val)

        print('')
        print('Training MAE error = %1.6e' % (train_err[-1]))
        print('Validation MAE error = %1.6e' % (val_err[-1]))

        errors = (np.array(train_err), np.array(val_err))

        return errors

    def total_loss(self, t_train, X_train, A_train, A_scaled_train):
        self.A_scaled, self.A_descaled = self.make_eval_graph(t_train, X_train)

        # Unweighted MSE loss on scaled data
        self.loss_A = torch.mean((self.A_scaled - A_scaled_train) ** 2).to(device)

        # loss calculation
        self.loss = self.loss_A

        self.MAE = torch.mean(
            torch.abs(self.A_descaled - A_train).to(device)).to(device) / torch.mean(
            torch.abs(A_train).to(device)
        ).to(device)  # Costate

        return self.loss, self.MAE, self.A_descaled

    def get_costate(self, X, t, theat1, theat2):
        X_NN = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        t_NN = torch.tensor(t, dtype=torch.float32, requires_grad=True)
        _, A_descaled = self.make_eval_graph(t_NN, X_NN)

        A = A_descaled.detach().cpu().numpy()
        lambda11_2 = A[0, :]
        lambda22_2 = A[9, :]

        U1 = lambda11_2 / 2
        U2 = lambda22_2 / 2

        return U1, U2

    def Hamilton_value(self, X, t, U, theta1, theta2):
        X_NN = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        t_NN = torch.tensor(t, dtype=torch.float32, requires_grad=True)
        _, A_descaled = self.make_eval_graph(t_NN, X_NN)

        A = A_descaled.detach().cpu().numpy()

        # costate calculation
        lambda11_current = A[0, :]
        lambda11_tc = A[1, :]
        lambda11_end = A[2, :]

        lambda12_current = A[3, :]
        lambda12_tc = A[4, :]
        lambda12_end = A[5, :]

        lambda21_current = A[6, :]
        lambda21_tc = A[7, :]
        lambda21_end = A[8, :]

        lambda22_current = A[9, :]
        lambda22_tc = A[10, :]
        lambda22_end = A[11, :]

        if lambda11_tc > t:
            lambda11_1 = -(lambda11_end - lambda11_current) / lambda11_tc
        if lambda11_tc <= t:
            lambda11_1 = self.problem.alpha

        if lambda12_tc > t:
            lambda12_1 = -(lambda12_end - lambda12_current) / lambda12_tc
        if lambda12_tc <= t:
            lambda12_1 = self.problem.alpha

        if lambda21_tc > t:
            lambda21_1 = -(lambda21_end - lambda21_current) / lambda21_tc
        if lambda21_tc <= t:
            lambda21_1 = self.problem.alpha

        if lambda22_tc > t:
            lambda22_1 = -(lambda22_end - lambda22_current) / lambda22_tc
        if lambda22_tc <= t:
            lambda22_1 = self.problem.alpha

        lambda11_2 = lambda11_current
        lambda12_2 = lambda12_current
        lambda21_2 = lambda21_current
        lambda22_2 = lambda22_current

        lambda11 = np.vstack((lambda11_1, lambda11_2))
        lambda12 = np.vstack((lambda12_1, lambda12_2))
        lambda21 = np.vstack((lambda21_1, lambda21_2))
        lambda22 = np.vstack((lambda22_1, lambda22_2))

        # control input U
        U1 = U[-2:-1, :]
        U2 = U[-1:, :]

        # U1 = lambda11_2 / 2
        # U2 = lambda22_2 / 2

        # max_acc = 10
        # min_acc = -5
        # U1[np.where(U1 > max_acc)] = max_acc
        # U1[np.where(U1 < min_acc)] = min_acc
        # U2[np.where(U2 > max_acc)] = max_acc
        # U2[np.where(U2 < min_acc)] = min_acc

        X1 = X[:self.problem.N_states]
        X2 = X[self.problem.N_states:2 * self.problem.N_states]

        x1 = torch.tensor(X1[0], requires_grad=True, dtype=torch.float32)  # including x1,v1
        x2 = torch.tensor(X2[0], requires_grad=True, dtype=torch.float32)  # including x2,v2

        x1_in = (x1 - self.problem.R1 / 2 + theta2 * self.problem.W2 / 2) * 10  # 3
        x1_out = -(x1 - self.problem.R1 / 2 - self.problem.W2 / 2 - self.problem.L1) * 10
        x2_in = (x2 - self.problem.R2 / 2 + theta1 * self.problem.W1 / 2) * 10
        x2_out = -(x2 - self.problem.R2 / 2 - self.problem.W1 / 2 - self.problem.L2) * 10

        Collision_F_x = self.problem.beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * \
                        torch.sigmoid(x2_in) * torch.sigmoid(x2_out)

        L1 = U1 ** 2 + Collision_F_x.detach().cpu().numpy()
        L2 = U2 ** 2 + Collision_F_x.detach().cpu().numpy()

        f1 = np.matmul(self.problem.A, X1) + np.matmul(self.problem.B, U1.reshape(1, -1))
        f2 = np.matmul(self.problem.A, X2) + np.matmul(self.problem.B, U2.reshape(1, -1))

        H1 = np.matmul(lambda11.T, f1) + np.matmul(lambda12.T, f2) - L1
        H2 = np.matmul(lambda21.T, f1) + np.matmul(lambda22.T, f2) - L2

        return H1, H2
