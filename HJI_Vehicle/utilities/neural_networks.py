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
from HJI_Vehicle.LBFGS import FullBatchLBFGS
from HJI_Vehicle.utilities.BVP_solver1 import solve_bvp1
from HJI_Vehicle.utilities.BVP_solver2 import solve_bvp2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FC_network(nn.Module):
	def __init__(self, layers, parameters, problem):
		super(FC_network, self).__init__()
		self.problem = problem
		self.feature1 = nn.Sequential(
			nn.Linear(layers[0], layers[1]),
			nn.Tanh(),
			nn.Linear(layers[1], layers[2]),
			nn.Tanh(),
			nn.Linear(layers[2], layers[3]),
			nn.Tanh(),
			nn.Linear(layers[3], layers[4])
		).to(device)

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
		x1 = self.feature1(x)
		# x2 = self.feature2(x)
		'''
		Key point to set the parameter in the self.Lambda. In below case, we consider the a-a pair, and the data set locates
		at the lower triangle, which means car 1 will pass the intersection firstly and car 2 follows. So we use 38.75m
		(upper boundary of collision area) for car 1, 34.25m(lower boundary of collision area) for car 2. If we consider
		the upper triangle of the data set from a-a pair, we will set 34.25 for car 1 and 38.75 for car 2.
		We can use trajectory.py to visualize trajectory of the data set.
		Below is the summary for upper and lower boundary of collision area of car 1 and 2:
		a-a case: car1 upper boundary: 38.75m, lower boundary: 34.25m; car2 upper boundary: 38.75m, lower boundary: 34.25m
		a-na case: car1 upper boundary: 38.75m, lower boundary: 34.25m; car2 upper boundary: 38.75m, lower boundary: 31.25m
		na-a case: car1 upper boundary: 38.75m, lower boundary: 31.25m; car2 upper boundary: 38.75m, lower boundary: 34.25m
		na-na case: car1 upper boundary: 38.75m, lower boundary: 31.25m; car2 upper boundary: 38.75m, lower boundary: 31.25m
		'''
		# if not self.problem.isUpper:
		# 	if self.problem.theta1 == 1 and self.problem.theta2 == 1:
		# 		L, U = 38.75, 34.25
		# 	elif self.problem.theta1 == 1 and self.problem.theta2 == 5:
		# 		L, U = 38.75, 31.25
		# 	elif self.problem.theta1 == 5 and self.problem.theta2 == 1:
		# 		L, U = 38.75, 34.25
		# 	else:
		# 		L, U = 38.75, 31.25
		# else:
		# 	if self.problem.theta1 == 1 and self.problem.theta2 == 1:
		# 		L, U = 34.25, 38.75
		# 	elif self.problem.theta1 == 1 and self.problem.theta2 == 5:
		# 		L, U = 34.25, 38.75
		# 	elif self.problem.theta1 == 5 and self.problem.theta2 == 1:
		# 		L, U = 31.25, 38.75
		# 	else:
		# 		L, U = 31.25, 38.75
		#
		# self.Lambda = torch.sigmoid(-(x[:, 0] - torch.tensor(L, dtype=torch.float32, requires_grad=True).to(device))) + \
		# 			  torch.sigmoid(-(x[:, 2] - torch.tensor(U, dtype=torch.float32, requires_grad=True).to(device)))
		# Lambda = torch.matmul(self.Lambda.reshape(-1, 1),
		# 					  torch.tensor(np.array([[1, 1]]), dtype=torch.float32, requires_grad=True).to(device)).to(device)
		# x = x1 * Lambda + x2 * (1 - Lambda)
		return x1

class HJB_network:
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
		self.U_lb = torch.tensor(scaling['U_lb'], requires_grad=True, dtype=torch.float32).to(device)
		self.U_ub = torch.tensor(scaling['U_ub'], requires_grad=True, dtype=torch.float32).to(device)
		self.V_min = torch.tensor(scaling['V_min'], requires_grad=True, dtype=torch.float32).to(device)
		self.V_max = torch.tensor(scaling['V_max'], requires_grad=True, dtype=torch.float32).to(device)

		self.problem = problem
		self.config = config
		self.layers = config.layers
		# print(self.layers)

		self.t1 = config.t1
		self.N_states = problem.N_states

		# Initializes the neural network
		self.FC_network = FC_network(config.layers, parameters, problem).to(device)

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
		V = self.FC_network(X.T).T
		V_descaled = ((self.V_max - self.V_min) * (V + 1.) / 2. + self.V_min)  # consider the scale range [-1,1]
		# print(V, V_descaled)
		return V, V_descaled

	def train(self, train_data, val_data, EPISODE=1000, LR=0.1):  # updated
		'''Implements training with L-BFGS.'''
		train_data.update({
			'A_scaled': 2. * (train_data['A'] - self.A_lb.detach().cpu().numpy()) / (
					self.A_ub.detach().cpu().numpy() - self.A_lb.detach().cpu().numpy()) - 1.,
			'V_scaled': 2. * (train_data['V'] - self.V_min.detach().cpu().numpy()) / (
					self.V_max.detach().cpu().numpy() - self.V_min.detach().cpu().numpy()) - 1.
		})

		self.Ns = self.config.batch_size
		if self.Ns is None:
			self.Ns = self.config.Ns['train']

		Ns_cand = self.config.Ns_cand
		Ns_max = self.config.Ns_max
		self.train_data_size = self.config.Ns['train']
		if self.Ns > self.train_data_size:
			new_data = self.generate_data(
				self.Ns - self.train_data_size, Ns_cand)
			for key in new_data.keys():
				train_data.update({
					key: np.hstack((train_data[key], new_data[key]))
				})

		self.Ns = np.minimum(self.Ns, Ns_max)

		self.t_train = torch.tensor(train_data['t'], requires_grad=True, dtype=torch.float32).to(device)
		self.X_train = torch.tensor(train_data['X'], requires_grad=True, dtype=torch.float32).to(device)
		self.A_train = torch.tensor(train_data['A'], requires_grad=True, dtype=torch.float32).to(device)
		self.V_train = torch.tensor(train_data['V'], requires_grad=True, dtype=torch.float32).to(device)

		self.A_scaled_train = torch.tensor(train_data['A_scaled'], requires_grad=True, dtype=torch.float32).to(device)
		self.V_scaled_train = torch.tensor(train_data['V_scaled'], requires_grad=True, dtype=torch.float32).to(device)

		self.weight_A = torch.tensor(self.config.weight_A, requires_grad=True, dtype=torch.float32).to(device)
		self.weight_U = torch.tensor(self.config.weight_U, requires_grad=True, dtype=torch.float32).to(device)

		# ----------------------------------------------------------------------
		train_err = []
		train_grad_err = []
		val_err = []
		val_grad_err = []
		iternum = 0

		interpolate = True
		max_ls = 1000

		optimizer = FullBatchLBFGS(self.FC_network.parameters(), lr=LR, history_size=10, debug=True)

		def current_loss():
			total_loss, _, _, _, _, _, _ = self.total_loss()
			optimizer.zero_grad()
			total_loss.backward(retain_graph=True)
			return total_loss

		obj = current_loss()

		for _ in range(EPISODE):
			def closure():
				total_loss, MAE_train, grad_MRL2_train, loss_V_train, loss_A_train, _, _ = self.total_loss()
				optimizer.zero_grad()
				total_loss.backward(retain_graph=True)
				print(iternum, MAE_train.detach().cpu().numpy(), grad_MRL2_train.detach().cpu().numpy(),
					  loss_V_train.detach().cpu().numpy(),
					  loss_A_train.detach().cpu().numpy(), end='\r')
				return total_loss

			options = {'closure': closure, 'current_loss': obj, 'eta': 2, 'max_ls': max_ls, 'interpolate': interpolate,
					   'inplace': False, 'damping': True}
			optimizer.step(options)
			iternum += 1

		_, MAE_train, grad_MRL2_train, loss_V_train, loss_A_train, V_train, dVdX_train = self.total_loss()

		train_err.append(MAE_train)
		train_grad_err.append(grad_MRL2_train)

		print('')
		print('loss_V = %1.1e' % (loss_V_train),
			  ', loss_A = %1.1e' % (loss_A_train))

		t_val = torch.tensor(val_data.pop('t'), requires_grad=True, dtype=torch.float32).to(device)
		X_val = torch.tensor(val_data.pop('X'), requires_grad=True, dtype=torch.float32).to(device)
		A_val = torch.tensor(val_data.pop('A'), requires_grad=True, dtype=torch.float32).to(device)
		V_val = torch.tensor(val_data.pop('V'), requires_grad=True, dtype=torch.float32).to(device)

		V_val_scaled, V_val_descaled = self.make_eval_graph(t_val, X_val)

		V_sum_1 = torch.sum(V_val_descaled[-2:-1])  # This is V1
		V_sum_1.requires_grad_()
		V_sum_2 = torch.sum(V_val_descaled[-1:])  # This is V2
		V_sum_2.requires_grad_()

		dVdX_1 = torch.autograd.grad(V_sum_1, X_val, create_graph=True)[0]  # [lambda11;lambda12]
		dVdX_2 = torch.autograd.grad(V_sum_2, X_val, create_graph=True)[0]  # [lambda21;lambda22]

		dVdX_val = torch.cat((dVdX_1, dVdX_2), 0)

		# Error metrics
		MAE_val = torch.mean(
			torch.abs(V_val_descaled - V_val)) / torch.mean(
			torch.abs(V_val)
		)  # Value
		grad_MRL2_val = torch.mean(
			torch.sqrt(torch.sum((dVdX_val - A_val) ** 2, dim=0))
		) / torch.mean(
			torch.sqrt(torch.sum(A_val ** 2, dim=0))
		)

		val_err.append(MAE_val)
		val_grad_err.append(grad_MRL2_val)

		print('')
		print('Training MAE error = %1.6e' % (train_err[-1]))
		print('Validation MAE error = %1.6e' % (val_err[-1]))
		print('Training grad. MRL2 error = %1.6e' % (train_grad_err[-1]))
		print('Validation grad. MRL2 error = %1.6e' % (val_grad_err[-1]))

		errors = (np.array(train_err), np.array(train_grad_err),
				  np.array(val_err), np.array(val_grad_err))

		# output V and dVdX after using NN
		model_data = dict()
		model_data.update({'t': train_data['t'],
						   'X': train_data['X'],
						   'V': V_train.detach().cpu().numpy(),
						   'A': dVdX_train.detach().cpu().numpy()})

		return errors, model_data

	def total_loss(self):
		self.V_scaled, self.V_descaled = self.make_eval_graph(self.t_train, self.X_train)

		V_sum_1 = torch.sum(self.V_descaled[-2:-1]).to(device)  # This is V1
		V_sum_1.requires_grad_()
		V_sum_2 = torch.sum(self.V_descaled[-1:]).to(device)  # This is V2
		V_sum_2.requires_grad_()

		self.dVdX_1 = (torch.autograd.grad(V_sum_1, self.X_train, create_graph=True)[0]).to(device)  # [lambda11;lambda12]
		self.dVdX_2 = (torch.autograd.grad(V_sum_2, self.X_train, create_graph=True)[0]).to(device)  # [lambda21;lambda22]

		self.dVdX = torch.cat((self.dVdX_1, self.dVdX_2), 0).to(device)

		# Unweighted MSE loss on scaled data
		self.loss_V = torch.mean((self.V_scaled - self.V_scaled_train) ** 2).to(device)

		# Unweighted MSE loss on value gradient
		dVdX_scaled = 2.0 * (self.dVdX - self.A_lb) / (self.A_ub - self.A_lb) - 1.
		self.loss_A = torch.mean(
			torch.sum((dVdX_scaled - self.A_scaled_train) ** 2, dim=0).to(device)
		).to(device)

		# loss calculation
		self.loss = self.loss_V
		self.loss = self.loss + self.weight_A * self.loss_A

		self.MAE = torch.mean(
			torch.abs(self.V_descaled - self.V_train).to(device)).to(device) / torch.mean(
			torch.abs(self.V_train).to(device)
		).to(device)  # Value

		self.grad_MRL2 = torch.mean(
			torch.sqrt(torch.sum((self.dVdX - self.A_train).to(device) ** 2, dim=0).to(device)).to(device)
		).to(device) / torch.mean(
			torch.sqrt(torch.sum(self.A_train ** 2, dim=0).to(device)).to(device)
		).to(device)

		return self.loss, self.MAE, self.grad_MRL2, self.loss_V, self.loss_A, self.V_descaled, self.dVdX

	def get_largest_A(self, t, X):
		'''Partially sorts space-time points by the predicted gradient norm.'''
		X = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
		t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)
		_, V_descaled = self.make_eval_graph(t, X)

		# It is equal to tf.gradients
		V_sum_1 = torch.sum(V_descaled[-2:-1])  # This is the V1
		V_sum_1.requires_grad_()
		V_sum_2 = torch.sum(V_descaled[-1:])  # This is the V2
		V_sum_2.requires_grad_()

		dVdX_1 = torch.autograd.grad(V_sum_1, X, create_graph=True)[0]  # [lamdba11;lambda12]
		dVdX_2 = torch.autograd.grad(V_sum_2, X, create_graph=True)[0]  # [lamdba21;lambda22]

		dVdX_norm_11 = torch.sqrt(torch.sum(dVdX_1[:self.N_states] ** 2, dim=0))
		max_idx_11 = torch.argmax(dVdX_norm_11)
		dVdX_norm_12 = torch.sqrt(torch.sum(dVdX_1[self.N_states:2 * self.N_states] ** 2, dim=0))
		max_idx_12 = torch.argmax(dVdX_norm_12)
		dVdX_norm_21 = torch.sqrt(torch.sum(dVdX_2[:self.N_states] ** 2, dim=0))
		max_idx_21 = torch.argmax(dVdX_norm_21)
		dVdX_norm_22 = torch.sqrt(torch.sum(dVdX_2[self.N_states:2 * self.N_states] ** 2, dim=0))
		max_idx_22 = torch.argmax(dVdX_norm_22)
		max_idx_X1 = max((max_idx_11, max_idx_21))
		max_idx_X2 = max((max_idx_12, max_idx_22))

		return max_idx_X1, max_idx_X2

	# trained model bvp guess
	def bvp_guess(self, t, X, eval_U=False):
		'''Predicts value, costate, and control with one session call.'''
		X = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
		t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)

		if eval_U:
			_, V_descaled = self.make_eval_graph(t, X)

			V_sum_1 = torch.sum(V_descaled[-2:-1])  # This is the V1
			V_sum_1.requires_grad_()
			V_sum_2 = torch.sum(V_descaled[-1:])  # This is the V2
			V_sum_2.requires_grad_()

			dVdX_1 = torch.autograd.grad(V_sum_1, X, create_graph=True)[0]  # [lamdba11;lambda12]
			dVdX_2 = torch.autograd.grad(V_sum_2, X, create_graph=True)[0]  # [lamdba21;lambda22]

			dVdX = torch.cat((dVdX_1, dVdX_2), 0)

			U1 = self.problem.make_U_NN_1(dVdX)
			U2 = self.problem.make_U_NN_2(dVdX)
			U = torch.cat((U1, U2), 0)

			return V_descaled, dVdX, U
		else:
			_, V_descaled = self.make_eval_graph(t, X)

			V_sum_1 = torch.sum(V_descaled[-2:-1])  # This is the V1
			V_sum_1.requires_grad_()
			V_sum_2 = torch.sum(V_descaled[-1:])  # This is the V2
			V_sum_2.requires_grad_()

			dVdX_1 = torch.autograd.grad(V_sum_1, X, create_graph=True)[0]  # [lamdba11;lambda12]
			dVdX_2 = torch.autograd.grad(V_sum_2, X, create_graph=True)[0]  # [lamdba21;lambda22]

			dVdX = torch.cat((dVdX_1, dVdX_2), 0)

			return V_descaled, dVdX

	def eval_U(self, t, X):
		'''(Near-)optimal feedback control for arbitrary inputs (t,X).'''
		X = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
		t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)
		_, V_descaled = self.make_eval_graph(t, X)

		# It is equal to tf.gradients
		V_sum_1 = torch.sum(V_descaled[-2:-1])  # This is the V1
		V_sum_1.requires_grad_()
		V_sum_2 = torch.sum(V_descaled[-1:])  # This is the V2
		V_sum_2.requires_grad_()

		dVdX_1 = torch.autograd.grad(V_sum_1, X, create_graph=True)[0]  # [lamdba11;lambda12]
		dVdX_2 = torch.autograd.grad(V_sum_2, X, create_graph=True)[0]  # [lamdba21;lambda22]

		dVdX = torch.cat((dVdX_1, dVdX_2), 0)

		U_1 = self.problem.make_U_NN_1(dVdX) #.detach().cpu().numpy()
		U_2 = self.problem.make_U_NN_2(dVdX) #.detach().cpu().numpy()
		U = np.vstack((U_1, U_2))

		return U

	def generate_data(self, Nd, Ns_cand):
		'''Generates additional data with NN warm start.'''
		print('')
		print('Generating data...')

		import warnings
		np.seterr(over='warn', divide='warn', invalid='warn')
		warnings.filterwarnings('error')

		N_states = self.problem.N_states

		t_OUT = np.empty((1, 0))
		X_OUT = np.empty((2 * N_states, 0))
		A_OUT = np.empty((4 * N_states, 0))
		V_OUT = np.empty((2, 0))

		Ns_sol = 0
		start_time = time.time()

		step = 0
		# ----------------------------------------------------------------------
		while Ns_sol < Nd:
			# Latin hypercube sampling
			# ToDO: too much space
			X0 = self.problem.sample_X0(Ns_cand)

			# bounds = np.hstack((self.lb.detach().cpu().numpy(), self.ub.detach().cpu().numpy()))
			# D = bounds.shape[0]
			# samples = self.problem.LHSample(D, bounds, Ns_cand)
			# X0 = np.array(samples).T

			# Picks random sample with largest gradient
			max_idx_X1, max_idx_X2 = self.get_largest_A(np.zeros((1, Ns_cand)), X0)
			X0_1 = X0[:2, max_idx_X1.detach().cpu().numpy()]
			X0_2 = X0[2:, max_idx_X2.detach().cpu().numpy()]
			X0 = np.concatenate((X0_1, X0_2))

			print(X0)

			bc = self.problem.make_bc(X0)

			# Integrates the closed-loop system (NN controller)
			SOL = solve_ivp(self.problem.dynamics, [0., self.t1], X0,
							method=self.config.ODE_solver,
							args=(self.eval_U,),
							rtol=1e-04)

			V_guess, A_guess = self.bvp_guess(SOL.t.reshape(1, -1), SOL.y)

			step += 1
			print(step)

			X_aug_guess = np.vstack((SOL.y, A_guess.detach().cpu().numpy(), V_guess.detach().cpu().numpy()))
			t_aug_guess = SOL.t

			SOL = solve_bvp1(self.problem.aug_dynamics, bc, t_aug_guess, X_aug_guess,
							 verbose=2, tol=5e-3, max_nodes=2500)

			t_M1 = SOL.x
			X_M1 = SOL.y[:2 * N_states]
			A_M1 = SOL.y[2 * N_states:6 * N_states]
			V1_M1 = -SOL.y[-2:-1]
			V2_M1 = -SOL.y[-1:]
			V_M1 = np.vstack((V1_M1, V2_M1))

			'''
			Second BVP Solver, the trajectory will have the interaction with the boundary line
			'''
			SOL = solve_bvp2(self.problem.aug_dynamics, bc, t_aug_guess, X_aug_guess,
							 verbose=2, tol=5e-3, max_nodes=2500)

			t_M2 = SOL.x
			X_M2 = SOL.y[:2 * N_states]
			A_M2 = SOL.y[2 * N_states:6 * N_states]
			V1_M2 = -SOL.y[-2:-1]
			V2_M2 = -SOL.y[-1:]
			V_M2 = np.vstack((V1_M2, V2_M2))

			if V1_M1[0, 0:1] + V2_M1[0, 0:1] >= V1_M2[0, 0:1] + V2_M2[0, 0:1]:
				t_OUT = np.hstack((t_OUT, t_M1.reshape(1, -1)))
				X_OUT = np.hstack((X_OUT, X_M1))
				A_OUT = np.hstack((A_OUT, A_M1))
				V_OUT = np.hstack((V_OUT, V_M1))
				print('Choose option1')

			if V1_M1[0, 0:1] + V2_M1[0, 0:1] < V1_M2[0, 0:1] + V2_M2[0, 0:1]:
				t_OUT = np.hstack((t_OUT, t_M2.reshape(1, -1)))
				X_OUT = np.hstack((X_OUT, X_M2))
				A_OUT = np.hstack((A_OUT, A_M2))
				V_OUT = np.hstack((V_OUT, V_M2))
				print('Choose option2')

			Ns_sol += 1

			print('----calculation end-------')

		print('Generated', X_OUT.shape[1], 'data from', Ns_sol,
			  'BVP solutions in %.1f' % (time.time() - start_time), 'sec')

		data = {'t':t_OUT, 'X': X_OUT, 'A': A_OUT, 'V': V_OUT}

		data.update({
			'A_scaled': 2. * (data['A'] - self.A_lb.detach().cpu().numpy()) / (
					self.A_ub.detach().cpu().numpy() - self.A_lb.detach().cpu().numpy()) - 1.,
			'V_scaled': 2. * (data['V'] - self.V_min.detach().cpu().numpy()) / (
					self.V_max.detach().cpu().numpy() - self.V_min.detach().cpu().numpy()) - 1.
		})

		return data

	# def Q_value(self, X, t):
	# 	X_NN = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
	# 	t_NN = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)
	# 	_, self.V_descaled = self.make_eval_graph(t_NN, X_NN)
	#
	# 	V_sum_1 = torch.sum(self.V_descaled[-2:-1])  # This is V1
	# 	V_sum_1.requires_grad_()
	# 	V_sum_2 = torch.sum(self.V_descaled[-1:])  # This is V2
	# 	V_sum_2.requires_grad_()
	#
	# 	dVdX_1 = (torch.autograd.grad(V_sum_1, X_NN, create_graph=True)[0])
	# 	dVdX_2 = (torch.autograd.grad(V_sum_2, X_NN, create_graph=True)[0])
	#
	# 	# V_curr = V_descaled.detach().cpu().numpy()
	#
	# 	# V1 = V_descaled[-2:-1].detach().cpu().numpy()
	# 	# V2 = V_descaled[-1:].detach().cpu().numpy()
	# 	#
	# 	# delta = 0.01
	# 	#
	# 	# X_x1 = torch.tensor(X + np.array([[delta], [0.], [0.], [0.]]), dtype=torch.float32, requires_grad=True)
	# 	# _, V_descaled = self.make_eval_graph(t_NN, X_x1)
	# 	# V1_x1 = V_descaled[-2:-1].detach().cpu().numpy()
	# 	# V2_x1 = V_descaled[-1:].detach().cpu().numpy()
	# 	#
	# 	# X_v1 = torch.tensor(X + np.array([[0.], [delta], [0.], [0.]]), dtype=torch.float32, requires_grad=True)
	# 	# _, V_descaled = self.make_eval_graph(t_NN, X_v1)
	# 	# V1_v1 = V_descaled[-2:-1].detach().cpu().numpy()
	# 	# V2_v1 = V_descaled[-1:].detach().cpu().numpy()
	# 	#
	# 	# X_x2 = torch.tensor(X + np.array([[0.], [0.], [delta], [0.]]), dtype=torch.float32, requires_grad=True)
	# 	# _, V_descaled = self.make_eval_graph(t_NN, X_x2)
	# 	# V1_x2 = V_descaled[-2:-1].detach().cpu().numpy()
	# 	# V2_x2 = V_descaled[-1:].detach().cpu().numpy()
	# 	#
	# 	# X_v2 = torch.tensor(X + np.array([[0.], [0.], [0.], [delta]]), dtype=torch.float32, requires_grad=True)
	# 	# _, V_descaled = self.make_eval_graph(t_NN, X_v2)
	# 	# V1_v2 = V_descaled[-2:-1].detach().cpu().numpy()
	# 	# V2_v2 = V_descaled[-1:].detach().cpu().numpy()
	# 	#
	# 	# dVdX1 = np.vstack(((V1_x1 - V1) / delta, (V1_v1 - V1) / delta, (V1_x2 - V1) / delta, (V1_v2 - V1) / delta))
	# 	# dVdX2 = np.vstack(((V2_x1 - V2) / delta, (V2_v1 - V2) / delta, (V2_x2 - V2) / delta, (V2_v2 - V2) / delta))
	# 	#
	# 	# A11 = dVdX1[:self.problem.N_states]
	# 	# A12 = dVdX1[self.problem.N_states:2 * self.problem.N_states]
	# 	# A21 = dVdX2[:self.problem.N_states]
	# 	# A22 = dVdX2[self.problem.N_states:2 * self.problem.N_states]
	# 	#
	# 	# X1 = X[:self.problem.N_states]
	# 	# X2 = X[self.problem.N_states:2 * self.problem.N_states]
	#
	# 	# x1 = torch.tensor(X1[0], requires_grad=True, dtype=torch.float32).to(device)  # including x1,v1
	# 	# x2 = torch.tensor(X2[0], requires_grad=True, dtype=torch.float32).to(device)  # including x2,v2
	# 	#
	# 	# x1_in = (x1 - self.problem.R1 / 2 + theta2 * self.problem.W2 / 2) * 10  # 3
	# 	# x1_out = -(x1 - self.problem.R1 / 2 - self.problem.W2 / 2 - self.problem.L1) * 10
	# 	# x2_in = (x2 - self.problem.R2 / 2 + theta1 * self.problem.W1 / 2) * 10
	# 	# x2_out = -(x2 - self.problem.R2 / 2 - self.problem.W1 / 2 - self.problem.L2) * 10
	# 	#
	# 	# Collision_F_x = self.problem.beta * torch.sigmoid(x1_in).to(device) * torch.sigmoid(x1_out).to(device) * \
	# 	# 				torch.sigmoid(x2_in).to(device) * torch.sigmoid(x2_out).to(device)
	# 	#
	# 	# U1 = U[-2:-1, :]
	# 	# U2 = U[-1:, :]
	# 	#
	# 	# L1 = U1 ** 2 + Collision_F_x.detach().cpu().numpy()
	# 	# L2 = U2 ** 2 + Collision_F_x.detach().cpu().numpy()
	# 	#
	# 	# dt = 0.5
	# 	#
	# 	# V1 = V_curr[-2:-1] - L1 * dt
	# 	# V2 = V_curr[-1:] - L2 * dt
	#
	# 	X1 = X[:self.problem.N_states]
	# 	X2 = X[self.problem.N_states:2 * self.problem.N_states]
	#
	# 	U1 = self.problem.make_U_NN_1(dVdX_1) #.detach().cpu().numpy()
	# 	U2 = self.problem.make_U_NN_2(dVdX_2) #.detach().cpu().numpy()
	#
	# 	dXdt1 = np.matmul(self.problem.A, X1) + np.matmul(self.problem.B, U1)
	# 	dXdt2 = np.matmul(self.problem.A, X2) + np.matmul(self.problem.B, U2)
	#
	# 	V1 = self.V_descaled[-2:-1].detach().cpu().numpy()
	# 	V2 = self.V_descaled[-1:].detach().cpu().numpy()
	#
	# 	Q1 = V1 - np.matmul(dVdX_1[:self.problem.N_states].detach().cpu().numpy().reshape(1, -1), dXdt1) * t  # How could I use this formula?
	# 	Q2 = V2 - np.matmul(dVdX_2[self.problem.N_states:2 * self.problem.N_states].detach().cpu().numpy().reshape(1, -1) , dXdt2) * t
	#
	# 	return Q1, Q2

	def Q_value(self, X, t, U, theta1, theta2, deltaT):
		X_NN = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
		t_NN = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)
		_, V_descaled = self.make_eval_graph(t_NN, X_NN)

		V_curr = V_descaled.detach().cpu().numpy()

		X1 = X[:self.problem.N_states]
		X2 = X[self.problem.N_states:2 * self.problem.N_states]

		x1 = torch.tensor(X1[0], requires_grad=True, dtype=torch.float32).to(device)  # including x1,v1
		x2 = torch.tensor(X2[0], requires_grad=True, dtype=torch.float32).to(device) # including x2,v2

		x1_in = (x1 - self.problem.R1 / 2 + theta2 * self.problem.W2 / 2) * 10  # 3
		x1_out = -(x1 - self.problem.R1 / 2 - self.problem.W2 / 2 - self.problem.L1) * 10
		x2_in = (x2 - self.problem.R2 / 2 + theta1 * self.problem.W1 / 2) * 10
		x2_out = -(x2 - self.problem.R2 / 2 - self.problem.W1 / 2 - self.problem.L2) * 10

		Collision_F_x = self.problem.beta * torch.sigmoid(x1_in).to(device) * torch.sigmoid(x1_out).to(device) * \
						torch.sigmoid(x2_in).to(device) * torch.sigmoid(x2_out).to(device)

		U1 = U[-2:-1, :]
		U2 = U[-1:, :]

		L1 = U1 ** 2 + Collision_F_x.detach().cpu().numpy()
		L2 = U2 ** 2 + Collision_F_x.detach().cpu().numpy()

		dt = 0.5

		V1 = V_curr[-2:-1] - L1 * deltaT
		V2 = V_curr[-1:] - L2 * deltaT

		return V1, V2