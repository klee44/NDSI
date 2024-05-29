import os
import logging
import pickle

import torch
import torch.nn as nn
import numpy as np
import math 
import glob
import re
import itertools
import toolz
import scipy
from shutil import copyfile
from collections import Counter

def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)

def init_network_weights(net, std = 0.1):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean=0, std=std)
			nn.init.constant_(m.bias, val=0)

def init_network_weights_xavier_normal(net):
	for m in net.modules():
		if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			nn.init.constant_(m.bias, val=0)

def init_network_weights_orthogonal(net):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.orthogonal_(m.weight)
			#nn.init.constant_(m.bias, val=0)

def init_network_weights_zero(net):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.zeros_(m.weight)
			nn.init.zeros_(m.bias)


def create_net(n_inputs, n_outputs, n_layers = 1, 
	n_units = 100, nonlinear = nn.Tanh):
	if n_layers == 0:
		layers = [nn.Linear(n_inputs, n_outputs)]
	else:
		layers = [nn.Linear(n_inputs, n_units)]
		for i in range(n_layers-1):
			layers.append(nonlinear())
			layers.append(nn.Linear(n_units, n_units))

		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)

def get_batch(data, t, batch_len=60, batch_size=100, device = torch.device("cpu"), reverse=False):
	r = torch.from_numpy(np.random.choice(np.arange(len(data),dtype=np.int64),batch_size, replace=False))
	s = torch.from_numpy(np.random.choice(np.arange(len(t) - batch_len, dtype=np.int64), batch_size, replace=False))
	batch_y0 = data[r,s,:]  # (M, D)
	batch_t = t[:batch_len]  # (T)
	batch_y = torch.stack([data[r,s + i,:] for i in range(batch_len)], dim=1)  # (T, M, D)
	if reverse: 
		batch_y0 = batch_y[:,-1,:]
		batch_t = batch_t.flip([0])
		batch_y = batch_y.flip([1])
	return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

def get_batch_two(data, t, batch_len=60, batch_size=100, device = torch.device("cpu"), reverse=False):
	r = torch.from_numpy(np.random.choice(np.arange(len(data),dtype=np.int64),batch_size, replace=False))
	s = torch.from_numpy(np.random.choice(np.arange(len(t) - batch_len, dtype=np.int64), batch_size, replace=False))
	batch_y0 = data[r,s,:]  # (M, D)
	batch_t = t[:batch_len]  # (T)
	batch_y = torch.stack([data[r,s + i,:] for i in range(batch_len)], dim=1)  # (T, M, D)
	
	batch_y0_backward = batch_y[:,-1,:]
	batch_t_backward = batch_t.flip([0])
	batch_y_backward = batch_y.flip([1])
	return batch_y0.to(device), batch_t.to(device), batch_y.to(device), batch_y0_backward.to(device), batch_t_backward.to(device), batch_y_backward.to(device)

def get_batch_two_single(data, t, batch_len=60, batch_size=100, device = torch.device("cpu"), reverse=False):
	s = torch.from_numpy(np.random.choice(np.arange(len(t) - batch_len, dtype=np.int64), batch_size, replace=False))
	batch_y0 = data[0,s,:]  # (M, D)
	batch_t = t[:batch_len]  # (T)
	batch_y = torch.stack([data[0,s + i,:] for i in range(batch_len)], dim=1)  # (T, M, D)
	
	batch_y0_backward = batch_y[:,-1,:]
	batch_t_backward = batch_t.flip([0])
	batch_y_backward = batch_y.flip([1])
	return batch_y0.to(device), batch_t.to(device), batch_y.to(device), batch_y0_backward.to(device), batch_t_backward.to(device), batch_y_backward.to(device)

def get_batch_two_single_time(data, t, batch_len=60, batch_size=100, device = torch.device("cpu"), reverse=False):
	s = torch.from_numpy(np.random.choice(np.arange(len(t) - batch_len, dtype=np.int64), batch_size, replace=False))
	batch_y0 = data[0,s,:]  # (M, D)
	batch_t = t[s:s+batch_len]  # (T)
	batch_y = torch.stack([data[0,s + i,:] for i in range(batch_len)], dim=1)  # (T, M, D)
	
	batch_y0_backward = batch_y[:,-1,:]
	batch_t_backward = batch_t.flip([0])
	batch_y_backward = batch_y.flip([1])
	return batch_y0.to(device), batch_t.to(device), batch_y.to(device), batch_y0_backward.to(device), batch_t_backward.to(device), batch_y_backward.to(device)

def get_batch_t(data, t, batch_len=60, batch_size=100, device = torch.device("cpu")):
	r = torch.from_numpy(np.random.choice(np.arange(len(data),dtype=np.int64),batch_size, replace=False))
	s = torch.from_numpy(np.random.choice(np.arange(len(t) - batch_len, dtype=np.int64), batch_size, replace=False))
	batch_y0 = data[r,s,:]  # (M, D)
	batch_t = t[:batch_len]  # (T)
	batch_t_ = torch.unsqueeze(torch.stack([t[s+i] for i in range(batch_len)], dim=1),-1)
	batch_y = torch.stack([data[r,s + i,:] for i in range(batch_len)], dim=1)  # (T, M, D)
	batch_y = torch.cat((batch_y,batch_t_),axis=-1)
	return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

def get_batch_traj(data, t, batch_size=100, device = torch.device("cpu")):
	r = torch.from_numpy(np.random.choice(np.arange(len(data),dtype=np.int64),batch_size, replace=False))
	batch_y0 = data[r,0,:]  # (M, D)
	batch_t = t[:]  # (T)
	batch_y = data[r,:,:]#torch.stack([data[r,s + i,:] for i in range(batch_len)], dim=1)  # (T, M, D)
	return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

class TensorProduct(nn.Module):
	def __init__(self, dim, order):
		super(TensorProduct, self).__init__()
		self.dim = dim
		self.indc = list(itertools.product(*[range(order+1) for _ in range(dim)]))
		self.nterms = len(self.indc)
		print(self.indc)

	def forward(self,x):
		ret = torch.stack([torch.prod(torch.stack([x[...,d]**ind[d] for d in range(self.dim)]),0) for ind in self.indc],-1)
		return ret

class TotalDegree(nn.Module):
	def __init__(self, dim, order):
		super(TotalDegree, self).__init__()
		self.dim = dim
		self.indc = Counter(map(toolz.compose(tuple,sorted),itertools.chain(*[itertools.product(*[range(dim) for _ in range(o)]) for o in range(order+1)])
                  ))
		print(sorted(self.indc))
		self.nterms = len(self.indc)

	def forward(self,x):
		ret = torch.cat([torch.unsqueeze(torch.prod(torch.stack([x[...,d]**ind.count(d) for d in range(self.dim)]),0),-1)  for ind in sorted(self.indc)],-1)
		return ret 

class TotalDegreeTrig(nn.Module):
	def __init__(self, dim, order):
		super(TotalDegreeTrig, self).__init__()
		self.dim = dim
		self.indc = Counter(map(toolz.compose(tuple,sorted),itertools.chain(*[itertools.product(*[range(dim) for _ in range(o)]) for o in range(order+1)])
                  ))
		print(sorted(self.indc))
		self.nterms = len(self.indc)+4

	def forward(self,x):
		ret = torch.cat([torch.unsqueeze(torch.prod(torch.stack([x[...,d]**ind.count(d) for d in range(self.dim)]),0),-1)  for ind in sorted(self.indc)],-1)
		ret = torch.cat((ret, torch.unsqueeze(torch.cos(x[:,0]),-1), torch.unsqueeze(torch.sin(x[:,0]),-1), torch.unsqueeze(torch.cos(x[:,1]),-1), torch.unsqueeze(torch.sin(x[:,1]),-1)),-1)
		return ret 

class Taylor(nn.Module):
	def __init__(self, dim, order):
		super(Taylor, self).__init__()
		self.dim = dim
		self.indc = Counter(map(toolz.compose(tuple,sorted),itertools.chain(*[itertools.product(*[range(dim) for _ in range(o)]) for o in range(order+1)])
                  ))
		print(sorted(self.indc))
		self.nterms = len(self.indc)

	def forward(self,x):
		ret = torch.cat([torch.unsqueeze(1.*self.indc[ind]/scipy.math.factorial(len(ind))*torch.prod(torch.stack([x[...,d]**ind.count(d) for d in range(self.dim)]),0),-1)  for ind in sorted(self.indc)],-1)
		return ret 

class ResBlock(nn.Module):
	def __init__(self, dim, n_layers=1, n_units=50, nonlinear=nn.Tanh, device="cpu"):
		super(ResBlock, self).__init__()
		self.dim = dim
		self.net = create_net(dim, dim, n_layers=n_layers, n_units=n_units, nonlinear = nn.Tanh).to(device)
		init_network_weights_zero(self.net)

	def forward(self, x):
		return self.net(x) + x
