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

class TotalDegree(nn.Module):
	def __init__(self, dim, order):
		super(TotalDegree, self).__init__()
		self.dim = dim
		self.indc = Counter(map(toolz.compose(tuple,sorted),itertools.chain(*[itertools.product(*[range(dim) for _ in range(o)]) for o in range(order+1)])
                  ))
		#print(sorted(self.indc))
		self.nterms = len(self.indc)

	def forward(self,x):
		ret = torch.cat([torch.unsqueeze(torch.prod(torch.stack([x[...,d]**ind.count(d) for d in range(self.dim)]),0),-1)  for ind in sorted(self.indc)],-1)
		return ret 
