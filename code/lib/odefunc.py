import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.utils as utils
from lib.utils import TotalDegree

import itertools
import toolz
from collections import Counter

#####################################################################################################

class ODEfuncPoly(nn.Module):
	def __init__(self, dim, order, feature_names=None,C_init=None, device = torch.device("cpu")):
		super(ODEfuncPoly, self).__init__()
		self.NFE = 0
		self.dim = dim
		self.TP = TotalDegree(dim,order)
		self.C = nn.Linear(self.TP.nterms,dim,bias=False)
		if C_init is not None:
			with torch.no_grad():
				self.C.weight.copy_(torch.tensor(C_init))
		nn.init.zeros_(self.C.weight)
		if feature_names is None:
			self.feature_names = ['x_{:d}'.format(i) for i in range(dim)]
		self.dict = []
        
	def get_dict(self):
		indc = sorted(self.TP.indc)
		power = [[ind.count(d) for d in range(self.dim)] for ind in indc]
		dict = []
		for ind in range(len(indc)):
			dict_cur = ""
			for i in range(len(power[ind])):
				if power[ind][i] != 0:
					dict_cur += self.feature_names[i]+"^"+str(power[ind][i])
			if dict_cur == '':
				dict_cur = 'c'
			dict.append(dict_cur)
		self.dict = dict
		for i in range(self.dim):
			dot_exp = "\dot{"+self.feature_names[i]+"}"
			for d in range(len(dict)):
				if self.C.weight[i,d] != 0: 
					if self.C.weight[i,d] > 0:
						dot_exp +="+"+str(self.C.weight[i,d].detach().numpy()) +  dict[d]
					else:
						dot_exp +=str(self.C.weight[i,d].detach().numpy()) +  dict[d]
			print(r"$"+dot_exp+"$")

	def forward(self, t, y):
		P = self.TP(y)
		output = self.C(P)
		return output 


