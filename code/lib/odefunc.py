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

class ODEfuncPOUPoly(nn.Module):
	def __init__(self, dim, order, npart, npoly, seqlen, C_init=None, device = torch.device("cpu")):
		super(ODEfuncPOUPoly, self).__init__()
		self.NFE = 0
		self.npart = npart
		self.TP = TotalDegree(dim,order)
		self.dim = dim
	
		self.net = nn.Sequential(
			DepthCat(1),
			POUPoly(self.TP.nterms, dim, npart, npoly, seqlen)
		)

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

		for p in range(self.npart):
			for i in range(self.dim):
				dot_exp = "\dot{"+self.feature_names[i]+"}"
				for d in range(len(dict)):
					if self.net[1].coeffs[len(dict)*i+d,p,0] != 0: 
						if self.net[1].coeffs[len(dict)*i+d,p,0] > 0:
							dot_exp +="+"+str(self.net[1].coeffs[len(dict)*i+d,p,0].detach().numpy()) +  dict[d]
						else:
							dot_exp +=str(self.net[1].coeffs[len(dict)*i+d,p,0].detach().numpy()) +  dict[d]
				print(r"$"+dot_exp+"$")

	def forward(self, t, y):
		for _, module in self.net.named_modules():
			if hasattr(module, 't'):
				module.t = t
		P = self.TP(y)
		output = self.net(P)
		return output 

class DepthCat(nn.Module):
	def __init__(self, idx_cat=1):
		super().__init__()
		self.idx_cat = idx_cat
		self.t = None

	def forward(self, x):
		t_shape = list(x.shape)
		t_shape[self.idx_cat] = 1
		t = self.t * torch.ones(t_shape).to(x)
		return torch.cat([x, t], self.idx_cat).to(x)

class POULayer(nn.Module):
	#def __init__(self, bias=True, npart=4, npoly=0):
	def __init__(self, npart=8, npoly=0, seqlen = 100):
		super().__init__()
		self.npart = npart
		self.npolydim = npoly+1
		self.xrbf =  nn.Parameter(torch.linspace(0., 1., 2*npart+1)[1:-1][::2])
		self.epsrbf = nn.Parameter(((.2)/npart)*torch.ones(npart))
		self.Ppow = torch.arange(0,float(self.npolydim))

	def reset_parameters(self):
		torch.nn.init.zeros_(self.coeffs)
		
	def getpoulayer(self, x):
		rrbf = torch.transpose(torch.pow(torch.abs(x.unsqueeze(0)-self.xrbf.unsqueeze(1)),1), 1, 0) 
		rbflayer = torch.exp(-(rrbf/(torch.pow(self.epsrbf,2.0)))) 
		rbfsum = torch.sum(rbflayer, axis=1)
		return torch.transpose(torch.transpose(rbflayer, 1, 0)/rbfsum, 1, 0)
		        
	def calculate_weights(self, t):
		basis = torch.pow(t,self.Ppow)
		poly = torch.einsum('ijk,k->ij',self.coeffs,basis) # i: # weights, j: # partitions
		parts = self.getpoulayer(t)
		return torch.einsum('ij,kj->i', poly, parts) 

class POUPoly(POULayer):
	def __init__(self, in_features, out_features, npart=4, npoly=2, seqlen=100):
		super().__init__(npart, npoly, seqlen)
        
		self.in_features, self.out_features = in_features, out_features
		self.seqlen = seqlen
		self.weight = torch.Tensor(out_features, in_features)
		self.register_parameter('bias', None)         

		self.coeffs = torch.nn.Parameter(torch.zeros((in_features)*out_features, self.npart, self.npolydim))        
		self.reset_parameters()  
		                
	def forward(self, input):
		t = input[-1,-1] / self.seqlen
		input = input[:,:-1]
		w = self.calculate_weights(t)
		self.weight = w[0:self.in_features*self.out_features].reshape(self.out_features, self.in_features)
		return torch.nn.functional.linear(input, self.weight)


