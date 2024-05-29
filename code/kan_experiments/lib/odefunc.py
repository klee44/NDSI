import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.utils as utils
from lib.utils import TensorProduct, Taylor, TotalDegree, TotalDegreeTrig

#from kan import KAN
from eff_kan import KAN

#####################################################################################################

class ODEfunc_KAN(nn.Module):
	#def __init__(self, dim, nlayer, nunit, grid=5, k=3, lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0.,device = torch.device("cpu")):
	def __init__(self, dim, kanlayer, grid=5, k=3, lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0.,device = torch.device("cpu")):
		super(ODEfunc_KAN, self).__init__()
		#dims = [dim] + [nunit for i in range(nlayer)] + [dim]
		if kanlayer is not None:
			dims = [dim] + kanlayer + [dim]
		else:
			dims = [dim] + [dim]
		#self.gradient_net = KAN([dim, 10, dim]).to(device)
		#self.gradient_net = KAN(dims, grid=grid, k=k)
		self.gradient_net = KAN(dims, grid_size=grid, spline_order=k)
		self.NFE = 0
		self.reg_total = 0.
		self.lamb_l1=lamb_l1
		self.lamb_entropy=lamb_entropy
		self.lamb_coef=lamb_coef
		self.lamb_coefdiff=lamb_coefdiff

	def reg(self, acts_scale, act_fun):
		def nonlinear(x, th=1e-16, factor=1.): 
			return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

		reg_ = 0.
		for i in range(len(acts_scale)):
			vec = acts_scale[i].reshape(-1, )

			p = vec / torch.sum(vec)
			l1 = torch.sum(nonlinear(vec))
			entropy = - torch.sum(p * torch.log2(p + 1e-4))
			reg_ += self.lamb_l1 * l1 + self.lamb_entropy * entropy  # both l1 and entropy
		for i in range(len(act_fun)):
			coeff_l1 = torch.sum(torch.mean(torch.abs(act_fun[i].coef), dim=1))
			coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(act_fun[i].coef)), dim=1))
			reg_ += self.lamb_coef * coeff_l1 + self.lamb_coefdiff * coeff_diff_l1

		return reg_
	
	def forward(self, t, y):
		output = self.gradient_net(y)
		#reg = self.reg(self.gradient_net.acts_scale, self.gradient_net.act_fun)
		#self.reg_total += reg
		return output 

class ODEFunc_(nn.Module):
	def __init__(self, input_dim, latent_dim, ode_func_net, device = torch.device("cpu")):
		"""
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for ODE. Analog of a continous latent state
		"""
		super(ODEFunc, self).__init__()

		self.input_dim = input_dim
		self.device = device

		self.gradient_net = ode_func_net

	def forward(self, t_local, y, backwards = False):
		"""
		Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

		t_local: current time point
		y: value at the current time point
		"""
		grad = self.get_ode_gradient_nn(t_local, y)
		if backwards:
			grad = -grad
		return grad

	def get_ode_gradient_nn(self, t_local, y):
		return self.gradient_net(y)

	def sample_next_point_from_prior(self, t_local, y):
		"""
		t_local: current time point
		y: value at the current time point
		"""
		return self.get_ode_gradient_nn(t_local, y)

class POULayer(nn.Module):
	def __init__(self, bias=True,npart=3, npoly=0):
		super().__init__()
		self.npart = npart
		self.npolydim = npoly+1
		self.xrbf =  nn.Parameter(torch.linspace(10.0, 60.0, npart))
		self.epsrbf = nn.Parameter(((1.0)/npart)*torch.ones(npart))
		self.Ppow = torch.arange(0,float(self.npolydim))

	def reset_parameters(self):
		torch.nn.init.zeros_(self.coeffs)
		
	def getpoulayer(self, x):
		rrbf = torch.transpose(torch.pow(torch.abs(x.unsqueeze(0)-self.xrbf.unsqueeze(1)),1), 1, 0) 
		rbflayer = torch.exp( - (rrbf/(torch.pow(self.epsrbf,2.0)+1.e-4) ) ) 
		rbfsum = torch.sum(rbflayer, axis=1)
		return torch.transpose(torch.transpose(rbflayer, 1, 0)/rbfsum, 1, 0)
		        
	def calculate_weights(self, t):
		basis = torch.pow(t,self.Ppow)
		poly = torch.einsum('ijk,k->ij',self.coeffs,basis)
		parts = self.getpoulayer(t)
		return torch.einsum('ij,kj->i', poly, parts) 

class POULinear(POULayer):
	def __init__(self, in_features, out_features, bias=True):       
		super().__init__(bias)
        
		self.in_features, self.out_features = in_features, out_features
		self.weight = torch.Tensor(out_features, in_features)
		if bias:
			self.bias = torch.Tensor(out_features)
		else:
			self.register_parameter('bias', None)         
		self.coeffs = torch.nn.Parameter(torch.Tensor((in_features+1)*out_features, self.npart, self.npolydim))        
		self.reset_parameters()  
		                
	def forward(self, input):
		t = input[-1,-1]
		input = input[:,:-1]
		w = self.calculate_weights(t)
		self.weight = w[0:self.in_features*self.out_features].reshape(self.out_features, self.in_features)
		self.bias = w[self.in_features*self.out_features:(self.in_features+1)*self.out_features].reshape(self.out_features)
		return torch.nn.functional.linear(input, self.weight, self.bias)

class POUPoly(POULayer):
	def __init__(self, in_features, out_features, bias=True):       
		super().__init__(bias)
        
		self.in_features, self.out_features = in_features, out_features
		self.weight = torch.Tensor(out_features, in_features)
		self.register_parameter('bias', None)         

		self.coeffs = torch.nn.Parameter(torch.zeros((in_features+1)*out_features, self.npart, self.npolydim))        
		self.reset_parameters()  
		                
	def forward(self, input):
		t = input[-1,-1]
		input = input[:,:-1]
		w = self.calculate_weights(t)
		self.weight = w[0:self.in_features*self.out_features].reshape(self.out_features, self.in_features)
		return torch.nn.functional.linear(input, self.weight)

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

class ODEfuncPOU(nn.Module):
	def __init__(self):
		super(ODEfuncPOU, self).__init__()
		
		hdim = 50 
		self.net = nn.Sequential(
			nn.Linear(2, hdim),
			nn.Tanh(),
			DepthCat(1),
			POULinear(hdim, hdim),
			nn.Tanh(),
			nn.Linear(hdim, 2)
		)
		
	def forward(self, t, y):
		for _, module in self.net.named_modules():
			if hasattr(module, 't'):
				module.t = t
		return self.net(y)

class ODEfuncPOUPoly(nn.Module):
	def __init__(self, dim, order, C_init=None, device = torch.device("cpu")):
		super(ODEfuncPOUPoly, self).__init__()
		self.NFE = 0
		self.TP = TotalDegree(dim,order)
	
		self.net = nn.Sequential(
			DepthCat(1),
			POUPoly(self.TP.nterms, dim)
		)

	def forward(self, t, y):
		for _, module in self.net.named_modules():
			if hasattr(module, 't'):
				module.t = t
		P = self.TP(y)
		output = self.net(P)
		return output 

class ODEfunc(nn.Module):
	def __init__(self, dim, nlayer, nunit, device = torch.device("cpu")):
		super(ODEfunc, self).__init__()
		self.gradient_net = utils.create_net(dim, dim, n_layers=nlayer, n_units=nunit, nonlinear = nn.Tanh).to(device)
		self.NFE = 0

	def forward(self, t, y):
		output = self.gradient_net(y)
		return output 

class ODEfuncPoly(nn.Module):
	def __init__(self, dim, order, C_init=None, device = torch.device("cpu")):
		super(ODEfuncPoly, self).__init__()
		self.NFE = 0
		#self.TP = TensorProduct(dim,order)
		self.TP = TotalDegree(dim,order)
		#self.TP = Taylor(dim,order)
		#self.C = nn.Parameter(torch.randn((self.TP.nterms, dim), requires_grad=True))
		self.C = nn.Linear(self.TP.nterms,dim,bias=False)
		if C_init is not None:
			with torch.no_grad():
				self.C.weight.copy_(torch.tensor(C_init))
			#self.C.weight = nn.Parameter(torch.tensor(C_init))
		#utils.init_network_weights_orthogonal(self.C)
		nn.init.zeros_(self.C.weight)

	def forward(self, t, y):
		P = self.TP(y)
		#output = torch.einsum('ab,za->zb',self.C,P)
		output = self.C(P)
		return output 

class ODEfuncPolyGS(nn.Module):
	def __init__(self, dim, order, device = torch.device("cpu")):
		super(ODEfuncPolyGS, self).__init__()
		self.NFE = 0
		self.TP = TotalDegree(dim,order)
		#self.C = torch.randn((self.TP.nterms, dim), requires_grad=False)
		self.C = nn.Parameter(torch.randn((self.TP.nterms, dim), requires_grad=True))
		nn.init.kaiming_uniform_(self.C, mode='fan_out',a=np.sqrt(5))
		logits_init = .0*torch.ones((self.TP.nterms,dim,1), requires_grad=True)
		self.logits = nn.Parameter(logits_init)
		self.odd = F.sigmoid(self.logits)
		self.mask = F.gumbel_softmax(torch.cat((1-self.odd, self.odd),axis=-1), tau=1, hard=True)[:,:,1]

	def update_mask(self):
		self.odd = F.sigmoid(self.logits)
		self.mask = F.gumbel_softmax(torch.cat((1-self.odd, self.odd),axis=-1), tau=.1, hard=False)[:,:,1]

	def forward(self, t, y):
		P = self.TP(y)
		output = torch.einsum('ab,za->zb',torch.mul(self.mask,self.C),P)
		#output = torch.einsum('ab,za->zb',self.C,P)
		return output 

class ODEfuncPolyTrig(nn.Module):
	def __init__(self, dim, order, device = torch.device("cpu")):
		super(ODEfuncPolyTrig, self).__init__()
		self.NFE = 0
		self.TP = TotalDegreeTrig(dim,order)
		self.C = nn.Linear(self.TP.nterms,dim,bias=False)

	def forward(self, t, y):
		P = self.TP(y)
		output = self.C(P)
		return output

class ODEfuncHNN(nn.Module):
	def __init__(self, dim, order, device = torch.device("cpu")):
		super(ODEfuncHNN, self).__init__()
		self.NFE = 0
		self.TP = TotalDegree(dim,order)
		self.C = nn.Linear(self.TP.nterms,1,bias=False)
		self.L = np.zeros((2,2))
		self.L[0,1], self.L[1,0] = 1, -1
		self.L = torch.tensor(self.L).to(device)

	def forward(self, t, y):
		P = self.TP(y)
		H = self.C(P) 

		dH = torch.autograd.grad(H.sum(), y, create_graph=True)[0]
		output = dH @ self.L.t()
		return output 

class ODEfuncHNNTrig(nn.Module):
	def __init__(self, dim, order, device = torch.device("cpu")):
		super(ODEfuncHNNTrig, self).__init__()
		self.NFE = 0
		self.TP = TotalDegreeTrig(dim,order)
		self.C = nn.Linear(self.TP.nterms,1,bias=False)
		self.L = np.zeros((2,2))
		self.L[0,1], self.L[1,0] = 1, -1
		self.L = torch.tensor(self.L).to(device)

	def forward(self, t, y):
		P = self.TP(y)
		H = self.C(P) 

		dH = torch.autograd.grad(H.sum(), y, create_graph=True)[0]
		output = dH @ self.L.t()
		return output 

class ODEfuncPortHNN(nn.Module):
	def __init__(self, dim, order, device = torch.device("cpu")):
		super(ODEfuncPortHNN, self).__init__()
		self.NFE = 0
		self.TP = TotalDegree(dim,order)
		self.C = nn.Linear(self.TP.nterms,1,bias=False)
		self.L = np.zeros((2,2))
		self.L[0,1], self.L[1,0] = 1, -1
		self.L = torch.tensor(self.L).to(device)
		self.N = nn.Parameter(torch.randn((1), requires_grad=True))

	def forward(self, t, y):
		P = self.TP(y)
		H = self.C(P) 

		dH = torch.autograd.grad(H.sum(), y, create_graph=True)[0]
		output = dH @ self.L.t()  
		#output[:,1] = output[:,1] + self.N*dH[:,1] + 0.2*torch.sin(1.2*t)
		output[:,1] = output[:,1] + self.N*dH[:,1] + 0.39*torch.sin(1.4*t)
		return output 

class ODEfuncGNN(nn.Module):
	def __init__(self, dim, order, D1, D2, device = torch.device("cpu")):
		super(ODEfuncGNN, self).__init__()
		self.NFE = 0

		self.P_E = TotalDegreeTrig(dim,order)
		#self.P_E = TotalDegree(dim,order)
		self.C = nn.Linear(self.P_E.nterms,1,bias=False)

		self.L = np.zeros((dim,dim))
		self.L[0,1], self.L[1,0] = 1, -1
		self.L = torch.tensor(self.L).to(device)

		self.D_M = nn.Parameter(torch.randn((D1, D2), requires_grad=True))
		self.L_M = nn.Parameter(torch.randn((dim, dim, D1), requires_grad=True))

	def friction_matrix(self,dE):
		D = self.D_M @ torch.transpose(self.D_M, 0, 1)
		L = (self.L_M - torch.transpose(self.L_M, 0, 1))/2.0
		zeta = torch.einsum('abm,mn,cdn->abcd',L,D,L) # zeta [alpha, beta, mu, nu] 
		self.M = torch.einsum('abmn,zb,zn->zam',zeta,dE,dE)

	def friction_matvec(self,dE,dS): 	
		D = self.D_M @ torch.transpose(self.D_M, 0, 1)
		L = (self.L_M - torch.transpose(self.L_M, 0, 1))/2.0
		zeta = torch.einsum('abm,mn,cdn->abcd',L,D,L) # zeta [alpha, beta, mu, nu] 
		MdS = torch.einsum('abmn,zb,zm,zn->za',zeta,dE,dS,dE)
		return MdS 

	def dEdt(self,dE,dS):
		D = self.D_M @ torch.transpose(self.D_M, 0, 1)
		L = (self.L_M - torch.transpose(self.L_M, 0, 1))/2.0
		zeta = torch.einsum('abm,mn,cdn->abcd',L,D,L) # zeta [alpha, beta, mu, nu] 
		MdS = torch.einsum('abmn,zb,zm,zn->za',zeta,dE,dS,dE)
		self.dEMdS = torch.einsum('za,za->z',dE,MdS)

	def dSdt(self,dE,dS):
		D = self.D_M @ torch.transpose(self.D_M, 0, 1)
		L = (self.L_M - torch.transpose(self.L_M, 0, 1))/2.0
		zeta = torch.einsum('abm,mn,cdn->abcd',L,D,L) # zeta [alpha, beta, mu, nu] 
		MdS = torch.einsum('abmn,zb,zm,zn->za',zeta,dE,dS,dE)
		self.dSMdS = torch.einsum('za,za->z',dS,MdS)

	def forward(self, t, y):
		P_E = self.P_E(y)
		E = self.C(P_E) 

		dE = torch.autograd.grad(E.sum(), y, create_graph=True)[0]
		LdE = dE @ self.L.t()

		S = y[:,-1]
		dS = torch.autograd.grad(S.sum(), y, create_graph=True)[0]

		MdS = self.friction_matvec(dE,dS)
		output = LdE + MdS

		#self.friction_matrix(dE) 
		#self.MdE = self.friction_matvec(dE,dE)
		#print(self.MdE)
		# post proc
		self.dEdt(dE,dS)
		self.dSdt(dE,dS)
		return output 

class ODEfunc_GENERIC(nn.Module):
	def __init__(self, output_dim, D1, D2, lE, nE, lS, nS, device=torch.device("cpu")):
		super(ODEfunc_GENERIC, self).__init__()
		self.output_dim = output_dim

		self.dimD = D1
		self.dimD2 = D2

		self.friction_D = nn.Parameter(torch.randn((self.dimD, self.dimD2), requires_grad=True))
		self.friction_L = nn.Parameter(torch.randn((self.output_dim, self.output_dim, self.dimD), requires_grad=True)) # [alpha, beta, m] or [mu, nu, n]

		self.poisson_xi = nn.Parameter(torch.randn((self.output_dim, self.output_dim, self.output_dim), requires_grad=True))

		self.energy = utils.create_net(output_dim, 1, n_layers=lE, n_units=nE, nonlinear = nn.Tanh).to(device)
		self.entropy = utils.create_net(output_dim, 1, n_layers=lS, n_units=nS, nonlinear = nn.Tanh).to(device)
		
		self.NFE = 0

	def Poisson_matvec(self,dE,dS):
		# zeta [alpha, beta, gamma]
		xi = (self.poisson_xi - self.poisson_xi.permute(0,2,1) + self.poisson_xi.permute(1,2,0) -
			self.poisson_xi.permute(1,0,2) + self.poisson_xi.permute(2,0,1) - self.poisson_xi.permute(2,1,0))/6.0
		
		# dE and dS [batch, alpha]
		LdE = torch.einsum('abc, zb, zc -> za',xi,dE,dS)
		return LdE 

	def friction_matvec(self,dE,dS): 	
		# D [m,n] L [alpha,beta,m] or [mu,nu,n] 
		D = self.friction_D @ torch.transpose(self.friction_D, 0, 1)
		L = (self.friction_L - torch.transpose(self.friction_L, 0, 1))/2.0
		zeta = torch.einsum('abm,mn,cdn->abcd',L,D,L) # zeta [alpha, beta, mu, nu] 
		MdS = torch.einsum('abmn,zb,zm,zn->za',zeta,dE,dS,dE)
		return MdS 

	def get_dSdt(self,dS,dE):
		xi = (self.poisson_xi - self.poisson_xi.permute(0,2,1) + self.poisson_xi.permute(1,2,0) -
			self.poisson_xi.permute(1,0,2) + self.poisson_xi.permute(2,0,1) - self.poisson_xi.permute(2,1,0))/6.0
		dSLdE = torch.einsum('abc, za, zb, zc -> z',xi,dS,dE,dS)

		D = self.friction_D @ torch.transpose(self.friction_D, 0, 1)
		L = (self.friction_L - torch.transpose(self.friction_L, 0, 1))/2.0
		zeta = torch.einsum('abm,mn,cdn->abcd',L,D,L) # zeta [alpha, beta, mu, nu] 
		dSMdS = torch.einsum('abmn,za,zb,zm,zn->z',zeta,dS,dE,dS,dE)

		return dSLdE + dSMdS
	
	def get_penalty(self):
		return self.LdS, self.MdE

	def forward(self, t, y):
		E = self.energy(y)
		S = self.entropy(y)

		dE = torch.autograd.grad(E.sum(), y, create_graph=True)[0]
		dS = torch.autograd.grad(S.sum(), y, create_graph=True)[0] 

		LdE = self.Poisson_matvec(dE,dS)
		MdS = self.friction_matvec(dE,dS)
		output = LdE  + MdS
		#print(output.shape)
		self.NFE = self.NFE + 1

		# compute penalty
		self.LdS = self.Poisson_matvec(dS,dS)
		self.MdE = self.friction_matvec(dE,dE)
		self.dSdt = self.get_dSdt(dS,dE)
		return output 
