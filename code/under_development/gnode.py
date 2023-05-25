import csv
import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np
import os, sys

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import lib.utils as utils

import random
from random import SystemRandom

import scipy
import scipy.io as sio 
import scipy.linalg as scilin
from scipy.optimize import newton, brentq
from scipy.special import legendre, roots_legendre

import matplotlib.pyplot as plt

from torchdiffeq import odeint as odeint
import argparse

parser = argparse.ArgumentParser(description='.')
parser.add_argument('--r', type=int, default=0, help='random_seed')
parser.add_argument('--lE', type=int, default=2, help='random_seed')
parser.add_argument('--lS', type=int, default=2, help='random_seed')
parser.add_argument('--nE', type=int, default=15, help='random_seed')
parser.add_argument('--nS', type=int, default=15, help='random_seed')
parser.add_argument('--D1', type=int, default=4, help='random_seed')
parser.add_argument('--D2', type=int, default=4, help='random_seed')

args = parser.parse_args()

torch.set_default_dtype(torch.float64)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

seed = args.r 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

save_path = 'experiments/'
utils.makedirs(save_path)
experimentID = int(SystemRandom().random()*100000)
ckpt_path = os.path.join(save_path, "experiment_" + str(experimentID) + '.ckpt')
fig_save_path = os.path.join(save_path,"experiment_"+str(experimentID))
utils.makedirs(fig_save_path)
ckpt_path_outer = os.path.join(save_path, "experiment_" + str(experimentID) + '_outer.ckpt')
print(ckpt_path)

# 
# Parameters for double pendulum
l1 = 1
l2 = 0.9
m1 = 1
m2 = 1
g  = 1
k1 = 0.1
k2 = 0.1

# Function for computing RHS of DP ODEs
def DoublePendulum(t, x):
    th1, th2, dth1, dth2 = x
    delth  = th1 - th2
  
    alpha  = k1 * dth1
    beta   = k2 * dth2
    gamma1 = 2. * alpha - 2.*beta * torch.cos(delth)
    gamma2 = 2. * alpha * torch.cos(delth) - 2. * (m1+m2) * beta / m2
  
    num1 = (m2 * l1 * dth1**2 * torch.sin(2. * delth)
            + 2. * m2 * l2 * dth2**2 * torch.sin(delth)
            + 2. * g * m2 * torch.cos(th2) * torch.sin(delth)
            + 2. * g * m1 * torch.sin(th1) + gamma1)
    den1 = -2. * l1 * ( m1 + m2 * torch.sin(delth)**2 )
  
    num2 = (m2 * l2 * dth2**2 * torch.sin(2. * delth)
            + 2. * (m1 + m2) * l1 * dth1**2 * torch.sin(delth) 
            + 2. * (m1 + m2) * g * torch.cos(th1) * torch.sin(delth) + gamma2)
    den2 = 2. * l2 * ( m1 + m2 * torch.sin(delth)**2 )
  
    return torch.tensor([dth1, dth2, num1/den1, num2/den2]).float()


# Wrapper to use odeint_adjoint method
# Never did get this working...
class Lambda(nn.Module):
    def forward(self, t, x):
        return DoublePendulum(t, x)

# Parameters for simulation
nSteps  = 5000
T       = 50.
s0      = torch.tensor([1.0, torch.pi/2., 0., 0.]).float()
t       = torch.linspace(0., T, nSteps).float()

# Compute solution to DP system
with torch.no_grad():
    sol = odeint(Lambda(), s0, t, method='dopri5')


# Post-processing to extract positions at nodes
th1     = sol[:,0]
th2     = sol[:,1]
x1      =  l1 * torch.sin(th1)
y1      = -l1 * torch.cos(th1)
x2      =  x1 + l2 * torch.sin(th2)
y2      =  y1 - l2 * torch.cos(th2)
x0      =  torch.zeros_like(x1)
y0      =  torch.zeros_like(y1)

# Defining node features as positions 
# pData is [nSteps, nNodes, nCoords]
#X       = torch.cat((x0.unsqueeze(1), x1.unsqueeze(1), x2.unsqueeze(1)), 1)
#Y       = torch.cat((y0.unsqueeze(1), y1.unsqueeze(1), y2.unsqueeze(1)), 1)
X       = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1)), 1)
Y       = torch.cat((y1.unsqueeze(1), y2.unsqueeze(1)), 1)
qData   = torch.cat((X.unsqueeze(-1),Y.unsqueeze(-1)), -1)

# Defining edge features as average velocities computed by finite differences
# qData is [nSteps, nEdges, nCoords]
pData    = torch.from_numpy(np.gradient(qData.numpy(),t.numpy(), axis=0))

qData = qData[::10,:,:]
pData = pData[::10,:,:]
t = t[::10]

qpData  = torch.cat((qData, pData), 1).float()

class ODEfunc(nn.Module):
	def __init__(self, output_dim):
		super(ODEfunc, self).__init__()
		self.output_dim = output_dim
		self.dimD = args.D1 
		self.dimD2 = args.D2 
		self.friction_D = nn.Parameter(torch.randn((self.dimD, self.dimD2), requires_grad=True))
		self.friction_L = nn.Parameter(torch.randn((self.output_dim, self.output_dim, self.dimD), requires_grad=True)) # [alpha, beta, m] or [mu, nu, n]

		self.poisson_xi = nn.Parameter(torch.randn((self.output_dim, self.output_dim, self.output_dim), requires_grad=True))

		self.energy = utils.create_net(output_dim, 1, n_layers=args.lE, n_units=args.nE, nonlinear = nn.Tanh).float().to(device)
		self.entropy = utils.create_net(output_dim, 1, n_layers=args.lS, n_units=args.nS, nonlinear = nn.Tanh).to(device).float()
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
		self.NFE = self.NFE + 1

		# compute penalty
		self.LdS = self.Poisson_matvec(dS,dS)
		self.MdE = self.friction_matvec(dE,dE)
		return output 


qpData.requires_grad=True
qpData = qpData.reshape((500, 2*4))
sData = torch.unsqueeze(torch.linspace(0.,1.,len(t),requires_grad=True),-1)
sData2 = torch.unsqueeze(torch.linspace(0.,1.,len(t),requires_grad=True),-1)
sData3 = torch.unsqueeze(torch.linspace(0.,1.,len(t),requires_grad=True),-1)
sData4 = torch.unsqueeze(torch.linspace(0.,1.,len(t),requires_grad=True),-1)

qpsData = torch.cat((qpData,sData,sData2,sData3,sData4),axis=-1).float()


def get_batch(data_qps, batch_len=120, batch_size=20):
	s = torch.from_numpy(np.random.choice(np.arange(train_end - batch_len, dtype=np.int64), batch_size, replace=False))
	batch_y0 = data_qps[s]  # (M, D)
	batch_t = t_q[:batch_len]  # (T)
	batch_y = torch.stack([data_qps[s + i] for i in range(batch_len)], dim=0)[:,:,:2]  # (T, M, D)
	return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

odefunc = ODEfunc(8+4).float() 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(odefunc))

best_loss = 1e30
best_loss_outer = 1e30
params = odefunc.parameters()
optimizer = optim.Adamax(params, lr=1e-2)
frame = 0

with open(f'gnode.csv', 'w+') as csvfile:
	fieldnames = ['Iteration', 'Training Loss', 'Best Loss']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()

	for itr in range(1, 30001):
		optimizer.zero_grad()
		pred_y = odeint(odefunc, qpsData[:1,:], t, rtol=1e-5, atol=1e-6, method='dopri5').to(device).squeeze()
		loss = torch.mean(torch.abs(pred_y[:,:8] - qpsData[:,:8]).sum(axis=1))
		print(itr, loss.item())
		
		loss.backward()
		optimizer.step()
	
	
		if best_loss > loss.item():
			print('saving ode...', loss.item())
			torch.save({
				'state_dict': odefunc.state_dict(),                                           
				}, ckpt_path)
			best_loss = loss.item()

		writer.writerow({'Iteration': itr, 'Training Loss': loss.item(), 'Best Loss': best_loss})
	
		if itr % 50 == 0:
			plt.figure()
			plt.tight_layout()
			save_file = os.path.join(fig_save_path,"image_{:03d}.png".format(frame))
			fig = plt.figure(figsize=(8,8))
			axes = []
			for i in range(4):
				axes.append(fig.add_subplot(2,2,i+1))
				for j in range(2):
					axes[i].plot(t,qpData.reshape((500,4,2))[:,i,j].detach().numpy(),lw=2,color='k')
					axes[i].plot(t,pred_y[:,:8].reshape((500,4,2))[:,i,j].detach().numpy(),lw=2,color='c',ls='--')
			plt.savefig(save_file)
			plt.close(fig)
			plt.close('all')
			plt.clf()
			frame += 1
