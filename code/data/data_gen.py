import os
import numpy as np

import torch
import torch.nn as nn

from torchdiffeq import odeint

def cubic_rhs(t,x):
	return torch.cat( (-0.1*x[:,0]**3+2.0*x[:,1]**3, -2.0*x[:,0]**3-0.1*x[:,1]**3), axis=-1)

def lv_rhs(t,x):
	alpha = 0.3543
	beta = 0.4301
	gamma = 0.3256
	delta = 0.2500
	return torch.cat( (alpha*x[:,0]-beta*x[:,0]*x[:,1], delta*x[:,0]*x[:,1]-gamma*x[:,1]), axis=-1)


def simulation_run(eqtype, T, dt, ntrain, nval, ntest):
	total_steps = int(T/dt)
	t = torch.linspace(0, T, total_steps+1)

	if eqtype == "cubic":
		n = 2
		rhs = cubic_rhs
		a, b = -1.0, 1.0
	elif eqtype == "lv":
		n = 2
		rhs = lv_rhs
		a, b = .25, 1.0
		
	train_data = np.zeros((ntrain, total_steps+1, n))
	print('generating training trials ...')
	for i in range(ntrain):
		x_init = torch.tensor(np.random.uniform(a, b, n)).unsqueeze(0)
		sol = odeint(rhs,x_init,t,method='dopri5').squeeze().detach().numpy()
		train_data[i, :, :] = sol

	val_data = np.zeros((nval, total_steps+1, n))
	print('generating validation trials ...')
	for i in range(nval):
		x_init = torch.tensor(np.random.uniform(a, b, n)).unsqueeze(0)
		sol = odeint(rhs,x_init,t,method='dopri5').squeeze().detach().numpy()
		val_data[i, :, :] = sol
	    
	test_data = np.zeros((ntest, total_steps+1, n))
	print('generating testing trials ...')
	for i in range(ntest):
		x_init = torch.tensor(np.random.uniform(a, b, n)).unsqueeze(0)
		sol = odeint(rhs,x_init,t,method='dopri5').squeeze().detach().numpy()
		test_data[i, :, :] = sol

	# need to add noise later
	return train_data, val_data, test_data, t

def simulate_lv_cp():
	np.random.seed(1)

	dt = 0.01       # set to 5e-4 for Lorenz
	noise = 0.0      # for study of noisy measurements, we use noise=0.01, 0.02; otherwise we leave it as 0.
	
	n_forwards = np.asarray([7,4,6,5])
	
	total_steps = 512 * n_forwards
	total_step = total_steps.sum()
	
	t = torch.linspace(0, (total_step)*dt, total_step+4)
	t_idx = np.insert(total_steps+1, 0, 0).cumsum()
	
	alphas = torch.tensor(np.random.uniform(.25, .5, len(n_forwards)))
	betas = torch.tensor(np.random.uniform(.25, .5, len(n_forwards)))
	deltas = torch.tensor(np.random.uniform(.25, .5, len(n_forwards)))
	gammas = torch.tensor(np.random.uniform(.25, .5, len(n_forwards)))
	
	print(alphas,betas,deltas,gammas)
	# simulation parameters
	
	n = 2
	# simulate training trials 
	train_data = [] 
	x_init = torch.tensor(np.random.uniform(.5, 2.0, n)).unsqueeze(0)
	for i in range(len(n_forwards)):
		def cubic_rhs_torch(t,x):
			return torch.cat( (alphas[i]*x[:,0]-betas[i]*x[:,0]*x[:,1], deltas[i]*x[:,0]*x[:,1]-gammas[i]*x[:,1]), axis=-1)
		sol = odeint(cubic_rhs_torch,x_init,t[t_idx[i]:t_idx[i+1]],method='dopri5').squeeze().detach().numpy()
		train_data.append(sol)
		#print(sol.shape)
		x_init = torch.tensor(sol[-1,:]).unsqueeze(0)
	train_data = np.concatenate(train_data, axis=0)
	train_data = np.delete(train_data, total_steps, 0)
	#print(train_data.shape)
	return train_data, t[:-4]

if __name__ == "__main__":
    train, val, test, t = simulation_run("cubic", 51.2, 0.01, 10, 10, 10)
    print(train.shape)
