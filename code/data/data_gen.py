import os
import numpy as np

import torch
import torch.nn as nn

from torchdiffeq import odeint

def cubic_rhs(t,x):
	return torch.cat( (-0.1*x[:,0]**3+2.0*x[:,1]**3, -2.0*x[:,0]**3-0.1*x[:,1]**3), axis=-1)


def simulation_run(eqtype, T, dt, ntrain, nval, ntest):
	total_steps = int(T/dt)
	t = torch.linspace(0, T, total_steps+1)

	if eqtype == "cubic":
		n = 2
		rhs = cubic_rhs
		a, b = -1.0, 1.0
		
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

if __name__ == "__main__":
    train, val, test, t = simulation_run("cubic", 51.2, 0.01, 10, 10, 10)
    print(train.shape)
