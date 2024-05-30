import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os, sys

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import random
from random import SystemRandom

import matplotlib.pyplot as plt

import lib.utils as utils
from lib.odefunc import ODEfunc_KAN
from torchdiffeq import odeint as odeint

import json

import argparse
parser = argparse.ArgumentParser(description='.')
parser.add_argument('--r', type=int, default=0, help='random_seed')

parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--nepoch', type=int, default=500, help='max epochs')
parser.add_argument('--niterbatch', type=int, default=100, help='max epochs')

parser.add_argument('--nlayer', type=int, default=4, help='max epochs')
parser.add_argument('--nunit', type=int, default=100, help='max epochs')
parser.add_argument('--grid', type=int, default=5, help='max epochs')
parser.add_argument('--k', type=int, default=3, help='max epochs')
parser.add_argument('--kanlayer', type=int, nargs='*', default=None, help='max epochs')

parser.add_argument('--lMB', type=int, default=100, help='length of seq in each MB')
parser.add_argument('--nMB', type=int, default=40, help='length of seq in each MB')

parser.add_argument('--odeint', type=str, default='rk4', help='integrator')

parser.add_argument('--lamb', type=float, default=0., help='learning rate')
parser.add_argument('--lamb_l1', type=float, default=1., help='learning rate')
parser.add_argument('--lamb_entropy', type=float, default=2., help='learning rate')
parser.add_argument('--lamb_coef', type=float, default=0., help='learning rate')
parser.add_argument('--lamb_coefdiff', type=float, default=0., help='learning rate')

args = parser.parse_args()

torch.set_default_dtype(torch.float32)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

seed = args.r
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

save_path = 'experiments/'
utils.makedirs(save_path)


f = open('data/strogatz_extended.json')
data = json.load(f)
f.close()
print(len(data))
for i in range(63):
	print(i,data[i]['eq_description'], len(data[i]['solutions'][0]))

for i in range(63):
	id_path = os.path.join(save_path, "id_"+str(i)) 
	utils.makedirs(id_path)
	for j in range(2):
		exp_path = os.path.join(id_path, "initcond_"+str(j))
		utils.makedirs(exp_path)
		print(exp_path)
		ckpt_path = os.path.join(exp_path, "model_kan" + str(seed) + '.ckpt')
		fig_save_path = os.path.join(exp_path,"model_kan"+str(seed))
		utils.makedirs(fig_save_path)
		print(fig_save_path)
		print(i)
		dim = data[i]['dim']

		t = np.array(data[i]['solutions'][0][j]['t'])	
		t = torch.tensor(t).squeeze().to(torch.float32)

		trajs = []
		for k in range(dim):
			trajs.append(np.expand_dims(np.array(data[i]['solutions'][0][j]['y'][k]),-1))
		trajs = np.concatenate(trajs, 1)
		train_data = torch.tensor(trajs).unsqueeze(0).to(torch.float32)

		odefunc = ODEfunc_KAN(dim, args.kanlayer, args.grid, args.k)
		
		params = odefunc.parameters()
		optimizer = optim.Adamax(params, lr=args.lr)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9987)
		
		best_loss = 1e30
		frame = 0 
		
		for itr in range(args.nepoch):
			print('=={0:d}=='.format(itr))
			#for mb_data in train_data:
			for k in range(args.niterbatch):
				optimizer.zero_grad()
				batch_y0, batch_t, batch_y, _, _, _ = utils.get_batch_two_single(train_data,t,args.lMB,args.nMB)
				pred_y = odeint(odefunc, batch_y0, batch_t, method=args.odeint).to(device).transpose(0,1)
				loss = torch.mean(torch.abs(pred_y - batch_y))
				print(itr,k,loss.item())
				loss.backward()
				optimizer.step()
			scheduler.step()
		
			with torch.no_grad():
				pred_y = odeint(odefunc, train_data[:,0,:], t, method=args.odeint).to(device).transpose(0,1)
				val_loss = torch.mean(torch.abs(pred_y - train_data)).item()
				print('val loss', val_loss)
					
				if best_loss > val_loss:
					print('saving...', val_loss)
					torch.save({'state_dict': odefunc.state_dict(),}, ckpt_path)
					best_loss = val_loss 
		
				plt.figure()
				plt.tight_layout()
				save_file = os.path.join(fig_save_path,"image_{:03d}.png".format(frame))
				fig = plt.figure(figsize=(4,4))
				axes = []
				axes.append(fig.add_subplot(1,1,1))
				for k in range(dim):
					#axes.append(fig.add_subplot(1,dim,k+1))
					axes[0].plot(t,train_data[0,:,k].detach().numpy(),lw=2,color='k')
					axes[0].plot(t,pred_y.detach().numpy()[0,:,k],lw=2,color='c',ls='--')
					plt.savefig(save_file)
				plt.close(fig)
				plt.close('all')
				plt.clf()
				frame += 1
		
		ckpt = torch.load(ckpt_path)
		odefunc.load_state_dict(ckpt['state_dict'])
		
		odefunc.NFE = 0
		test_loss = 0
		
		pred_y = odeint(odefunc, train_data[:,0,:], t, method=args.odeint).to(device).transpose(0,1)
		test_loss = torch.mean(torch.abs(pred_y - train_data)).item()
		print('test loss', test_loss)
		
		fig = plt.figure(figsize=(4,4))
		axes = []
		axes.append(fig.add_subplot(1,1,1))
		for k in range(dim):
			#axes.append(fig.add_subplot(1,dim,k+1))
			axes[0].plot(t,train_data[0,:,k],lw=3,color='k')
			axes[0].plot(t,pred_y.detach().numpy()[0,:,k],lw=2,color='c',ls='--')
		
		save_file = os.path.join(fig_save_path,"image_best.png")
		plt.savefig(save_file)

