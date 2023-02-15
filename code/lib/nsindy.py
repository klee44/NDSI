import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os, sys

import lib.utils as utils
from lib.odefunc import ODEfuncPoly
from lib.torchdiffeq import odeint as odeint
import torch.nn.utils.prune as prune
from lib.prune import ThresholdPruning

import matplotlib.pyplot as plt

def train_prune(odefunc, lr, nepoch, niterbatch, lMB, nMB, odeint_method, train_data, val_data, t, ckpt_path, fig_save_path):
	train_data = torch.tensor(train_data)
	val_data = torch.utils.data.DataLoader(torch.tensor(val_data),batch_size=50)
	
	parameters_to_prune = ((odefunc.C, "weight"),)
	
	params = odefunc.parameters()
	optimizer = optim.Adamax(params, lr=lr)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9987)
	
	best_loss = 1e30
	frame = 0 
	
	for itr in range(nepoch):
		#print('=={0:d}=='.format(itr))
		for i in range(niterbatch):
			optimizer.zero_grad()
			batch_y0, batch_t, batch_y = utils.get_batch(train_data,t,lMB,nMB)
			pred_y = odeint(odefunc, batch_y0, batch_t, method=odeint_method).transpose(0,1)
			loss = torch.mean(torch.abs(pred_y - batch_y))
			l1_norm = 1e-4*torch.norm(odefunc.C.weight, p=1)
			loss += l1_norm
			#print(itr,i,loss.item(),l1_norm.item())
			loss.backward()
			optimizer.step()
			prune.global_unstructured(parameters_to_prune, pruning_method=ThresholdPruning, threshold=1e-6)
		scheduler.step()
	
		print(itr, loss.item())	
		
		with torch.no_grad():
			val_loss = 0
			#print(odefunc.C.weight)
			
			for d in val_data:
				pred_y = odeint(odefunc, d[:,0,:], t, method=odeint_method).transpose(0,1)
				val_loss += torch.mean(torch.abs(pred_y - d)).item()
			print('val loss', val_loss)
				
			if best_loss > val_loss:
				print('saving...', val_loss)
				torch.save({'state_dict': odefunc.state_dict(),}, ckpt_path)
				best_loss = val_loss 
	
			plt.figure()
			plt.tight_layout()
			save_file = os.path.join(fig_save_path,"image_{:03d}.png".format(frame))
			fig = plt.figure(figsize=(8,4))
			axes = []
			for i in range(2):
				axes.append(fig.add_subplot(1,2,i+1))
				axes[i].plot(t,d[0,:,i].detach().numpy(),lw=2,color='k')
				axes[i].plot(t,pred_y.detach().numpy()[0,:,i],lw=2,color='c',ls='--')
				plt.savefig(save_file)
			plt.close(fig)
			plt.close('all')
			plt.clf()
			frame += 1
	ckpt = torch.load(ckpt_path)
	odefunc.load_state_dict(ckpt['state_dict'])
	
	prune.remove(odefunc.C, 'weight')
	print(odefunc.C.weight)
	torch.save({'state_dict': odefunc.state_dict(),}, ckpt_path)

def proximal(w, lam=0.1, eta=0.1):
	"""Proximal step"""
	# w shape dim \times number of dicts
	#dim, ndicts = w.shape
	#tmp = w.flatten()	
	alpha = torch.clamp(torch.abs(w) - lam * eta, min=0)
	w.data = torch.sign(w) * alpha#).reshape((dim, ndicts))

def train_ISTA(odefunc, lr, nepoch, niterbatch, lMB, nMB, odeint_method, train_data, t, l1_reg, ckpt_path, fig_save_path):
	train_data = torch.tensor(train_data)
	
	params = odefunc.parameters()
	optimizer = optim.Adamax(params, lr=lr)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9987)
	
	best_loss = 1e30
	frame = 0 
	
	for itr in range(nepoch):
		#print('=={0:d}=='.format(itr))
		for i in range(niterbatch):
			optimizer.zero_grad()
			batch_y0, batch_t, batch_y = utils.get_batch(train_data,t,lMB,nMB)
			pred_y = odeint(odefunc, batch_y0, batch_t, method=odeint_method).transpose(0,1)
			loss = torch.mean(torch.abs(pred_y - batch_y))
			#l1_norm = 1e-4*torch.norm(odefunc.C.weight, p=1)
			#loss += l1_norm
			loss.backward(retain_graph = True)
			optimizer.step()
			proximal(odefunc.C.weight, lam=l1_reg, eta=0.01)
		scheduler.step()
		print(itr, loss.item())	
	print(odefunc.C.weight)

def test(odefunc, odeint_method, test_data, t, ckpt_path, fig_save_path):
	odefunc.NFE = 0
	test_sol = np.zeros_like(test_data)
	test_data_loader = torch.utils.data.DataLoader(torch.tensor(test_data),batch_size=50)
	test_loss = 0
	batch_idx = 50
	for i, d in enumerate(test_data_loader):
		pred_y = odeint(odefunc, d[:,0,:], t, method=odeint_method).transpose(0,1)
		test_sol[batch_idx*i:batch_idx*(i+1),:,:] = pred_y.detach().numpy() 
		test_loss += torch.mean(torch.abs(pred_y - d)).item()
	print('test loss', test_loss)
	
	fig = plt.figure(figsize=(12,4))
	axes = []
	for i in range(2):
		axes.append(fig.add_subplot(1,2,i+1))
		axes[i].plot(t,test_data[0,:,i],lw=3,color='k')
		axes[i].plot(t,test_sol[0,:,i],lw=2,color='c',ls='--')
	
	save_file = os.path.join(fig_save_path,"image_best.png")
	plt.savefig(save_file)



def train_prune_pou(odefunc, lr, nepoch, niterbatch, lMB, nMB, odeint_method, train_data, val_data, t, ckpt_path, fig_save_path):
	train_data = torch.tensor(train_data).unsqueeze(0)
	
	parameters_to_prune = ((odefunc.net[1], "coeffs"),)
	
	params = odefunc.parameters()
	optimizer = optim.Adamax(params, lr=lr)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9987)
	
	best_loss = 1e30
	frame = 0 
	
	for itr in range(nepoch):
		#print('=={0:d}=='.format(itr))
		for i in range(niterbatch):
			optimizer.zero_grad()
			batch_y0, batch_t, batch_y_forward, batch_yT, batch_t_backward, batch_y_backward = utils.get_batch_two_single_time(train_data,t,lMB,nMB,reverse=False)
			pred_y_forward = odeint(odefunc, batch_y0, batch_t, method=odeint_method).transpose(0,1)
			pred_y_backward = odeint(odefunc, batch_yT, batch_t_backward, method=odeint_method).transpose(0,1)
			loss = torch.mean(torch.abs(pred_y_forward - batch_y_forward))
			loss += torch.mean(torch.abs(pred_y_backward - batch_y_backward))
			l1_norm = 1e-4*torch.norm(odefunc.net[1].coeffs, p=1)
			loss += l1_norm
			print(itr,i,loss.item(),l1_norm.item())
			loss.backward()
			optimizer.step()
			if itr >= 5:
				prune.global_unstructured(parameters_to_prune, pruning_method=ThresholdPruning, threshold=1e-6)
		#scheduler.step()
	
		print(itr, loss.item())	
		print(odefunc.net[1].coeffs)
		with torch.no_grad():
			val_loss = 0

			parts = odefunc.net[1].getpoulayer(t)
			pred_y = odeint(odefunc, train_data[:,0,:], t, method='rk4').transpose(0,1)
			val_loss = torch.mean(torch.abs(pred_y - train_data)).item()
			print('val loss', val_loss)
				
			if best_loss > val_loss:
				print('saving...', val_loss)
				torch.save({'state_dict': odefunc.state_dict(),}, ckpt_path)
				best_loss = val_loss 

			plt.figure()
			plt.tight_layout()
			save_file = os.path.join(fig_save_path,"image_{:03d}.png".format(frame))
			fig = plt.figure(figsize=(12,4))
			axes = []
			for i in range(2):
				axes.append(fig.add_subplot(1,3,i+1))
				
				axes[i].plot(t,train_data[0,:,i].detach().numpy(),lw=2,color='k')
				axes[i].plot(t,pred_y.detach().numpy()[0,:,i],lw=2,color='c',ls='--')

			axes.append(fig.add_subplot(1,3,3))
			for i in range(parts.shape[1]):
				axes[2].plot(t, parts[:,i].detach().numpy())

			plt.savefig(save_file)
			plt.close(fig)
			plt.close('all')
			plt.clf()
			frame += 1
	ckpt = torch.load(ckpt_path)
	odefunc.load_state_dict(ckpt['state_dict'])
	
	prune.remove(odefunc.net[1], 'coeffs')
	torch.save({'state_dict': odefunc.state_dict(),}, ckpt_path)
