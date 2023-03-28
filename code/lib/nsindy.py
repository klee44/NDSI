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

def proximal(w, lam=0.1, eta=0.1, thresh=0):
	"""Proximal step"""
	# w shape dim \times number of dicts
	#dim, ndicts = w.shape
	#tmp = w.flatten()	
	#print(lam)
	#print(w)
	alpha = torch.clamp(torch.abs(w) - lam * eta, min=thresh)
	#print(alpha)
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



def train_prune_pou(odefunc, lr, nepoch, niterbatch, lMB, nMB, odeint_method, train_data, val_data, t, ckpt_path, fig_save_path, frame_index):
	train_data = torch.tensor(train_data).unsqueeze(0)
	
	parameters_to_prune = ((odefunc.net[1], "coeffs"),)
	
	params_pou = nn.ParameterList([])
	params_dict = nn.ParameterList([]) 
	for n, p in odefunc.named_parameters():
		if n == 'xrbf': 
			print(n, p)
			params_pou.append(p)
		else:
			params_dict.append(p)
	optimizer = optim.Adamax([{'params': params_pou}, {'params':params_dict}], lr=lr[0])
	optimizer.param_groups[0]['lr'] = lr[1] 
	#params = odefunc.parameters()
	#optimizer = optim.Adamax(params, lr=lr)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9987)
	
	best_loss = 1e30
	frame = frame_index 
	
	for itr in range(nepoch):
		#print('=={0:d}=='.format(itr))
		for i in range(niterbatch):
			optimizer.zero_grad()
			batch_y0, batch_t, batch_y_forward, batch_yT, batch_t_backward, batch_y_backward = utils.get_batch_two_single_time(train_data,t,lMB,nMB,reverse=False)
			pred_y_forward = odeint(odefunc, batch_y0, batch_t, method=odeint_method).transpose(0,1)
			pred_y_backward = odeint(odefunc, batch_yT, batch_t_backward, method=odeint_method).transpose(0,1)
			loss = torch.mean(torch.abs(pred_y_forward - batch_y_forward))
			loss += torch.mean(torch.abs(pred_y_backward - batch_y_backward))
			loss += torch.mean(torch.abs(pred_y_backward - pred_y_backward.flip(1)))
			l1_norm = 5e-4*torch.norm(odefunc.net[1].coeffs, p=1)
			loss += l1_norm
			l2_norm = 5e-3*torch.sum(torch.norm(odefunc.net[1].coeffs, p=2, dim=0))
			#loss += l2_norm
			eps_l1_norm = 1e-4*torch.sum(torch.exp(odefunc.net[1].epsrbf))
			#loss += eps_l1_norm
			print(itr,i,loss.item(),l1_norm.item(),l2_norm.item(),eps_l1_norm.item())
			loss.backward()
			optimizer.step()
			if itr >= 5:
				prune.global_unstructured(parameters_to_prune, pruning_method=ThresholdPruning, threshold=1e-6)
		if itr < 1700:
			scheduler.step()
	
		print(itr, loss.item())	
		print(odefunc.net[1].coeffs)
		'''
		with torch.no_grad():
			val_loss = 0

			parts = odefunc.net[1].getpoulayer(t/t[-1])
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
				axes[2].plot(t/t[-1], parts[:,i].detach().numpy())

			plt.savefig(save_file)
			plt.close(fig)
			plt.close('all')
			plt.clf()
			frame += 1
		'''
	ckpt = torch.load(ckpt_path)
	odefunc.load_state_dict(ckpt['state_dict'])
	
	prune.remove(odefunc.net[1], 'coeffs')
	torch.save({'state_dict': odefunc.state_dict(),}, ckpt_path)

'''
def proximal_pou(w, lam=0.1, eta=0.1):
	"""Proximal step"""
	# w shape: (number of dicts * 2) \times npart \times 1
	alpha = torch.clamp(torch.abs(w) - lam * eta, min=0)
	w.data = torch.sign(w) * alpha#).reshape((dim, ndicts))
'''

def proximal_pou(w, lam=0.1, eta=0.1):
	"""Proximal step"""
	# w shape: (number of dicts * 2) \times npart \times 1
	w_tmp = torch.sum(w**2, dim=0).pow(0.5) - lam * eta
	alpha = torch.clamp(w_tmp, 0)
	w.data = torch.nn.functional.normalize(w, dim=0) * alpha[None, :, :]
	'''
        wadj = w.view(func.dims[0], -1, func.dims[0])  # [j, m1, i]
        tmp = torch.sum(wadj**2, dim=1).pow(0.5) - lam * eta
        alpha = torch.clamp(tmp, min=0)
        v = torch.nn.functional.normalize(wadj, dim=1) * alpha[:, None, :]
        w.data = v.view(-1, func.dims[0])
	'''
def train_pou_ISTA(odefunc, lr, nepoch, niterbatch, lMB, nMB, odeint_method, train_data, t, l1_reg, ckpt_path, fig_save_path, frame_index):
	train_data = torch.tensor(train_data).unsqueeze(0)
	
	params_pou = nn.ParameterList([])
	params_dict = nn.ParameterList([]) 
	for n, p in odefunc.named_parameters():
		if n[-3:] == 'rbf': 
			print(n, p)
			params_pou.append(p)
		else:
			params_dict.append(p)
	optimizer = optim.Adamax([{'params': params_pou}, {'params':params_dict}], lr=lr[0])
	optimizer.param_groups[0]['lr'] = lr[1] #1e-4#1e-3
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9987)
	
	best_loss = 1e30
	frame = frame_index 
	
	for itr in range(nepoch):
		#print('=={0:d}=='.format(itr))
		with torch.no_grad():
			parts = odefunc.net[1].getpoulayer(t/t[-1])
			plt.figure()
			plt.tight_layout()
			save_file = os.path.join(fig_save_path,"image_{:03d}.png".format(frame))
			fig = plt.figure(figsize=(4,4))
			axes = []
			axes.append(fig.add_subplot(1,1,1))
			for i in range(parts.shape[1]):
				axes[0].plot(t/t[-1], parts[:,i].detach().numpy())

			plt.savefig(save_file)
			plt.close(fig)
			plt.close('all')
			plt.clf()
			frame += 1
		for i in range(niterbatch):
			optimizer.zero_grad()
			batch_y0, batch_t, batch_y_forward, batch_yT, batch_t_backward, batch_y_backward = utils.get_batch_two_single_time(train_data,t,lMB,nMB,reverse=False)
			pred_y_forward = odeint(odefunc, batch_y0, batch_t, method=odeint_method).transpose(0,1)
			pred_y_backward = odeint(odefunc, batch_yT, batch_t_backward, method=odeint_method).transpose(0,1)
			loss = torch.mean(torch.abs(pred_y_forward - batch_y_forward))
			loss += torch.mean(torch.abs(pred_y_backward - batch_y_backward))
			loss += torch.mean(torch.abs(pred_y_backward - pred_y_backward.flip(1)))
			#l1_norm = torch.sum(odefunc.net[1].epsrbf * l1_reg[1])
			#print(loss, l1_norm)
			#loss += l1_norm
			loss.backward(retain_graph = True)
			optimizer.step()
			proximal(odefunc.net[1].coeffs, lam=l1_reg[0], eta=0.01)
			proximal_pou(odefunc.net[1].coeffs, lam=l1_reg[1], eta=0.01)
		#scheduler.step()
		print(itr, loss.item())	
	print(odefunc.net[1].coeffs)	


def train_pou_ISTA_alt(odefunc, lr, nepoch, niterbatch, lMB, nMB, odeint_method, train_data, t, l1_reg, ckpt_path, fig_save_path, frame_index):
	train_data = torch.tensor(train_data).unsqueeze(0)
	
	params_pou = nn.ParameterList([])
	params_dict = nn.ParameterList([]) 
	for n, p in odefunc.named_parameters():
		if n[-3:] == 'rbf': 
			print(n, p)
			params_pou.append(p)
		else:
			params_dict.append(p)
	#optimizer = optim.Adamax([{'params': params_pou}, {'params':params_dict}], lr=lr[0])
	#optimizer.param_groups[0]['lr'] = lr[1] #1e-4#1e-3
	optimizer_coeff = optim.Adamax([{'params':params_dict}], lr=lr[0])
	optimizer_pou = optim.Adamax([{'params':params_pou}], lr=lr[1])

	#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9987)
	
	best_loss = 1e30
	frame = frame_index 
	
	for itr in range(nepoch):
		#print('=={0:d}=='.format(itr))
		with torch.no_grad():
			parts = odefunc.net[1].getpoulayer(t/t[-1])
			plt.figure()
			plt.tight_layout()
			save_file = os.path.join(fig_save_path,"image_{:03d}.png".format(frame))
			fig = plt.figure(figsize=(4,4))
			axes = []
			axes.append(fig.add_subplot(1,1,1))
			for i in range(parts.shape[1]):
				axes[0].plot(t/t[-1], parts[:,i].detach().numpy())

			plt.savefig(save_file)
			plt.close(fig)
			plt.close('all')
			plt.clf()
			frame += 1
		for i in range(niterbatch):
			optimizer_coeff.zero_grad()
			batch_y0, batch_t, batch_y_forward, batch_yT, batch_t_backward, batch_y_backward = utils.get_batch_two_single_time(train_data,t,lMB,nMB,reverse=False)
			pred_y_forward = odeint(odefunc, batch_y0, batch_t, method=odeint_method).transpose(0,1)
			pred_y_backward = odeint(odefunc, batch_yT, batch_t_backward, method=odeint_method).transpose(0,1)
			loss = torch.mean(torch.abs(pred_y_forward - batch_y_forward))
			loss += torch.mean(torch.abs(pred_y_backward - batch_y_backward))
			loss += torch.mean(torch.abs(pred_y_backward - pred_y_backward.flip(1)))
			#l1_norm = torch.sum(odefunc.net[1].epsrbf * l1_reg[1])
			#print(loss, l1_norm)
			#loss += l1_norm
			loss.backward(retain_graph = True)
			optimizer_coeff.step()
			proximal(odefunc.net[1].coeffs, lam=l1_reg[0], eta=0.01)
			proximal_pou(odefunc.net[1].coeffs, lam=l1_reg[1], eta=0.01)
		for i in range(niterbatch):
			optimizer_pou.zero_grad()
			batch_y0, batch_t, batch_y_forward, batch_yT, batch_t_backward, batch_y_backward = utils.get_batch_two_single_time(train_data,t,lMB,nMB,reverse=False)
			pred_y_forward = odeint(odefunc, batch_y0, batch_t, method=odeint_method).transpose(0,1)
			pred_y_backward = odeint(odefunc, batch_yT, batch_t_backward, method=odeint_method).transpose(0,1)
			loss = torch.mean(torch.abs(pred_y_forward - batch_y_forward))
			loss += torch.mean(torch.abs(pred_y_backward - batch_y_backward))
			loss += torch.mean(torch.abs(pred_y_backward - pred_y_backward.flip(1)))
			#l1_norm = torch.sum(odefunc.net[1].epsrbf * l1_reg[1])
			#print(loss, l1_norm)
			#loss += l1_norm
			loss.backward(retain_graph = True)
			optimizer_pou.step()
			proximal(odefunc.net[1].coeffs, lam=l1_reg[0], eta=0.01)
			proximal_pou(odefunc.net[1].coeffs, lam=l1_reg[1], eta=0.01)
		#scheduler.step()
		print(itr, loss.item())	
	print(odefunc.net[1].coeffs)	



def train_pou_ISTA_alt_mlp(odefunc, lr, nepoch, niterbatch, lMB, nMB, odeint_method, train_data, t, l1_reg, ckpt_path, fig_save_path, frame_index):
	train_data = torch.tensor(train_data).unsqueeze(0)
	
	params_pou = nn.ParameterList([])
	params_dict = nn.ParameterList([]) 
	for n, p in odefunc.named_parameters():
		if n[-3:] == 'rbf': 
			print(n, p)
			params_pou.append(p)
		else:
			params_dict.append(p)
	#optimizer = optim.Adamax([{'params': params_pou}, {'params':params_dict}], lr=lr[0])
	#optimizer.param_groups[0]['lr'] = lr[1] #1e-4#1e-3
	optimizer_coeff = optim.Adamax([{'params':params_dict}], lr=lr[0])
	optimizer_pou = optim.Adamax([{'params':params_pou}], lr=lr[1])

	#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9987)
	
	best_loss = 1e30
	frame = frame_index 
	
	for itr in range(nepoch):
		#print('=={0:d}=='.format(itr))
		with torch.no_grad():
			parts = odefunc.getpoulayer(t/t[-1])
			plt.figure()
			plt.tight_layout()
			save_file = os.path.join(fig_save_path,"image_{:03d}.png".format(frame))
			fig = plt.figure(figsize=(4,4))
			axes = []
			axes.append(fig.add_subplot(1,1,1))
			for i in range(parts.shape[1]):
				axes[0].plot(t/t[-1], parts[:,i].detach().numpy())

			plt.savefig(save_file)
			plt.close(fig)
			plt.close('all')
			plt.clf()
			frame += 1
		for i in range(niterbatch):
			optimizer_coeff.zero_grad()
			batch_y0, batch_t, batch_y_forward, batch_yT, batch_t_backward, batch_y_backward = utils.get_batch_two_single_time(train_data,t,lMB,nMB,reverse=False)
			pred_y_forward = odeint(odefunc, batch_y0, batch_t, method=odeint_method).transpose(0,1)
			pred_y_backward = odeint(odefunc, batch_yT, batch_t_backward, method=odeint_method).transpose(0,1)
			loss = torch.mean(torch.abs(pred_y_forward - batch_y_forward))
			loss += torch.mean(torch.abs(pred_y_backward - batch_y_backward))
			loss += torch.mean(torch.abs(pred_y_backward - pred_y_backward.flip(1)))
			#l1_norm = torch.sum(odefunc.net[1].epsrbf * l1_reg[1])
			#print(loss, l1_norm)
			#loss += l1_norm
			loss.backward(retain_graph = True)
			optimizer_coeff.step()
			proximal(odefunc.net[1].coeffs, lam=l1_reg[0], eta=0.01)
			proximal_pou(odefunc.net[1].coeffs, lam=l1_reg[1], eta=0.01)
		for i in range(niterbatch):
			optimizer_pou.zero_grad()
			batch_y0, batch_t, batch_y_forward, batch_yT, batch_t_backward, batch_y_backward = utils.get_batch_two_single_time(train_data,t,lMB,nMB,reverse=False)
			pred_y_forward = odeint(odefunc, batch_y0, batch_t, method=odeint_method).transpose(0,1)
			pred_y_backward = odeint(odefunc, batch_yT, batch_t_backward, method=odeint_method).transpose(0,1)
			loss = torch.mean(torch.abs(pred_y_forward - batch_y_forward))
			loss += torch.mean(torch.abs(pred_y_backward - batch_y_backward))
			loss += torch.mean(torch.abs(pred_y_backward - pred_y_backward.flip(1)))
			#l1_norm = torch.sum(odefunc.net[1].epsrbf * l1_reg[1])
			#print(loss, l1_norm)
			#loss += l1_norm
			loss.backward(retain_graph = True)
			optimizer_pou.step()
			#proximal(odefunc.net[1].coeffs, lam=l1_reg[0], eta=0.01)
			proximal_pou(odefunc.net[1].coeffs, lam=l1_reg[1], eta=0.01)
		#scheduler.step()
		print(itr, loss.item())	
	print(odefunc.net[1].coeffs)	


def train_prune_pou_mlp(odefunc, lr, nepoch, niterbatch, lMB, nMB, odeint_method, train_data, val_data, t, ckpt_path, fig_save_path, frame_index):
	train_data = torch.tensor(train_data).unsqueeze(0)
	
	#parameters_to_prune = ((odefunc.net[1], "coeffs"),)
	
	params_pou = nn.ParameterList([])
	params_dict = nn.ParameterList([]) 
	for n, p in odefunc.named_parameters():
		if n == 'xrbf': 
			print(n, p)
			params_pou.append(p)
		else:
			params_dict.append(p)
	optimizer = optim.Adamax([{'params': params_pou}, {'params':params_dict}], lr=lr[0])
	optimizer.param_groups[0]['lr'] = lr[1] 
	#params = odefunc.parameters()
	#optimizer = optim.Adamax(params, lr=lr)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9987)
	
	best_loss = 1e30
	frame = frame_index 
	
	for itr in range(nepoch):
		#print('=={0:d}=='.format(itr))
		for i in range(niterbatch):
			optimizer.zero_grad()
			batch_y0, batch_t, batch_y_forward, batch_yT, batch_t_backward, batch_y_backward = utils.get_batch_two_single_time(train_data,t,lMB,nMB,reverse=False)
			pred_y_forward = odeint(odefunc, batch_y0, batch_t, method=odeint_method).transpose(0,1)
			pred_y_backward = odeint(odefunc, batch_yT, batch_t_backward, method=odeint_method).transpose(0,1)
			loss = torch.mean(torch.abs(pred_y_forward - batch_y_forward))
			loss += torch.mean(torch.abs(pred_y_backward - batch_y_backward))
			loss += torch.mean(torch.abs(pred_y_backward - pred_y_backward.flip(1)))
			l1_norm = 5e-4*torch.norm(odefunc.net[1].coeffs, p=1)
			loss += l1_norm
			l2_norm = 5e-3*torch.sum(torch.norm(odefunc.net[1].coeffs, p=2, dim=0))
			#loss += l2_norm
			eps_l1_norm = 1e-4*torch.sum(torch.exp(odefunc.net[1].epsrbf))
			#loss += eps_l1_norm
			print(itr,i,loss.item(),l1_norm.item(),l2_norm.item(),eps_l1_norm.item())
			loss.backward()
			optimizer.step()
			#if itr >= 5:
			#	prune.global_unstructured(parameters_to_prune, pruning_method=ThresholdPruning, threshold=1e-6)
		if itr < 1700:
			scheduler.step()
	
		print(itr, loss.item())	
		print(odefunc.net[1].coeffs)
		'''
		with torch.no_grad():
			val_loss = 0

			parts = odefunc.net[1].getpoulayer(t/t[-1])
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
				axes[2].plot(t/t[-1], parts[:,i].detach().numpy())

			plt.savefig(save_file)
			plt.close(fig)
			plt.close('all')
			plt.clf()
			frame += 1
		'''
	ckpt = torch.load(ckpt_path)
	odefunc.load_state_dict(ckpt['state_dict'])
	
	#prune.remove(odefunc.net[1], 'coeffs')
	torch.save({'state_dict': odefunc.state_dict(),}, ckpt_path)

def train_pou_ISTA_mlp(odefunc, lr, nepoch, niterbatch, lMB, nMB, odeint_method, train_data, t, l1_reg, ckpt_path, fig_save_path, frame_index):
	train_data = torch.tensor(train_data).unsqueeze(0)
	
	params_pou = nn.ParameterList([])
	params_dict = nn.ParameterList([]) 
	for n, p in odefunc.named_parameters():
		if n[-3:] == 'rbf': 
			print(n, p)
			params_pou.append(p)
		else:
			params_dict.append(p)
	optimizer = optim.Adamax([{'params': params_pou}, {'params':params_dict}], lr=lr[0])
	optimizer.param_groups[0]['lr'] = lr[1] #1e-4#1e-3
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9987)
	
	best_loss = 1e30
	frame = frame_index 
	
	for itr in range(nepoch):
		#print('=={0:d}=='.format(itr))
		with torch.no_grad():
			parts = odefunc.getpoulayer(t/t[-1])
			plt.figure()
			plt.tight_layout()
			save_file = os.path.join(fig_save_path,"image_{:03d}.png".format(frame))
			fig = plt.figure(figsize=(4,4))
			axes = []
			axes.append(fig.add_subplot(1,1,1))
			for i in range(parts.shape[1]):
				axes[0].plot(t/t[-1], parts[:,i].detach().numpy())

			plt.savefig(save_file)
			plt.close(fig)
			plt.close('all')
			plt.clf()
			frame += 1
		for i in range(niterbatch):
			optimizer.zero_grad()
			batch_y0, batch_t, batch_y_forward, batch_yT, batch_t_backward, batch_y_backward = utils.get_batch_two_single_time(train_data,t,lMB,nMB,reverse=False)
			pred_y_forward = odeint(odefunc, batch_y0, batch_t, method=odeint_method).transpose(0,1)
			pred_y_backward = odeint(odefunc, batch_yT, batch_t_backward, method=odeint_method).transpose(0,1)
			loss = torch.mean(torch.abs(pred_y_forward - batch_y_forward))
			loss += torch.mean(torch.abs(pred_y_backward - batch_y_backward))
			loss += torch.mean(torch.abs(pred_y_backward - pred_y_backward.flip(1)))
			#l1_norm = torch.sum(odefunc.net[1].epsrbf * l1_reg[1])
			#print(loss, l1_norm)
			#loss += l1_norm
			loss.backward(retain_graph = True)
			optimizer.step()
			#proximal(odefunc.net[1].coeffs, lam=l1_reg[0], eta=0.01)
			proximal_pou(odefunc.net[1].coeffs, lam=l1_reg[1], eta=0.01)
		#scheduler.step()
		print(itr, loss.item())	
	print(odefunc.net[1].coeffs)	
