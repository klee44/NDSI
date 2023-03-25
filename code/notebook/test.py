import sys
sys.path.append("../")

import numpy as np
from data.data_gen import simulate_lv_cp
from lib.nsindy import train_pou_ISTA, train_pou_ISTA_alt, train_prune_pou, test
import lib.utils as utils
import random
from random import SystemRandom
import os

import torch
torch.set_default_dtype(torch.float64) 

seed = 4
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


save_path = 'experiments/'                                                                            
utils.makedirs(save_path)
experimentID = int(SystemRandom().random()*100000)                                                    
ckpt_path = os.path.join(save_path, "experiment_" + str(experimentID) + '.ckpt')                      
fig_save_path = os.path.join(save_path,"experiment_"+str(experimentID))                               
utils.makedirs(fig_save_path)
print(ckpt_path)  

train_data, t = simulate_lv_cp()

from lib.odefunc import ODEfuncPOUPoly
print(t[-1])
odefunc = ODEfuncPOUPoly(2, 2, npart=8, npoly=0, seqlen = t[-1])

l1_reg = []
l1_reg.append(.001)
l1_reg.append(.001)
lr = []
lr.append(1e-2)
lr.append(1e-2)
train_pou_ISTA_alt(odefunc, lr=lr, nepoch = 1000, niterbatch = 100, lMB = 100, nMB = 1, odeint_method="dopri5", train_data = train_data, t=t, l1_reg = l1_reg, ckpt_path = ckpt_path,fig_save_path=fig_save_path, frame_index=0)

#l1_reg = []
#l1_reg.append(.005)
#l1_reg.append(.005)
l1_reg[0] *= 1 / (torch.abs(odefunc.net[1].coeffs.data)**(0.5) + 1e-4)#torch.tensor(1/ (np.abs(tmp)**(0.5) + 1e-4))#1 / (torch.abs(odefunc.net[1].coeffs.data)**(0.5) + 1e-4)
l1_reg[1] *= 1 / (torch.sum(odefunc.net[1].coeffs**2, dim=0).pow(0.5)**.5 +1e-4 )#torch.tensor(1 / (np.sum(tmp**2, axis=0)**(0.5)**.1 + 1e-1)) #1 / (torch.sum(odefunc.net[1].coeffs**2, dim=0).pow(0.5)**4 + 1e-4)
print(l1_reg)


odefunc.net[1].reset_parameters()
odefunc.net[1].xrbf.data =  torch.linspace(0., 1., 2*8+1)[1:-1][::2]
odefunc.net[1].epsrbf.data = torch.log((.15/8)*torch.ones(8))
lr = []
lr.append(1e-2)
lr.append(1e-2)
train_pou_ISTA(odefunc, lr=lr, nepoch = 1000, niterbatch = 100, lMB = 100, nMB = 1, odeint_method="dopri5", train_data = train_data, t=t, l1_reg = l1_reg, ckpt_path = ckpt_path,fig_save_path=fig_save_path,frame_index=1000)
#train_ISTA(odefunc, 1e-2, 20, 100, 5, 10, "dopri5", train_data,t,l1_reg,ckpt_path,fig_save_path)

lr = []
lr.append(1e-2)
lr.append(1e-2)

odefunc.net[1].reset_parameters()
train_prune_pou(odefunc, lr, 10000, 100, 100, 1, "dopri5", train_data,train_data,t,ckpt_path,fig_save_path,frame_index=2000)

odefunc.get_dict()
print(odefunc.net[1].xrbf, odefunc.net[1].epsrbf)
