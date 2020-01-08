from __future__ import print_function

import os
import ast
import re
from shutil import rmtree
from datetime import datetime
from collections import defaultdict

from itertools import islice
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import json
import yaml
from easydict import EasyDict as edict

from IPython.display import clear_output
from IPython.core.debugger import set_trace
from tqdm import tqdm_notebook

import torch
from torch import nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter


from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from utils import vis_batch, save_dict, save_batch, collate_fn, load_config

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn = True
device = torch.cuda.current_device()

config = load_config('config.yaml')
opt =  edict({"lr_policy":config.lr_policy,
                        "epoch_count":config.epoch_count,
                        "niter_decay":config.niter_decay,
                        "lr_decay_iters":config.lr_decay_iters,
                        "niter":config.niter})

train_set = FashionEdgesDataset(config.dataset)
training_data_loader = DataLoader(dataset=train_set,
                                  batch_size=config.batch_size, 
                                  collate_fn = collate_fn,
                                  shuffle=True)
N = len(training_data_loader)


netG = define_G(config.input_nc, config.output_nc, config.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
netD = define_D(config.input_nc + config.output_nc, config.ndf, 'basic', gpu_id=device)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
netG_scheduler = get_scheduler(optimizerG, opt)
netD_scheduler = get_scheduler(optimizerD, opt)

continue_training = config.continue_training

exp_path = './logs/' + config.comment + datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
writer = SummaryWriter(os.path.join(exp_path, "tb"))

if not os.path.isdir(exp_path):
    os.makedirs(exp_path)

if continue_training:
	checkpoints_path = os.path.join(exp_path, 'checkpoints')
    last_exp = os.listdir(checkpoints_path)[-1]
    start_epoch = int(last_exp)
    print ("Loaded checkpoint from epoch", start_epoch)
    
    # load weights
    state_dict = torch.load(os.path.join(checkpoints_path, last_exp))
    loss_d_history = state_dict['loss_d_history']
    loss_g_history = state_dict['loss_g_history']
    loss_g_l1_history  = state_dict['loss_g_l1_history']
    loss_g_gan_history  = state_dict['loss_g_gan_history']
    
    netG.load_state_dict(state_dict['Generator'])
    optimizerG.load_state_dict(state_dict['OptGenerator'])
    netD.load_state_dict(state_dict['Discriminator'])
    optimizerD.load_state_dict(state_dict['OptDiscriminator'])
else:
    start_epoch = 0
    loss_d_history = []
    loss_g_history = []
    loss_g_l1_history  = []
    loss_g_gan_history  = []


for epoch in range(start_epoch, config.niter + config.niter_decay + 1):
    metric_dict = defaultdict(list)
    for iteration, batch in enumerate(training_data_loader, 1):
        
        if batch is None:
            continue
        
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = netG(real_a)

        ######################
        # (1) Update D network
        ######################
        print ('Update D network, epoch {0}, iter: {1}'.format(epoch, iteration))

        optimizerD.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = netD.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        loss_d.backward()
        metric_dict['loss_d'].append(loss_d.item())
       
        optimizerD.step()

        ######################
        # (2) Update G network
        ######################
        print ('Update G network, epoch {0}, iter: {1}'.format(epoch, iteration))

        optimizerG.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * lamb
        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        
        metric_dict['loss_g'].append(loss_g.item())
        metric_dict['loss_g_l1'].append(loss_g_l1.item())
        metric_dict['loss_g_gan'].append(loss_g_gan.item())
        
        optimizerG.step()
    	
        for title, value in metric_dict.items():
	        writer.add_scalar(f"{name}/{title}", value[-1], n_iters_total)

        if iteration%config.VIS_FREQ==0:
                samples = vis_batch(fake_b.detach(), config.n_images_vis)
                writer.add_image(f"keypoints_vis/{batch_i}", samples, global_step=n_iters_total)

        #checkpoint
        if iteration%config.SAVE_FREQ==0:
            save(exp_path, 
            	N*epoch + iteration,
            	netD,
            	netG,
            	optimizerD,
            	optimizerG,
            	loss_d_history,
            	loss_g_history,
            	loss_g_l1_history,
            	loss_g_gan_history)
    
    # dump to tensorboard per-epoch stats
    for title, value in metric_dict.items():
        writer.add_scalar(f"{title}_epoch", np.mean(value), epoch)

    update_learning_rate(netG_scheduler, optimizerG)
    update_learning_rate(netD_scheduler, optimizerD)

        

    
