from __future__ import print_function

import os
import sys
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


from IPython.core.debugger import set_trace

import torch
from torch import nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from utils import vis_batch, save_batch, collate_fn, load_config
from dataset import FashionEdgesDataset

config = load_config('config.yaml')

device = torch.cuda.current_device()

train_set = FashionEdgesDataset(config.dataset.paths, config.resolution)
training_data_loader = DataLoader(dataset=train_set,
                                  batch_size=config.batch_size, 
                                  collate_fn = collate_fn,
                                  shuffle=True)
N = len(training_data_loader)


netG = define_G(config.input_nc, config.output_nc, config.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
netD = define_D(config.input_nc + config.output_nc, config.ndf, 'basic', gpu_id=device)

s_total = 0
for param in netG.parameters():
    s_total+=param.numel()
print ('Params in Generator:', s_total)

s_total = 0
for param in netD.parameters():
    s_total+=param.numel() 
print ('Params in Discriminator:', s_total)


criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

if config.use_scheduler:
    netG_scheduler = get_scheduler(optimizerG, config)
    netD_scheduler = get_scheduler(optimizerD, config)

# paths
exp_path = './logs/' + config.comment + "@" + datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
checkpoints_path = os.path.join(exp_path, 'checkpoints')
images_path = os.path.join(exp_path, 'images')
results_path = os.path.join(exp_path, 'results')
if not os.path.isdir(exp_path):
    os.makedirs(exp_path)
if not os.path.isdir(checkpoints_path):
    os.makedirs(checkpoints_path)
if not os.path.isdir(images_path):
    os.makedirs(images_path)
if not os.path.isdir(results_path):
    os.makedirs(results_path)               

writer = SummaryWriter(os.path.join(exp_path, "tb"))
print ('Experiment directory created:', exp_path)

start_epoch = 0
continue_training = config.continue_training
if continue_training:
    last_exp = sorted(os.listdir(checkpoints_path))[-1]
    start_epoch = int(last_exp)
    print ("Loaded checkpoint from epoch", start_epoch)
    
    # load weights
    state_dict = torch.load(os.path.join(checkpoints_path, last_exp))
    netG.load_state_dict(state_dict['Generator'])
    optimizerG.load_state_dict(state_dict['OptGenerator'])
    netD.load_state_dict(state_dict['Discriminator'])
    optimizerD.load_state_dict(state_dict['OptDiscriminator'])


for epoch in range(start_epoch, config.niter + config.niter_decay + 1):
    print ('EPOCH', epoch)
    metric_dict = defaultdict(list)
    for iteration, batch in enumerate(training_data_loader):
        n_iters_total = epoch * N + iteration
        if batch is None:
            print ('NONE batch')
            continue

        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = netG(real_a)

        ######################
        # (1) Update D network
        ######################

        optimizerD.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab.detach()) # detach - don't penalize Generator
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

        optimizerG.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * config.lamb
        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        
        metric_dict['loss_g'].append(loss_g.item())
        metric_dict['loss_g_l1'].append(loss_g_l1.item())
        metric_dict['loss_g_gan'].append(loss_g_gan.item())
        
        optimizerG.step()
    	
        for title, value in metric_dict.items():
	        writer.add_scalar(f"train/{title}", value[-1], n_iters_total)

        if iteration%config.VIS_FREQ==0:
                generated_samples = vis_batch(fake_b.detach(), config.n_images_vis) 
                edges_samples = vis_batch(real_a.detach(), config.n_images_vis) 

                writer.add_image(f"{n_iters_total}/images",
                                 generated_samples.transpose(2,0,1), # hcw -> chw
                                  global_step=n_iters_total) 

                writer.add_image(f"{n_iters_total}/edges",
                                 edges_samples.transpose(2,0,1),
                                 global_step=n_iters_total) 

                # plt.imsave(os.path.join(images_path, '{}.jpg'.format(n_iters_total)), generated_samples)
                save_batch(fake_b.detach(), folder=results_path)


    #checkpoint
    if epoch%config.SAVE_FREQ==0 and epoch > 0:
        print ('Saving at epoch:', epoch)
        dict_to_save = {"iteration":n_iters_total,
                    	"netD":netD.state_dict(),
                    	"netG":netG.state_dict(),
                    	"optimizerD":optimizerD.state_dict(),
                    	"optimizerG":optimizerG.state_dict()}

        path = os.path.join(checkpoints_path, 'checkpoint_{:03}'.format(epoch))
        torch.save(dict_to_save, path)
    
    # dump to tensorboard per-epoch stats
    for title, value in metric_dict.items():
        writer.add_scalar(f"epoch/{title}_epoch", np.mean(value), epoch)
    
    writer.add_scalar(f"epoch/lr_G",
                      optimizerG.state_dict()['param_groups'][0]['lr'],
                      epoch)

    writer.add_scalar(f"epoch/lr_D", 
                      optimizerD.state_dict()['param_groups'][0]['lr'], 
                      epoch)    


    if config.use_scheduler and epoch >= config.niter:    
        update_learning_rate(netG_scheduler, optimizerG)
        update_learning_rate(netD_scheduler, optimizerD)

        

    
