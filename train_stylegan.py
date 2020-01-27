# code from https://github.com/tomguluson92/StyleGAN_PyTorch

import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter


from networks_stylegan import StyleGenerator, StyleDiscriminator
from loss import gradient_penalty, R1Penalty, R2Penalty
from opts import TrainOptions, INFO

from torchvision.utils import save_image
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random
import torch
import os
import sys
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace

from utils import vis_batch, save_batch, collate_fn, load_config
from dataset import FashionEdgesDataset

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Hyper-parameters
CRITIC_ITER = 5
SAVE_SAMPLE_FREQ = 100
device = torch.cuda.current_device()
opts = TrainOptions().parse()
config = load_config('config.yaml')

# Create the model
start_epoch = 0
G = StyleGenerator()
D = StyleDiscriminator()
G.to(opts.device)
D.to(opts.device)

# Create dataset
train_set = FashionEdgesDataset(config.dataset.paths, 
                                config.resolution, 
                                only_edge = True,
                                pad_resolution = [1024,1024])

loader = DataLoader(dataset=train_set,
                      batch_size=config.batch_size, 
                      collate_fn = collate_fn,
                      shuffle=True)
N = len(loader)


# Create the criterion, optimizer and scheduler
optim_D = optim.Adam(D.parameters(), lr=0.00001, betas=(0.5, 0.999))
optim_G = optim.Adam(G.parameters(), lr=0.00001, betas=(0.5, 0.999))
scheduler_D = optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)
scheduler_G = optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)


# create stuff
exp_path = './stylegan_logs/' + "@" + datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
checkpoints_path = os.path.join(exp_path, 'checkpoints')
images_path = os.path.join(exp_path, 'images')
writer = SummaryWriter(os.path.join(exp_path, "tb"))

if not os.path.isdir(exp_path):
    os.makedirs(exp_path)
if not os.path.isdir(checkpoints_path):
    os.makedirs(checkpoints_path)
if not os.path.isdir(images_path):
    os.makedirs(images_path)

print ('Experiment directory created:', exp_path)

# Train
fix_z = torch.randn([opts.batch_size, 512]).to(opts.device)
softplus = nn.Softplus()
for ep in range(start_epoch, opts.epoch):
    bar = tqdm(loader)
    metric_dict = defaultdict(list)
    for i, real_img in enumerate(bar):
        if real_img is None:
            continue
        n_iters_total = ep * N + i
        # =======================================================================================================
        #   (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # =======================================================================================================
        # Compute adversarial loss toward discriminator
        D.zero_grad()
        real_img = real_img.to(opts.device) # NoneType sometines
        real_logit = D(real_img)
        fake_img = G(torch.randn([real_img.size(0), 512]).to(opts.device))
        fake_logit = D(fake_img.detach())
        d_loss = softplus(fake_logit).mean()
        d_loss = d_loss + softplus(-real_logit).mean()

        if opts.r1_gamma != 0.0:
            r1_penalty = R1Penalty(real_img.detach(), D)
            d_loss = d_loss + r1_penalty * (opts.r1_gamma * 0.5)

        if opts.r2_gamma != 0.0:
            r2_penalty = R2Penalty(fake_img.detach(), D)
            d_loss = d_loss + r2_penalty * (opts.r2_gamma * 0.5)

        metric_dict['loss_d'].append(d_loss.item())


        # Update discriminator
        d_loss.backward()
        optim_D.step()

        # =======================================================================================================
        #   (2) Update G network: maximize log(D(G(z)))
        # =======================================================================================================
        if i % CRITIC_ITER == 0:
            G.zero_grad()
            fake_logit = D(fake_img)
            g_loss = softplus(-fake_logit).mean()
            metric_dict['loss_g'].append(g_loss.item())

            # Update generator
            g_loss.backward()
            optim_G.step()

        # Output training stats
        bar.set_description("Epoch {} [{}, {}] [G]: {} [D]: {}".format(ep, i+1, len(loader), metric_dict['loss_g'][-1], metric_dict['loss_d'][-1]))

        for title, value in metric_dict.items():
            writer.add_scalar(f"train/{title}", value[-1], n_iters_total)

        # Check how the generator is doing by saving G's output on fixed_noise
        if i%SAVE_SAMPLE_FREQ == 0:
            with torch.no_grad():
                fake_img = G(fix_z).detach().cpu()
                save_image(fake_img, os.path.join(images_path, str(n_iters_total) + '.png'), nrow=4, normalize=True)


    # Save model
    state = {
        'G': G.state_dict(),
        'D': D.state_dict(),
        'G_opt':optim_G.state_dict(),
        'D_opt':optim_D.state_dict(),
        'start_epoch': ep,
    }
    torch.save(state, os.path.join(checkpoints_path, 'latest.pth'))

    scheduler_D.step()
    scheduler_G.step()

    # dump to tensorboard per-epoch stats
    for title, value in metric_dict.items():
        writer.add_scalar(f"epoch/{title}_epoch", np.mean(value), ep)
