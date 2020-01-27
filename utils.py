import re
import os
import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict
from skimage.feature import canny
from skimage.morphology import dilation, disk, square
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import flood

def image2edges(image, 
                low_thresh=0.01, 
                high_thresh=0.2, 
                sigma=0.5, 
                selem=True, 
                d = 2,
                randomize_disc=True,
                randomize_sigma=True):
        '''
        image - np.array
        '''
        if randomize_sigma:
            sigma = np.random.choice([1,2,4,6], 1)[0]
        image_gray_rescaled = rgb2gray(image)
        edges = canny(image_gray_rescaled, 
                      sigma = sigma, 
                      low_threshold=low_thresh, 
                      high_threshold=high_thresh)
        if selem:
            if randomize_disc:
                d = np.random.choice([2,4,6,8], 1)[0]
            selem = disk(d)
            edges = dilation(edges, selem)
  
        return edges.astype(float)


def edges2mask(edge, padding=10):

    h,w = edge.shape[:2]
    shape = flood(np.pad(edge, padding, mode='constant'),seed_point=(0, 0))

    shape = ~shape[padding:-padding, padding:-padding]
    return shape.astype(float)        


def calc_gradient_penalty(netD, real_data, fake_data):
    batch_size = real_data.shape[0]
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous().view(batch_size, 3, 128, 128)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def tensor2numpy(img):  
    #[h,w,c] -> [c,h,w]
    return img.permute(1,2,0).cpu().numpy()
    
def vis_batch(batch, number):

    batch_size = batch.shape[0]
    if number < 4:
        l = min(batch_size,number) 
        fig, axes = plt.subplots(nrows = 1 , ncols = l, figsize = (5*l,5))
    else:    
        l = min(int(np.sqrt(batch_size)),int(np.sqrt(number))) 
        fig, axes = plt.subplots(nrows = l , ncols = l, figsize = (2*l,2*l))
    
    iterable = [axes] if l == 1 else axes.flatten()
    for i, ax in enumerate(iterable):

        img_i = batch[i]
        rgb = (img_i.shape[0] == 3)
        img_i = tensor2numpy(img_i) if rgb else img_i[0].cpu().numpy()
        ax.imshow(img_i)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.close('all')    
    return fig_to_array(fig)    
    
def save_batch(batch, folder = './results'):

    '''
    batch - tensors
    '''
    
    for tensor in batch:
        names = os.listdir(folder)
        last_number = 0 if names == [] else int(re.findall('\d+', \
                                                sorted(names, key=lambda x: int(re.findall('\d+',x)[0]))[-1])[0])
        plt.figure(figsize=(5,5))
        plt.imshow(tensor2numpy(tensor.detach()))
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(folder,'{}.jpg'.format(last_number+1)))
    
    
def collate_fn(items):
    items = list(filter(lambda x: x is not None, items))

    if len(items) == 0:
        print("All items in batch are None")
        return None

    if type(items[0]) is torch.Tensor:    
        return torch.stack(items)
    else:
        batch = [torch.stack([item[i] for item in items]) for i in range(len(items[0]))]

    return batch   


def load_config(path):
    with open(path) as fin:
        config = edict(yaml.safe_load(fin))

    return config     


def fig_to_array(fig):
    fig.canvas.draw()
    fig_image = np.array(fig.canvas.renderer._renderer)

    return fig_image[...,:3]
