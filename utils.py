import re
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
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
                d = 2):
        '''
        image - np.array
        '''
        image_gray_rescaled = rgb2gray(image)
        edges = canny(image_gray_rescaled, 
                      sigma = sigma, 
                      low_threshold=low_thresh, 
                      high_threshold=high_thresh)
        if selem:
            selem = disk(d)
            edges = dilation(edges, selem)
  
        return edges.astype(float)


def edges2mask(edge, padding=10):

    h,w = edge.shape[:2]
    shape = flood(np.pad(edge, padding, mode='constant'),seed_point=(0, 0))

    shape = ~shape[padding:-padding, padding:-padding]
    return shape.astype(float)        



def tensor2numpy(img): 
    #[h,w,c] -> [c,h,w]
    return img.permute(1,2,0).cpu().numpy()
    
def vis_batch(batch, number):
    batch_size = batch.shape[0]
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

def save_dict(exp_path, dict_to_save, specification):

    path = os.path.join(exp_path,specification)
    torch.save(dict_to_save, path)
    
def save_batch(batch, folder = './results'):
    
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
    
    batch = [torch.stack([item[i] for item in items]) for i in range(len(items[0]))]

    return batch    