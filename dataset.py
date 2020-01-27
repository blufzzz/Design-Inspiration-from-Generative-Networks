from utils import image2edges, edges2mask, tensor2numpy
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from skimage.transform import resize


def read_paths_file(images_paths_file):
    with open(images_paths_file) as file:
        images_paths = file.read().splitlines()
    return images_paths     

class FashionEdgesDataset(Dataset):
    def __init__(self, 
                 images_paths_file, 
                 resolution, 
                 check_corrupted=True, 
                 randomize_disc=True,
                 randomize_sigma = True, 
                 only_edge = False,
                 pad_resolution = None):
        
        self.check_corrupted = check_corrupted
        self.corrupted_images = set()
        self.randomize_disc = randomize_disc
        self.randomize_sigma = randomize_sigma
        self.resolution = resolution
        self.only_edge = only_edge
        self.pad_resolution = pad_resolution

        if type(images_paths_file) is str:
            self.images_paths = read_paths_file(images_paths_file)

        elif type(images_paths_file) is list:
            self.images_paths = []
            for file_path in images_paths_file:
                self.images_paths += read_paths_file(file_path)     

    def _is_appropriate(self, img, offset = 20):
        return np.all(img[:offset,:offset] > 250) and np.all(img[-offset:,-offset:] > 250)
    
    def __getitem__(self, idx):
        
        image_path = self.images_paths[idx]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.check_corrupted and not self._is_appropriate(img):
            self.corrupted_images.add(image_path)
            return None
        
        edges = image2edges(img, 
                            randomize_disc=self.randomize_disc, 
                            randomize_sigma = self.randomize_sigma)
        
        img = resize(img, self.resolution, anti_aliasing=True)
        edges = image2edges(img)
        img = img.transpose(2,0,1) #/ 255.

        if self.pad_resolution is not None:
            d1 = max(0,(self.pad_resolution[0] - self.resolution[0])//2)
            d2 = max(0,(self.pad_resolution[1] - self.resolution[1])//2)

            img = np.pad(img, ((0,0), (d1, d1), (d2, d2)), constant_values=255)
            edges = np.pad(edges, ((d1, d1), (d2, d2)), constant_values=0)

        
        if self.only_edge:
            return torch.tensor(edges, dtype=torch.float32).unsqueeze(0) 
        else:    
            return torch.tensor(edges, dtype=torch.float32).unsqueeze(0),\
                    torch.tensor(img, dtype=torch.float32)

    def __len__(self):
        return len(self.images_paths)