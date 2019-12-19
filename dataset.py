import os
import cv2
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import torch
import ast
from utils import image2edges, edges2mask,tensor2numpy
from skimage.transform import resize


class FashionEdgesDataset(Dataset):
    def __init__(self, 
                 images_fold, 
                 attr_file=None, 
                 check_corrupted=False,
                 size = (128,128),
                 return_mask=True):
        
        self.corrupted_images = set()
        self.check_corrupted = check_corrupted
        self.images_fold = images_fold
        self.return_mask = return_mask
        self.size = size
        
        if attr_file is not None:
            images2attr_dict = {}
            with open(attr_file, 'r') as f:
                dicts = f.readlines()
            for d in tqdm_notebook(dicts):
                dct = ast.literal_eval(d)
                image_name = dct['image']
                
                if check_corrupted:
                    img = Image.open(os.path.join(self.images_fold, image_name))
                    if not self._is_appropriate(np.array(img)):
                        self.corrupted_images.add(image_name)
                        continue

                images2attr_dict[image_name] = dct['attributes']
                
            self.images2attr = images2attr_dict
            self.images_names = list(images2attr_dict.keys())
        else:
            self.images_names = os.listdir(self.images_fold)
    
    def _is_appropriate(self, img, thresh = 10):
    
        return np.all(img[:thresh,:thresh] == 1)
    
    def __getitem__(self, idx):
        
        image_name = self.images_names[idx]
        img = cv2.imread(os.path.join(self.images_fold, image_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.

        if not self.check_corrupted and not self._is_appropriate(np.array(img)):
            return None

        edges_np = image2edges(img)
        img = resize(img, self.size, anti_aliasing=True)
        img = ToTensor()(img)

        mask = None
        if self.return_mask:
            mask_np = edges2mask(edges_np)
            mask_np = resize(mask_np, self.size, anti_aliasing=False)
            mask_np[mask_np > 0.] = 1.
            mask = torch.tensor(mask_np, dtype=torch.float64).unsqueeze(0)

        edges_np = resize(edges_np, self.size, anti_aliasing=False)
        edges_np[edges_np > 0.] = 1.
        edges = torch.tensor(edges_np, dtype=torch.float64).unsqueeze(0)
        
        return edges, img, mask

    def __len__(self):
        return len(self.images_names)


  