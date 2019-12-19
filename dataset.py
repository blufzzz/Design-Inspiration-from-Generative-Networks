import os
import skimage
import imageio
from PIL import Image
from skimage import io
import numpy as np
from torchvision.transforms import Resize, ToTensor, Normalize
import torch
import ast
from utils import image2edges, edges2mask

class FashionEdgesDataset(Dataset):
    def __init__(self, 
                 images_fold, 
                 attr_file=None, 
                 check_corrupted=False,
                 return_mask=True):
        
        self.corrupted_images = set()
        self.check_corrupted = check_corrupted
        self.images_fold = images_fold
        
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
    
        return np.all(img[:thresh,:thresh] == 255)
    
    def __getitem__(self, idx):
        
        image_name = self.images_names[idx]
        img = Image.open(os.path.join(self.images_fold, image_name))#.convert('RGB')        
        if not self.check_corrupted and not self._is_appropriate(np.array(img)):
            return None
        
        img = Resize((128, 128))(img)
        
        img = ToTensor()(img)
        if img.shape[0] > 3:
            img = img[:3]
        
        img_np = tensor2numpy(img)

        edges_np = image2edges(img_np)
        edges = torch.tensor(edges_np, dtype=torch.float32).unsqueeze(0)
        if return_mask:
            mask = edges2mask(edges_np)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return edges, mask, img
        else:
            return edges, img   

    def __len__(self):
        return len(self.images_names)


  