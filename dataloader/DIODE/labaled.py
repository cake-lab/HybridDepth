"""A loader for the labeled ARKitScenes dataset."""
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms_v2
import imageio
import numpy as np
# from .gen_focalStack import run

class DIODEDataset(Dataset):
    """Python interface for the labeled subset of the ARkitScenes dataset.

    To save memory, call the `close()` method of this class to close
    the dataset file once you're done using it.
    """

    def __init__(self, path, img_size=(192, 256), stage="val"):
        
        self.root_dir = path
        self.stage = stage
        
        # open csv file
        with open(f'dataloader/DIODE/input_split/val_indoors.csv','r') as f:
                self.samples_pth= f.readlines()
        self.img_size = img_size

    def __len__(self):
        return len(self.samples_pth)

    def __getitem__(self, idx):
        # open the color image
        if idx == 60:
            print(self.samples_pth[idx].split(',')[0])
        paths = self.samples_pth[idx].split(',')
        color_img_path = paths[0].replace('./',' ').strip()
        depth_img_path = paths[1].replace('./',' ').strip()
        mask_img_path = paths[2].replace('./',' ').strip()
        
        color_img = imageio.imread(os.path.join(self.root_dir,color_img_path), pilmode="RGB")
        depth_img = np.load(os.path.join(self.root_dir,depth_img_path))
        mask_img = np.load(os.path.join(self.root_dir,mask_img_path))
        
        depth_img[mask_img == 0] = 0
        # print(depth_img.max())
        depth_img = depth_img.squeeze()
        
        # Convert to float tensor
        color_img = torch.from_numpy(color_img).float()  
        depth_img = torch.from_numpy(depth_img).float()
        mask_img = torch.from_numpy(mask_img).bool()
        depth_img = depth_img.unsqueeze(0)

        if mask_img.dim() == 2:
            mask_img = mask_img.unsqueeze(0)  # Adds a channel dimension at the front
        
        color_img = color_img.permute(2, 0, 1)
        # depth_img = depth_img.permute(2, 0, 1)
        
            
        t_resize = transforms_v2.Resize(self.img_size, antialias=True)

        color_img = t_resize(color_img)
        depth_img = t_resize(depth_img)
        mask_img = t_resize(mask_img)

        return color_img, depth_img, mask_img
    
    
