import torch
from torch.utils.data import Dataset
from utils.synthetic import camera_lib
from dataloader.Matterport3D.labeled import Matterport3dDataset
from typing import Tuple
from torchvision import transforms
import torch.multiprocessing as mp

class Matterport3DLoader(Dataset):
    """A PyTorch dataset for depth from focus (DFF)."""

    def __init__(
            self,
            Matterport3D_root: str = "",
            img_size: Tuple = (192, 256), stage: str = "train"):
        super(Matterport3DLoader, self).__init__()

        self.matterport3d_dataset = Matterport3dDataset(
            Matterport3D_root,
            img_size=img_size,
            stage=stage
        )

        self.renderer = camera_lib.GaussPSF(7)
        self.renderer.cuda()
        
        # self.camera = camera_lib.ThinLenCamera(fnumber=2.4, focal_length=3.3 * 1e-3, sensor_size=4.8 * 1e-3, img_size=1440)
        self.camera = camera_lib.ThinLenCamera(img_size=480)
        
        self.mean_input = [0.485, 0.456, 0.406]
        self.std_input = [0.229, 0.224, 0.225]

    def __len__(self) -> int:
        return len(self.matterport3d_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_aif, depth = self.matterport3d_dataset[index]
        # print(depth.shape)
        # create a mask
        mask = depth != 0
        rgb_aif = rgb_aif / 255.0
    
        # uncomment the following lines to use generate focal stack during training
        min_depth = torch.min(depth)
        max_depth = torch.max(depth)

        focus_distances = torch.linspace(min_depth, max_depth, steps=10)

        focal_stack = camera_lib.render_defocus(
            rgb_aif,
            depth,
            self.camera,
            self.renderer,
            focus_distances
        )
        
        # Normalize the focal stack using mean and standard deviation
        normalize = transforms.Normalize(self.mean_input, self.std_input)
        
        focal_stack_normalized = normalize(focal_stack)
        
        return rgb_aif, focal_stack_normalized, depth, focus_distances, mask
