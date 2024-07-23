import torch
from torch.utils.data import Dataset, random_split
from utils.synthetic import camera_lib
from dataloader.ToF18.labeled import ToFLabeledDataset
from typing import Tuple
from torchvision import transforms


class ToFLoader(Dataset):
    """A PyTorch dataset for depth from focus (DFF)."""

    def __init__(
        self,
        tof18_data_root: str = "",
        img_size: Tuple = (480, 640),
    ) -> None:
        super(ToFLoader, self).__init__()

        self.tof18_dataset = ToFLabeledDataset(
            tof18_data_root,
            img_size=img_size
        )

        self.renderer = camera_lib.GaussPSF(3)
        self.renderer.cuda()
        self.camera = camera_lib.ThinLenCamera()
        self.mean_input= [0.485, 0.456, 0.406]
        self.std_input=[0.229, 0.224, 0.225]
        
    def __len__(self) -> int:
        return len(self.tof18_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_aif, depth = self.tof18_dataset[index]
        # create a mask 
        # mask = depth > 0
        mask = depth != 0
        
        # mask = torch.logical_and(depth > 1e-3,depth < 10)
        rgb_aif = rgb_aif.div(255)
        # m = torch.median(depth)
        
        min = torch.min(depth)
        max  = torch.max(depth)

        focus_distances = torch.linspace(min,max, steps=10)
        
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
        
        # Normalize the rgb_aif using mean and standard deviation
        rgb_aif = normalize(rgb_aif)

        return rgb_aif, focal_stack_normalized, depth,focus_distances, mask