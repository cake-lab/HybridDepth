import torch
from torch.utils.data import Dataset, random_split
from utils.synthetic import camera_lib
from dataloader.NYU.labeled import NYUv2LabeledDataset
from typing import Tuple
from torchvision import transforms
from torchvision.transforms import functional as TF


class NYULoader(Dataset):
    """A PyTorch dataset for depth from focus (DFF)."""

    def __init__(
        self,
        nyuv2_data_root: str = "",
        img_size: Tuple = (480, 640),
        remove_white_border: bool = True,
        stage: str = "test",
        n_stack: int = 10,
    ) -> None:
        super(NYULoader, self).__init__()

        self.nyuv2_dataset = NYUv2LabeledDataset(
            nyuv2_data_root,
            img_size=img_size,
            remove_white_border=remove_white_border,
            stage=stage
        )
        self.renderer = camera_lib.GaussPSF(7)
        self.renderer.cuda()
        self.camera = camera_lib.ThinLenCamera(8, focal_length=50e-3, pixel_size=1.2e-5)
        self.mean_input= [0.485, 0.456, 0.406]
        self.std_input=[0.229, 0.224, 0.225]
        
        self.n_stack = n_stack
        
    def __len__(self) -> int:
        return len(self.nyuv2_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_aif, depth = self.nyuv2_dataset[index]
        # create a mask 
        # mask = depth > 0
        # mask = depth != 0
        
        mask = torch.logical_and(depth > 1e-3,depth < 10)
        rgb_aif = rgb_aif / 255.0
        
        min = torch.min(depth[mask==1])
        max  = torch.max(depth[mask==1])
        focus_distances = torch.linspace(min,max, steps=self.n_stack)
        
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
        
        return rgb_aif, focal_stack_normalized, depth,focus_distances, mask
    