import torch
from torch.utils.data import Dataset
from utils.synthetic import camera_lib
from dataloader.DIODE.labaled import DIODEDataset
from typing import Tuple
from torchvision import transforms
import torch.multiprocessing as mp

class DIODELoader(Dataset):
    """A PyTorch dataset for depth from focus (DFF)."""

    def __init__(
            self,
            DIODE_data_root: str = "",
            img_size: Tuple = (192, 256), stage: str = "train"):
        super(DIODELoader, self).__init__()

        self.DIODE_dataset = DIODEDataset(
            DIODE_data_root,
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
        return len(self.DIODE_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_aif, depth, mask = self.DIODE_dataset[index]
        # create a mask
        # mask = depth != 0
        rgb_aif = rgb_aif / 255.0
        
        # uncomment the following lines to use generate focal stack during training
        # mask_1 = mask == 1
        min_depth = torch.min(depth)
        max_depth = torch.max(depth[mask == 1])
        # find the second max depth
        print(depth[mask==1].view(-1).topk(5, largest=True))
        avg_depth = torch.mean(depth[mask == 1])
        
        # print(max_depth, min_depth, avg_depth)

        focus_distances = torch.linspace(0, avg_depth + 1, steps=10)

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
