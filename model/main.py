from typing import Any, Tuple
from time import time as gettime
import pytorch_lightning as pl
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn

from loss import MSGradientLoss, SILogLoss
from dataclasses import dataclass, field

from model.depthNet import DepthNet
import torch.nn.functional as F

from utils.io import IO
from utils.metric_cal import calmetrics, calmetrics_conf

# @dataclass
# class LearningRateConfig:
#     lr_steps: list = field(default_factory=lambda: [399, 699])


class DepthNetModule(pl.LightningModule):
    """A baseline DFF model."""

    def __init__(
        self, invert_depth: bool = False, lr: float = 0.1, wd: float = 0.1, *args: Any, **kwargs: Any
    ) -> None:
        super(DepthNetModule, self).__init__(*args, **kwargs)

        self.lr = lr
        self.wd = wd
        self.model = DepthNet(invert_depth=invert_depth)
        self.io = IO(
            "results/csv/metrics.csv", "results/imgs/ours", "results/csv/std.csv"
        )
        self.grad_loss = MSGradientLoss(num_scales=4)
        self.abs_loss = nn.L1Loss()
        self.silog = SILogLoss()

        # self.lr_opt = LearningRateConfig()
    def load_from_pth(self, file_path):
        state_dict = torch.load(file_path, map_location=self.device)
        self.load_state_dict(state_dict)

    def forward(self, rgb_aif: torch.Tensor, focal_stack: torch.Tensor, foc_dist) -> torch.Tensor:
        return self.model.forward(rgb_aif, focal_stack, foc_dist)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        loss, _, _, _, _, _, _, _ = self._common_step(batch, batch_idx)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        loss, rgb_aif, depth_prd, depth_gt, scale_map, mask, gl_depth, rel_depth = self._common_step(
            batch, batch_idx, stage="val"
        )
        # run only if epoch is divisible by 5
        # if self.trainer.current_epoch % 5 == 0:
        # depth_prd_np = (depth_prd.cpu().numpy()
        #                 if depth_prd.is_cuda else depth_prd.numpy())
        # depth_gt_np = depth_gt.cpu().numpy() if depth_gt.is_cuda else depth_gt.numpy()
        # mask_np = mask.cpu().numpy()
        # metrics = calmetrics(depth_prd_np, depth_gt_np, mask_np)
        # self.io.save_metrics_to_csv(metrics)
        # self.io.tensors_to_image(rgb_aif, depth_prd, depth_gt, rel_depth, scale_map, metrics, batch_idx)
        # self.io.tensors_to_image(rgb_aif, depth_prd, gt_depth=depth_gt, rel_depth=rel_depth,
        #                          scale_map=scale_map, gl_depth=gl_depth, metrics=metrics, batch_idx=batch_idx)

        return loss

    def test_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int):
        
        if len(batch) == 5:
            rgb_aif, focal_stack, depth_gt, focus_dist, mask = batch
        else : 
            rgb_aif, focal_stack, depth_gt, focus_dist, mask, conf_gt = batch
            
        with torch.no_grad():
            # Run forward pass.
            start_time = gettime()
            depth_prd, rel_depth, gl_depth, _, scale_map = self.forward(
                rgb_aif, focal_stack, focus_dist)
            time = (gettime() - start_time)

        l_grad = self.grad_loss(depth_prd, depth_gt)
        siloss = self.silog(depth_prd, depth_gt, mask)
        loss = siloss + l_grad * 0.5
        
        depth_prd_np = (depth_prd.cpu().numpy()
                        if depth_prd.is_cuda else depth_prd.numpy())
        depth_gt_np = depth_gt.cpu().numpy() if depth_gt.is_cuda else depth_gt.numpy()

        
        # only consider pixels with depth values < 2 meters and > 0
        # mask_np = (depth_gt_np < 2.4) & (depth_gt_np > 0)

        mask_np = mask.cpu().numpy()

        metrics = calmetrics(depth_prd_np, depth_gt_np, mask_np)
        metrics = np.append(metrics, time)

        self.io.save_metrics_to_csv(metrics)
        if batch_idx in [0, 10, 20, 30, 100, 110, 160, 180, 380, 400]:
                np.save(f"results/imgs/ours/{batch_idx}_rgb.npy", rgb_aif.cpu().numpy())
                np.save(f"results/imgs/ours/{batch_idx}_gt.npy", depth_gt_np)
                np.save(f"results/imgs/ours/{batch_idx}_depth.npy", depth_prd_np)
        self.io.tensors_to_image(rgb_aif, depth_prd, gt_depth=depth_gt, rel_depth=rel_depth,
                                 scale_map=scale_map, gl_depth=gl_depth, metrics=metrics, batch_idx=batch_idx)

        return loss

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], save_images: bool = False):
        with torch.no_grad():
            rgb_aif, focal_stack, focus_dist, mask = batch
            depth_prd, rel_depth, gl_depth, _, scale_map = self.forward(
                rgb_aif, focal_stack, focus_dist)
            if save_images:
                self.save_depth_maps(depth_prd, rgb_aif,
                                     rel_depth, gl_depth, scale_map)

        return depth_prd

    def _common_step(self, batch, batch_idx: int, stage: str = "train"):
        rgb_aif, focal_stack, depth_gt, focus_dist, mask = batch
        self.log_image(rgb_aif, f"{stage}/rgb_aif")
        # Run forward pass.
        depth_prd, rel_depth, gl_depth, _, scale_map = self.forward(
            rgb_aif, focal_stack, focus_dist)
        t_gt = depth_gt - depth_gt.min()
        t_gt = t_gt / t_gt.max()
        t_prd = depth_prd - depth_prd.min()
        t_prd = t_prd / t_prd.max()

        t = torch.concat([t_gt, t_prd], dim=0)
        self.log_depth(t, f"{stage}/gt_and_prd")
        l_grad = self.grad_loss(depth_prd, depth_gt)
        siloss = self.silog(depth_prd, depth_gt, mask)

        loss = siloss + l_grad * 0.5

        self.log(f"{stage}/total_loss", loss, prog_bar=True, sync_dist=True)

        return loss, rgb_aif, depth_prd, depth_gt, scale_map, mask, gl_depth, rel_depth

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(
    #         self.model.parameters(),
    #         lr=1e-4,
    #         betas=(0.5, 0.9),
    #     )
    #     return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.wd,
                                      betas=(0.9, 0.999)
                                      )

        return optimizer
        # def lr_lambda(step):
        #     if step < self.lr_opt.lr_steps[0]:
        #         return 1
        #     elif step < self.lr_opt.lr_steps[1]:
        #         return 0.5
        #     else:
        #         return 0.5
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        # return {"optimizer": optimizer,
        #         "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"}}

    def log_depth(
        self, depth_maps: torch.Tensor, name: str, log_every_n_step: int = 10
    ) -> None:
        if self.global_step % log_every_n_step == 0:
            # Check if the tensor is 4D (batch of 3D tensors)
            if depth_maps.ndim != 4 or depth_maps.shape[1] != 1:
                raise ValueError("Expected depth maps with shape [N, 1, H, W]")

            # Normalize and clip each depth map in the batch
            depth_maps = torch.clip(depth_maps, 0, 1)
            processed_maps = []
            # Process each depth map in the batch
            for depth_map in depth_maps:
                depth_map_np = depth_map.squeeze(0).detach().cpu().numpy()
                depth_map_np = (depth_map_np - np.min(depth_map_np)) / (
                    np.max(depth_map_np) - np.min(depth_map_np)
                )
                colored_map = plt.get_cmap("jet")(depth_map_np)[
                    :, :, :3
                ]  # Apply colormap and remove alpha channel
                colored_map_tensor = (
                    torch.from_numpy(colored_map).float().permute(
                        2, 0, 1).unsqueeze(0)
                )
                processed_maps.append(colored_map_tensor)
            all_maps = torch.cat(processed_maps, dim=0)

            self.logger.experiment.add_image(
                name,
                torchvision.utils.make_grid(all_maps, nrow=2, padding=2),
                global_step=self.global_step,
                dataformats="CHW",
            )

    def log_image(
        self, img: torch.Tensor, name: str, log_every_n_step: int = 10
    ) -> None:
        if self.global_step % log_every_n_step == 0:
            img = img  # / 2 + 0.5
            img = torch.clip(img, 0, 1)
            self.logger.experiment.add_image(
                name,
                torchvision.utils.make_grid(
                    img, nrow=2, padding=2).detach().cpu(),
                global_step=self.global_step,
                dataformats="CHW",
            )
