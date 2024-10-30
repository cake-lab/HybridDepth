import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.midas.midas_net_custom import MidasNet_small_videpth
from .modules.DFV import DFFNet
from .modules.GlobalScaleEstimator import LeastSquaresEstimator
from .modules.depth_anything.dpt import DepthAnything


class DepthNet(nn.Module):
    """A baseline model for depth from focus (DFF)."""

    def __init__(self, invert_depth: bool = True):
        super(DepthNet, self).__init__()
        self.invert_depth = invert_depth
        self.depth_anything = DepthAnything.from_pretrained(
            'LiheYoung/depth_anything_{:}14'.format('vits')).eval()
        self.depth_anything.requires_grad_(False)
        self.depth_anything.cuda()
        self.depth_anything.eval()

        self.DFF_model = DFFNet(clean=False, level=4, use_diff=1)
        self.DFF_model.cuda()
        DFV_weights = None  # Don't need this for evaluation, only add for training

        if DFV_weights is not None:
            pretrained_dict = torch.load(DFV_weights)
            pretrained_dict['state_dict'] = {
                k.replace('module.', ''): v
                for k, v in pretrained_dict['state_dict'].items()
                if 'disp' not in k
            }
            self.DFF_model.load_state_dict(
                pretrained_dict['state_dict'], strict=False)
            print('Loaded DFV model')
        else:
            print('Run with random init')

        self.DFF_model.eval()
        self.DFF_model.requires_grad_(False)

        self.ScaleMapLearner = MidasNet_small_videpth(
            path=None,
            min_pred=0.02,
            max_pred=0.28,
            backbone="efficientnet_lite3",
            in_channels=3
        )
        self.ScaleMapLearner.train()

    def predic_depth_rel(self, batch):
        org_size = batch.shape
        batch = F.interpolate(batch, size=(518, 518), mode='bicubic', align_corners=True)
        depth = self.depth_anything(batch).unsqueeze(1)
        depth = F.interpolate(depth, size=(org_size[2], org_size[3]), mode='bicubic', align_corners=True)
        return depth

    def forward(self, rgb_aif, focal_stack, disp_dist):
        org_size = rgb_aif.shape
        self.DFF_model.eval()

        rel_depth = self.predic_depth_rel(rgb_aif)
        pred_dff, std, _ = self.DFF_model(focal_stack, disp_dist)
        if len(pred_dff) == 4:
            pred_dff = pred_dff[3]

        valid_mask = torch.ones((org_size[0], 1, org_size[2], org_size[3]),
                                dtype=torch.bool, device=rgb_aif.device, requires_grad=False)
        if self.invert_depth:
            rel_depth = 1.0 / rel_depth
            rel_depth[rel_depth == float("inf")] = 0

        GlobalAlignment = LeastSquaresEstimator(
            estimate=rel_depth,
            target=pred_dff,
            valid=valid_mask
        )

        GlobalAlignment.compute_scale_and_shift()
        GlobalAlignment.apply_scale_and_shift()
        intr_depth = GlobalAlignment.output.float()

        scale_map = self.scaffholding(intr_depth, pred_dff)

        sample = {"int_depth": intr_depth,
                  "int_scales": scale_map, "int_depth_no_tf": intr_depth}

        x = torch.cat([sample["int_depth"], sample["int_scales"], std], dim=1)
        d = sample["int_depth_no_tf"]

        metric_depth, scale_map = self.ScaleMapLearner(x, d)

        return metric_depth, rel_depth, intr_depth, std, pred_dff

    def scaffholding(self, intr_depth, pred_disp):
        scale_map = torch.ones_like(intr_depth)
        scale_factors = pred_disp / intr_depth
        scale_map = torch.where(torch.isfinite(
            scale_factors), scale_factors, torch.tensor(1.0).to(intr_depth.device))
        return scale_map