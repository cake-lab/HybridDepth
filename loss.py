import torch
import torch.nn as nn
import kornia

import torch.nn.functional as F
import torch.cuda.amp as amp


class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None):

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)

            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        return loss

@torch.jit.script
def pyrdown(input_tensor: torch.Tensor, num_scales: int = 4):
    """ Creates a downscale pyramid for the input tensor. """
    output = [input_tensor]
    for _ in range(num_scales - 1):
        down = kornia.filters.blur_pool2d(output[-1], 3)
        output.append(down)
    return output


class MSGradientLoss(nn.Module):
    def __init__(self, num_scales: int = 4):
        super().__init__()

        self.num_scales = num_scales

    def forward(self, depth_pred, depth_gt):
        depth_pred_pyr = pyrdown(depth_pred, self.num_scales)
        depth_gtn_pyr = pyrdown(depth_gt, self.num_scales)

        grad_loss = torch.tensor(0, dtype=depth_gt.dtype, device=depth_gt.device)
        for depth_pred_down, depth_gtn_down in zip(depth_pred_pyr, depth_gtn_pyr):

            depth_gtn_grad = kornia.filters.spatial_gradient(depth_gtn_down)
            # mask_down_b = depth_gtn_grad.isfinite().all(dim=1, keepdim=True)
            # Mask where depth_gt_grad is not zero
            mask_not_zero = depth_gtn_grad != 0
            # Making sure the mask includes all channels
            mask_down_b = mask_not_zero.all(dim=1, keepdim=True)

            depth_pred_grad = kornia.filters.spatial_gradient(
                                    depth_pred_down).masked_select(mask_down_b)

            grad_error = torch.abs(depth_pred_grad - 
                                    depth_gtn_grad.masked_select(mask_down_b))
            grad_loss += torch.mean(grad_error)

        return grad_loss
    
