"""A library for camera operations."""
from typing import List

import gauss_psf_cuda as gauss_psf
import torch
import torch.nn as nn
from torch.autograd import Function


class GaussPSFFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, device, kernel_size=11):
        with torch.no_grad():
            x = (
                torch.arange(kernel_size // 2, -kernel_size // 2, -1)
                .view(kernel_size, 1)
                .float()
                .repeat(1, kernel_size)
                .to(device)
            )

            y = (
                torch.arange(kernel_size // 2, -kernel_size // 2, -1)
                .view(1, kernel_size)
                .float()
                .repeat(kernel_size, 1)
                .to(device)
            )

        outputs, wsum = gauss_psf.forward(input, weights, x, y)
        ctx.save_for_backward(input, outputs, weights, wsum, x, y)
        return outputs

    @staticmethod
    def backward(ctx, grad):
        input, outputs, weights, wsum, x, y = ctx.saved_variables
        x = -x
        y = -y
        grad_input, grad_weights = gauss_psf.backward(
            grad.contiguous(), input, outputs, weights, wsum, x, y
        )
        return grad_input, grad_weights, None, None


class GaussPSF(nn.Module):
    def __init__(self, kernel_size):
        super(GaussPSF, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, image, psf):
        psf = psf.unsqueeze(1).expand_as(image).contiguous()
        return GaussPSFFunction.apply(image, psf, image.device, self.kernel_size)


class ThinLenCamera:
    def __init__(
        self,
        fnumber=0.5,
        focal_length=2.9 * 1e-3,
        sensor_size=3.1 * 1e-3,
        img_size=560,
        pixel_size=None,
    ):
        self.focal_length = focal_length
        self.D = self.focal_length / fnumber
        self.pixel_size = pixel_size
        if not self.pixel_size:
            self.pixel_size = sensor_size / img_size

    def getCoC(self, dpt, focus_dist):
        # dpt : BxFS H W
        # focus_dist : BxFS H W
        sensor_dist = focus_dist * self.focal_length / (focus_dist - self.focal_length)
        CoC = (
            self.D
            * sensor_dist
            * torch.abs(1 / self.focal_length - 1 / sensor_dist - 1 / (dpt + 1e-8))
        )
        sigma = CoC / 2 / self.pixel_size
        return sigma.type(torch.float32)


def render_defocus(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    camera: ThinLenCamera,
    psf_renderer: GaussPSF,
    focus_distances: List[float],
):
    device = rgb.device

    rgb = rgb.float()
    rgb = rgb.cuda().contiguous()

    depth = depth.float()
    depth = depth.cuda().contiguous()

    focal_stack = []

    for fd in focus_distances:
        defocus = camera.getCoC(depth, fd).type(torch.float32)
        fs = psf_renderer(rgb.unsqueeze(0).cuda(), defocus.cuda())
        focal_stack.append(fs[0].to(device))

    focal_stack = torch.stack(focal_stack, dim=0)

    return focal_stack