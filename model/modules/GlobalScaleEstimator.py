import torch

def compute_scale_and_shift_ls(prediction, target, mask):
    sum_axes = (2, 3)
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, sum_axes)
    a_01 = torch.sum(mask * prediction, sum_axes)
    a_11 = torch.sum(mask, sum_axes)

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, sum_axes)
    b_1 = torch.sum(mask * target, sum_axes)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

class LeastSquaresEstimator(object):
    def __init__(self, estimate, target, valid):
        self.estimate = estimate
        self.target = target
        self.valid = valid

        # To be computed
        self.scale = 1.0
        self.shift = 0.0
        self.output = None

    def compute_scale_and_shift(self):
        self.scale, self.shift = compute_scale_and_shift_ls(self.estimate, self.target, self.valid)

    def apply_scale_and_shift(self):
        # Adjust self.scale and self.shift to have a shape of [batch_size, 1, 1, value]
        scale_expanded = self.scale.unsqueeze(1).unsqueeze(2)  # Shape becomes [4, 1, 1, 1]
        shift_expanded = self.shift.unsqueeze(1).unsqueeze(2)  # Shape becomes [4, 1, 1, 1]

        # Perform the operation
        self.output = self.estimate * scale_expanded + shift_expanded
        # self.output = self.estimate * self.scale + self.shift

    def clamp_min_max(self, clamp_min=None, clamp_max=None):
        if clamp_min is not None:
                self.output = torch.clamp(self.output, max=clamp_max)
        if clamp_max is not None:
            self.output = torch.clamp(self.output, min=clamp_min)