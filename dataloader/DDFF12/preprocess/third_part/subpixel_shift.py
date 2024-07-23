#! /usr/bin/python3

####################################################################
# adopted from https://github.com/hazirbas/ddff-toolbox/blob/02064fc92bcd3867afcc57ceb7fd0c80c5494099/utility/python/subpixel_shift.py
# 2015.05.12 Hae-Gon Jeon
# Accurate Depth Map Estimation from a Lenslet Light Field Camera
# CVPR 2015

# and Ported to python3 by Sebastian Soyer
# Deep Depth From Focus
# ACCV 2018
# ACADEMIC USE ONLY
####################################################################

import numpy as np

def subpixel_shift(f, delta, nr, nc, mode):
    """Image shift with sub-pixel precision using phase theorem input.

    Args:
        f: fft of color image.
        delta: 1X2 matrix for sub-pixel displacement.
        nr - row size of input image.
        nc - column size of input image.
        mode - 1) color image, 2) gradient image.

    Returns:
        output: A sub-pixel shifted image

    """

    deltar = delta[0]
    deltac = delta[1]
    phase = 2


    Nr = np.fft.ifftshift(range(int(-np.fix(nr/2)),int(np.ceil(nr/2))))
    Nc = np.fft.ifftshift(range(int(-np.fix(nc/2)),int(np.ceil(nc/2))))
    [Nc,Nr] = np.meshgrid(Nc,Nr)

    result = np.fft.ifft2(f * np.tile(np.exp(1j*2*np.pi*(deltar*Nr/nr+deltac*Nc/nc))[:,:,np.newaxis], [1, 1, 3]), axes=(0,1)) * np.exp(-1j*phase)

    if mode == 1:
        return np.clip(abs(result), 0, 1)
    else:
        return abs(result)
