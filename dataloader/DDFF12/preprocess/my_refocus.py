#! /usr/bin/python3

import numpy as np
import scipy.io
import cv2
import os
import sys

from glob import glob
# import subpixel_shift

####################################################################
# adopted from https://github.com/hazirbas/ddff-toolbox/blob/02064fc92bcd3867afcc57ceb7fd0c80c5494099/utility/python/subpixel_shift.py
# 2015.05.12 Hae-Gon Jeon
# Accurate Depth Map Estimation from a Lenslet Light Field Camera
# CVPR 2015

# and Ported to python3 by Sebastian Soyer
# Deep Depth From Focus
# ACCV 2018
# ACADEMIC USE ONLY

# Code to generate DDFF-12 train and val split from their raw data
# Last modification: Fengting Yang 2020-03-8
####################################################################


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


def refocus(light_field, calib_mat, output_folder, stack_size=10):
    #Load calibration parameters
    mat = scipy.io.loadmat(calib_mat)
    if 'IntParamLF' in mat:
        mat = np.squeeze(mat['IntParamLF'])
    else:
        return
    K2 = mat[1]
    fxy = mat[2:4]
    flens = max(fxy)
    fsubaperture = 521.4052 # pixel
    baseline = K2/flens*1e-3 # meters

    depth_range = [0.5, 7]
    disparity_range = (baseline*fsubaperture / depth_range)
    disparities = np.linspace(disparity_range[0], disparity_range[1], num=stack_size)

    #Read light field image according to instructions given at http://hazirbas.com/datasets/ddff12scene/
    lf = np.load(light_field) / 255.0

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # save all in focus
    lfsize = (lf.shape[2], lf.shape[3])
    uvcenter = np.asarray((np.asarray([lf.shape[0], lf.shape[1]]) + 1) / 2)
    allF_img = np.uint8(lf[int(uvcenter[0]), int(uvcenter[1])].clip(0,1)*255)
    cv2.imwrite(output_folder + "/" + "00.png", cv2.cvtColor(allF_img, cv2.COLOR_RGB2BGR))

    for idx, disparity in enumerate(disparities):

        image = np.zeros( lfsize + (3,))

        for u in range(lf.shape[0]):
            for v in range(lf.shape[1]):
                shift = (uvcenter - np.asarray([u+1,v+1])) * disparity
                shifted = subpixel_shift(
                    np.fft.fft2(np.squeeze(lf[u,v]), axes=(0,1)),
                    shift,
                    lfsize[0],
                    lfsize[1],
                    1)
                image = image + shifted

        
        image = image / np.prod([lf.shape[0], lf.shape[1]])
        image = np.uint8(image * 255.0)

        #Convert RGB to BGR (OpenCV assumes image to be BGR) and write output image
        cv2.imwrite(output_folder + "/" + "{0:02d}".format(idx+1) + ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    data_pth = '/home/ashka/dff/data/DDFF12/LightField/test' # raw lightfield data pth
    out_pth = '/home/ashka/dff/data/DDFF12/images/train' # raw image path
    for fd in os.listdir(data_pth):
        if not os.path.isdir(out_pth + '/{}'.format(fd)):
            os.makedirs(out_pth + '/{}'.format(fd))
        tmp_pth = data_pth + '/' + fd
        npy_lst = glob(tmp_pth + '/*.npy')
        n_img = len(npy_lst)
        for i, npy_pth in enumerate(npy_lst):
            sys.stdout.write('\r {}: {}/{}'.format(fd, i, n_img))
            sys.stdout.flush()
            img_name = os.path.basename(npy_pth)[3:7]
            if os.path.isdir(out_pth + '/{}/{}'.format(fd, img_name)) and \
                    len(os.listdir(out_pth + '/{}/{}'.format(fd, img_name))) == 11:
                continue
            refocus(npy_pth, '/home/ashka/dff/dataloader/DDFF12/preprocess/third_part/IntParamLF.mat', out_pth + '/{}/{}'.format(fd, img_name) )
