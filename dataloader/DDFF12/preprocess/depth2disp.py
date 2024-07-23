#! /usr/bin/python3

import numpy as np
import scipy.io
import cv2

####################################################################
# Adopted from - https://github.com/hazirbas/ddff-toolbox/blob/master/utility/python/refocus.py
####################################################################


def dpth2disp(dpth_pth, calib_mat=None, mat=None):
    #Load calibration parameters
    if mat is None:
        mat = scipy.io.loadmat(calib_mat)
        if 'IntParamLF' in mat:
            mat = np.squeeze(mat['IntParamLF'])
        else:
            return
    K2 = mat[1]
    fxy = mat[2:4]
    flens = max(fxy)
    fsubaperture = 521.4052 # pixel
    baseline = K2/flens*1e-3 # meters/ meter/px

  
    #Read light field image according to instructions given at http://hazirbas.com/datasets/ddff12scene/
    dpth = cv2.imread(dpth_pth, -1) / 1000.
    disp = baseline*fsubaperture / dpth
    return disp, dpth
