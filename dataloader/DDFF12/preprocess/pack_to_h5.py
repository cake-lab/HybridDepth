import argparse
import os
import cv2
import h5py
import numpy as np
from glob import glob
import scipy.io
from depth2disp import dpth2disp
import sys


def pack_h5(stack_train_dir, disp_train_dir, train_lst, val_lst, out_dir, mat, sck_per_scene=100):

    b_create = False
    hdf5 = None
    cnt = {True: 0, False: 0}

    for dir in os.listdir(disp_train_dir):
        if dir in train_lst:
            key_img, key_disp, key_aif, b_train = 'stack_train', 'disp_train', 'AiF_train', True
        else:
            key_img, key_disp, key_aif, b_train = 'stack_val', 'disp_val', 'AiF_val', False


        for sck_id in sorted(os.listdir(stack_train_dir + '/' + dir)):
            img_list = sorted(glob(os.path.join(stack_train_dir, dir, sck_id, '*.png')))
            dpth_pth = '{}/{}/DEPTH_{}.png'.format(disp_train_dir, dir, sck_id)

            aif_img = cv2.imread(img_list[0])[:,:,::-1]
            disp,depth = dpth2disp(dpth_pth, mat=mat)
            disp[disp==np.inf] = 0
            stack_samples = np.asarray([cv2.imread(img_list[x])[:,:,::-1] for x in range(1, 11)])

            if not b_create:
                n_train = int (len(train_lst) * sck_per_scene) # every scene has 100
                n_val = int(len(val_lst) * sck_per_scene)
                stack_shape = stack_samples.shape
                disp_shape = disp.shape
                # Create hdf5 dataset
                hdf5 = h5py.File(out_dir, mode='w')
                hdf5.create_dataset("stack_train", shape=(n_train, ) + stack_shape)
                hdf5.create_dataset("disp_train", shape=(n_train, ) + disp_shape)
                hdf5.create_dataset("AiF_train", shape=(n_train, )+ aif_img.shape)  # not used in our work
                hdf5.create_dataset("stack_val", shape=(n_val,) +  stack_shape)
                hdf5.create_dataset("disp_val", shape=(n_val,) + disp_shape)
                hdf5.create_dataset("AiF_val", shape=(n_val,)  + aif_img.shape) # not used in our work
                
                depth_shape = depth.shape
                hdf5.create_dataset("depth_train", shape=(n_train,) + depth_shape)
                hdf5.create_dataset("depth_val", shape=(n_val,) + depth_shape)
                b_create = True

            # save to
            assert hdf5 is not None
            # Save depth map to HDF5
            key_depth = 'depth_train' if b_train else 'depth_val'
            hdf5[key_depth][cnt[b_train]] = depth
            hdf5[key_img][cnt[b_train]] = stack_samples
            hdf5[key_disp][cnt[b_train]] = disp
            hdf5[key_aif][cnt[b_train]] = aif_img
            cnt[b_train] += 1

            # interface
            sys.stdout.write('\r{}: {}/{}'.format(dir, sck_id, sck_per_scene))
            sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", default='/home/ashka/dff/data/DDFF12/images/train',
                            help="input directory containing training focal stacks", type=str)
    parser.add_argument("--disp_folder",default='/home/ashka/dff/data/DDFF12/DepthRegistered/train',
                         help="input directory containing training disparities", type=str)
    parser.add_argument("--calib_mat", default='/home/ashka/dff/dataloader/DDFF12/preprocess/third_part/IntParamLF.mat',
                        help="input file of camera calibration result", type=str)
    parser.add_argument("--outfile", default='/home/ashka/dff/data/ddff_trainVal.h5', help="h5 file to be written", type=str)
    args = parser.parse_args()

    np.random.seed(1)

    # pick train and val
    scene_lst = os.listdir(args.img_folder)
    val_idx = np.random.choice(len(scene_lst), 2, replace=False)

    # to get exactly same train/val split with ours, please replace the follow 2 lines with the hard-code list
    val_list = [scene_lst[x] for x in val_idx]  # ['glassroom', 'office41']
    train_list = [x for x in scene_lst if x not in val_list] # ['socialcorner', 'studentlab', 'seminaroom', 'kitchen']

    # load calib mat
    mat = scipy.io.loadmat(args.calib_mat)
    if 'IntParamLF' in mat:
        mat = np.squeeze(mat['IntParamLF'])
    else:
        raise('Calib Mat not found! ')

    print(val_list, train_list)
    # pack h5
    pack_h5(args.img_folder, args.disp_folder, train_list,val_list, args.outfile, mat)
