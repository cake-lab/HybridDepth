import torch
import sys
sys.path.append('../')

from utils.synthetic import camera_lib
import numpy as np
import torch
import os
import imageio
import matplotlib.pyplot as plt



ARKit_scense_path = '/mnt/IRONWOLF1/ashkan/data/ARKitScenes/upsampling'
split = ['Training', 'Validation']


val_video_ids = os.listdir(os.path.join(ARKit_scense_path, split[1]))
trian_video_ids = os.listdir(os.path.join(ARKit_scense_path, split[0]))


def create_focal_stack_dir(video_id, split):
    focal_stack_dir = os.path.join(
        ARKit_scense_path, split, video_id, 'focal_stack')
    if not os.path.exists(focal_stack_dir):
        os.makedirs(focal_stack_dir)
    return True

def gen_focal_stack(split, stack_size=10):
    video_id_list = trian_video_ids if split == 'Training' else val_video_ids

    for video_id in video_id_list:
        rgb_list = os.listdir(os.path.join(
            ARKit_scense_path, split, video_id, 'wide'))

        # check if the focal stack is already created
        focal_stack_dir = os.path.join(
            ARKit_scense_path, split, video_id, 'focal_stack')
        if len(os.listdir(focal_stack_dir)) > 0:
            print(f'Focal stack for {video_id} is already created')
            continue
        else:
            print(f'Creating focal stack for {video_id}')
            for rgb in rgb_list:
                # open the color image
                color_img = imageio.imread(os.path.join(
                    ARKit_scense_path, split, video_id, 'wide', rgb), pilmode="RGB")
                depth_img = imageio.imread(os.path.join(
                    ARKit_scense_path, split, video_id, 'highres_depth', rgb), pilmode="I")

                # Convert to float tensor
                color_img = torch.from_numpy(color_img).float()
                depth_img = torch.from_numpy(depth_img).float()
                # Move tensors to GPU
                color_img = color_img.to(device)
                depth_img = depth_img.to(device)
                
                
                depth_img = depth_img / 1000.0
                depth_img = depth_img.unsqueeze(0)

                color_img = color_img.permute(2, 0, 1)

                # Create the focal stack
                color_img = color_img/255.0
                min_depth = torch.min(depth_img)
                max_depth = torch.max(depth_img)

                focus_distances = torch.linspace(
                    min_depth, max_depth, steps=stack_size).to(device)

                focal_stack = camera_lib.render_defocus(
                    color_img,
                    depth_img,
                    camera,
                    renderer,
                    focus_distances
                ).to(device)
                
                # save the focal stack
                focal_stack_dir = os.path.join(
                    ARKit_scense_path, split, video_id, 'focal_stack')
                for i in range(stack_size):
                    if os.path.exists(os.path.join(focal_stack_dir, f'{rgb[:-4]}')):
                        folder_dir = os.path.join(focal_stack_dir, f'{rgb[:-4]}')
                        imageio.imwrite(os.path.join(folder_dir, f'{i}.png'), (
                            focal_stack[i].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8))
                    else:
                        os.mkdir(os.path.join(focal_stack_dir, f'{rgb[:-4]}'))

for video_id in val_video_ids:
    create_focal_stack_dir(video_id, split[1])

for video_id in trian_video_ids:
    create_focal_stack_dir(video_id, split[0])
    
    

# Assuming camera_lib is already imported and initialized for GPU use if necessary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Initialize your renderer and camera here
renderer = camera_lib.GaussPSF(9).to(device)
camera = camera_lib.ThinLenCamera().to(device)



def run (split, stack_size):
    gen_focal_stack(split, stack_size)
    
