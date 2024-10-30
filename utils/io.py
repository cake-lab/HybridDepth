import numpy as np
import os
import csv
from PIL import Image
import matplotlib.pyplot as plt
import torch
import imageio
import cv2


def align_images(base_img, img_to_align):
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
    align_gray = cv2.cvtColor(img_to_align, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(base_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(align_gray, None)
    if descriptors1 is None or descriptors2 is None:
        print("No descriptors found; check the input images.")
        return img_to_align

    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    matches = matcher.match(descriptors1, descriptors2)

    matches = sorted(matches, key=lambda x: x.distance)
    numGoodMatches = int(len(matches) * 0.15)
    good_matches = matches[:numGoodMatches]
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    if h is None:
        print("Homography could not be computed.")
        return img_to_align

    height, width = base_img.shape[:2]
    aligned_image = cv2.warpPerspective(img_to_align, h, (width, height))

    return aligned_image

def prepare_input_image(data_dir):
    image_files = sorted(os.listdir(data_dir))
    
    # Check for non-jpg files
    for file in image_files:
        if not file.endswith('.jpg'):
            raise ValueError(f"File {file} is not a .jpg file")

    # Sort on the focus distance
    image_files = sorted(image_files, key=lambda x: float(
        x.split("_")[-1].removesuffix(".jpg")))
    base_image = imageio.imread(os.path.join(data_dir, image_files[0]))
    base_image = cv2.resize(base_image, (640, 480))
    aligned_images = []
    focus_dist = np.array([])

    base_torch = torch.from_numpy(base_image).permute(
        2, 0, 1).unsqueeze(0).float()
    base_torch = base_torch / 255.0
    aligned_images.append(base_torch)

    focus_dist = np.append(focus_dist, float(
        image_files[0].split("_")[-1].removesuffix(".jpg")))
    for img_name in image_files[1:]:
        img_path = os.path.join(data_dir, img_name)
        focus_dist = np.append(focus_dist, float(
            img_name.split("_")[-1].removesuffix(".jpg")))

        current_image = imageio.imread(img_path)
        # Resize the image to 480x640
        current_image = cv2.resize(current_image, (640, 480))
        # aligned_img = current_image
        aligned_img = align_images(base_image, current_image)

        aligned_img = torch.from_numpy(aligned_img).permute(
            2, 0, 1).unsqueeze(0).float()
        aligned_img = aligned_img / 255.0

        aligned_images.append(aligned_img)

    rgb_img = aligned_images[0]

    focal_stack = torch.cat(aligned_images, dim=0)

    focus_dist = torch.from_numpy(focus_dist).float().unsqueeze(0)
    mean_input = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_input = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    normalized_focal_stack = (
        focal_stack - mean_input.to(focal_stack.device)) / std_input.to(focal_stack.device)
    normalized_rgb_img = (
        rgb_img - mean_input.to(rgb_img.device)) / std_input.to(rgb_img.device)
    normalized_focal_stack = normalized_focal_stack.unsqueeze(0)

    print(f'Focal stack shape: {normalized_focal_stack.shape}')
    print(f'RGB image shape: {normalized_rgb_img.shape}')
    print(f'Focus distances shape: {focus_dist.shape}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalized_focal_stack = normalized_focal_stack.to(device)
    rgb_img = rgb_img.to(device)
    focus_dist = focus_dist.to(device)

    return normalized_focal_stack, rgb_img, focus_dist


class IO:
    def __init__(self, csv_path, img_dir, std_dir=None):
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.std_dir = std_dir
        # remove the file if it already exists
        if os.path.isfile(self.csv_path):
            os.remove(self.csv_path)

        if os.path.isfile(self.std_dir) and self.std_dir is not None:
            os.remove(self.std_dir)

        # create the image directory if it doesn't exist
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

    def save_std_to_csv(self, std):
        # Convert metrics to a flat list if it's a NumPy array

        std = std.cpu().numpy().flatten().mean()
        # Define the header
        header = ["std"]
        # std = std.mean()
        # Check if the file already exists
        file_exists = os.path.isfile(self.std_dir)

        with open(self.std_dir, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write the header only if the file is being created
            if not file_exists:
                writer.writerow(header)

            # Write the std
            writer.writerow([std])

    def save_metrics_to_csv(self, metrics):
        # Convert metrics to a flat list if it's a NumPy array
        if isinstance(metrics, np.ndarray):
            metrics = metrics.flatten().tolist()

        # Define the header
        header = ["MSE", "RMSE", "LogRMSE", "AbsRel",
                  "SqureREL", "SSIM", "A1", "A2", "A3"]

        # add tiem to header if it exist in metrics
        if len(metrics) > 9:
            header.append("time")

        # check if the directory exist
        if not os.path.exists(os.path.dirname(self.csv_path)):
            os.makedirs(os.path.dirname(self.csv_path))

        # Check if the file already exists
        file_exists = os.path.isfile(self.csv_path)

        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)

            # Write the metrics
            writer.writerow(metrics)

    def normalize_depth_map(self, depth_map):
        # Normalize the depth map to be between 0 and 1
        depth_map_min = depth_map.min()
        depth_map_max = depth_map.max()
        return (depth_map - depth_map_min) / (depth_map_max - depth_map_min)

    def normalize_image(self, img):
        img = img.astype(np.float32)
        return (img - img.min()) / (img.max() - img.min())

    def tensors_to_image(self, rgb_aif, pred_depth, gt_depth, rel_depth, scale_map, gl_depth, metrics=None, batch_idx=None):
        if (batch_idx % 10 == 0):
            # Convert tensors to numpy arrays
            def tensor_to_numpy(tensor):
                return tensor.squeeze().cpu().numpy()

            # Flatten metrics and convert tensors to numpy arrays
            metrics = metrics.flatten().tolist()

            rgb_aif_np = tensor_to_numpy(rgb_aif)
            pred_depth_np = tensor_to_numpy(pred_depth)
            gt_depth_np = tensor_to_numpy(gt_depth)
            rel_depth_np = tensor_to_numpy(rel_depth)
            scale_map_np = tensor_to_numpy(scale_map)
            gl_depth_np = tensor_to_numpy(gl_depth)

            # Normalize and transpose RGB image if needed
            if isinstance(rgb_aif, torch.Tensor):
                if rgb_aif_np.shape[0] == 3:  # For RGB images
                    rgb_aif_np = np.transpose(rgb_aif_np, (1, 2, 0))
                    rgb_aif_np = self.normalize_image(rgb_aif_np)

            # Determine color range from GT depth
            vmin, vmax = gt_depth_np.min(), gt_depth_np.max()
            if vmin > pred_depth_np.min():
                vmin = pred_depth_np.min()
            if vmax < pred_depth_np.max():
                vmax = pred_depth_np.max()

            # Create a figure with 2 rows and 4 columns
            fig, axes = plt.subplots(2, 6, figsize=(30, 6))

            # First row: Original images with original color maps
            
            axes[0, 0].imshow(rgb_aif_np)
            axes[0, 0].set_title("RGB Image")
            axes[0, 0].axis('off')

            scale_map = axes[0, 1].imshow(scale_map_np, cmap=plt.cm.plasma)
            axes[0, 1].set_title("DFV")
            axes[0, 1].axis('off')
            plt.colorbar(scale_map, ax=axes[0, 1])

            rel_depth = axes[0, 2].imshow(rel_depth_np, cmap=plt.cm.plasma)
            axes[0, 2].set_title("Rel Depth Map")
            axes[0, 2].axis('off')
            plt.colorbar(rel_depth, ax=axes[0, 2])

            gl_depth = axes[0, 3].imshow(gl_depth_np, cmap=plt.cm.plasma)
            axes[0, 3].set_title("Global scaling and shift")
            axes[0, 3].axis('off')
            plt.colorbar(gl_depth, ax=axes[0, 3])

            pred_img = axes[0, 4].imshow(pred_depth_np, cmap=plt.cm.plasma)
            pred_title = "Predicted Depth"
            if metrics[1] is not None:
                pred_title += f"\nRMSE: {metrics[1]:.3f}"
            if metrics[3] is not None:
                pred_title += f"\nAbsRel: {metrics[3]:.3f}"
            if metrics[5] is not None:
                pred_title += f"\nSSIM: {metrics[5]:.3f}"
            axes[0, 4].set_title(pred_title)
            axes[0, 4].axis('off')
            plt.colorbar(pred_img, ax=axes[0, 4])

            gt_img = axes[0, 5].imshow(gt_depth_np, cmap=plt.cm.plasma)
            gt_title = "Ground Truth Depth"
            axes[0, 5].set_title(gt_title)
            axes[0, 5].axis('off')
            plt.colorbar(gt_img, ax=axes[0, 5])

            # Second row: Images with fixed color range based on GT
            axes[1, 0].imshow(rgb_aif_np)
            axes[1, 0].set_title("RGB Image")
            axes[1, 0].axis('off')

            scale_map_fixed = axes[1, 1].imshow(
                scale_map_np, cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
            axes[1, 1].set_title("DFV")
            axes[1, 1].axis('off')
            plt.colorbar(scale_map_fixed, ax=axes[1, 1])

            rel_depth_fixed = axes[1, 2].imshow(
                rel_depth_np, cmap=plt.cm.plasma)
            axes[1, 2].set_title("Rel Depth Map")
            axes[1, 2].axis('off')
            plt.colorbar(rel_depth_fixed, ax=axes[1, 2])

            gl_depth_fixed = axes[1, 3].imshow(
                gl_depth_np, cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
            axes[1, 3].set_title("Global scaling and shift")
            axes[1, 3].axis('off')
            plt.colorbar(gl_depth_fixed, ax=axes[1, 3])

            pred_img_fixed = axes[1, 4].imshow(
                pred_depth_np, cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
            # axes[3].set_title("Predicted Depth")
            pred_title = "Predicted Depth (Fixed color range)"
            axes[1, 4].set_title(pred_title)
            axes[1, 4].axis('off')
            plt.colorbar(pred_img_fixed, ax=axes[1, 4])

            gt_img_fixed = axes[1, 5].imshow(
                gt_depth_np, cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
            axes[1, 5].set_title("Ground Truth Depth (Fixed Range)")
            axes[1, 5].axis('off')
            plt.colorbar(gt_img_fixed, ax=axes[1, 5])

            # Save the image
            plt.savefig(f'{self.img_dir}/val_img{batch_idx}.png',
                        bbox_inches='tight')
            plt.close(fig)
