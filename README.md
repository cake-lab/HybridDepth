<div align="center">
<h1>Hybrid Depth: Robust Depth Fusion </br> By Leveraging Depth from Focus and Single-Image Priors</h1>

[**Ashkan Ganj**](https://ashkanganj.me/)<sup>1</sup> · [**Hang Su**](https://suhangpro.github.io/)<sup>2</sup> · [**Tian Guo**](https://tianguo.info/)<sup>1</sup>

<sup>1</sup>Worcester Polytechnic Institute
&emsp;&emsp;&emsp;<sup>2</sup>Nvidia Research

<a href="https://arxiv.org/pdf/2407.18443"><img src='https://img.shields.io/badge/arXiv-Hybrid Depth-red' alt='arXiv'></a>
<a href="https://ieeexplore.ieee.org/document/10765280"><img src='https://img.shields.io/badge/ISMAR24-Poster-blue'></a>
<a href="https://huggingface.co/AshkanGanj/HybridDepth"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Link-yellow'></a>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hybriddepth-robust-depth-fusion-for-mobile-ar/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=hybriddepth-robust-depth-fusion-for-mobile-ar)

</div>

<div align="center">
  <p style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; width:40%" >
  📢 We released an improved version of HybridDepth, now available with new features and optimized performance!
  </p>


This work presents HybridDepth. HybridDepth is a practical depth estimation solution based on focal stack images captured from a camera. This approach outperforms state-of-the-art models across several well-known datasets, including NYU V2, DDFF12, and ARKitScenes.

![teaser](assets/teaser.png)
</div>

## 📢 News

- **2024-10-30**:  Released **version 2** of HybridDepth with improved performance and pre-trained weights.
- **2024-10-30**: Integrated support for TorchHub for easy model loading and inference.
- **2024-07-25**: Initial release of pre-trained models.
- **2024-07-23**: GitHub repository and HybridDepth model went live.


## 🚀 Usage

### Colab Notebook Starter File

Quickly get started with HybridDepth using the [Colab notebook](https://colab.research.google.com/github/cake-lab/HybridDepth/blob/main/notebooks/HybridDepthStarterNB.ipynb).

### Using TorchHub

You can select a pre-trained model directly with TorchHub.

Available Pre-trained Models:

* `HybridDepth_NYU5`: Pre-trained on the NYU Depth V2 dataset using a 5-focal stack input, with both the DFF branch and refinement layer trained.
* `HybridDepth_NYU10`: Pre-trained on the NYU Depth V2 dataset using a 10-focal stack input, with both the DFF branch and refinement layer trained.
* `HybridDepth_DDFF5`: Pre-trained on the DDFF dataset using a 5-focal stack input.
* `HybridDepth_NYU_PretrainedDFV5`: Pre-trained only on the refinement layer with NYU Depth V2 dataset using a 5-focal stack, following pre-training with DFV.

```python
model_name = 'HybridDepth_NYU_PretrainedDFV5' #change this
model = torch.hub.load('cake-lab/HybridDepth', model_name , pretrained=True)
model.eval()
```

### Local Installation

1. **Clone the repository and install the dependencies:**

```bash
git clone https://github.com/cake-lab/HybridDepth.git
cd HybridDepth
conda env create -f environment.yml
conda activate hybriddepth
```

2. **Download Pre-Trained Weights:**

Download the weights for the model from the links below and place them in the `checkpoints` directory:

- [HybridDepth_NYU_FocalStack5](https://github.com/cake-lab/HybridDepth/releases/download/v2.0/NYUBest5-DFV-Trained.ckpt)
- [HybridDepth_NYU_FocalStack10](https://github.com/cake-lab/HybridDepth/releases/download/v2.0/NYUBest10-DFV-Trained.ckpt)
- [HybridDepth_DDFF_FocalStack5](https://github.com/cake-lab/HybridDepth/releases/download/v2.0/DDFF12.ckpt)
- [HybridDepth_NYU_PretrainedDFV_FocalStack5](https://github.com/cake-lab/HybridDepth/releases/download/v2.0/NyuBest5.ckpt)

3. **Prediction**

For inference, you can run the following code:

```python
# Load the model checkpoint
model_path = 'checkpoints/NYUBest5.ckpt'
model = DepthNetModule.load_from_checkpoint(model_path)
model.eval()
model = model.to('cuda')
```

After loading the model, use the following code to process the input images and get the depth map:

_Note_: Currently, the `prepare_input_image` function only supports `.jpg` images. Modify the function if you need support for other image formats.

```python
from utils.io import prepare_input_image

data_dir = 'focal stack images directory' # Path to the focal stack images in a folder

# Load the focal stack images
focal_stack, rgb_img, focus_dist = prepare_input_image(data_dir)

# Run inference
with torch.no_grad():
   out = model(rgb_img, focal_stack, focus_dist)

metric_depth = out[0].squeeze().cpu().numpy() # The metric depth
```

### 🧪 Evaluation

Please first Download the weights for the model from the links below and place them in the `checkpoints` directory:

- [HybridDepth_NYU_FocalStack5](https://github.com/cake-lab/HybridDepth/releases/download/v2.0/NYUBest5-DFV-Trained.ckpt)
- [HybridDepth_NYU_FocalStack10](https://github.com/cake-lab/HybridDepth/releases/download/v2.0/NYUBest10-DFV-Trained.ckpt)
- [HybridDepth_DDFF_FocalStack5](https://github.com/cake-lab/HybridDepth/releases/download/v2.0/DDFF12.ckpt)
- [HybridDepth_NYU_PretrainedDFV_FocalStack5](https://github.com/cake-lab/HybridDepth/releases/download/v2.0/NyuBest5.ckpt)

#### Dataset Preparation

1. **NYU Depth V2**: Download the dataset following the instructions provided [here](https://github.com/cleinc/bts/tree/master/pytorch#nyu-depvh-v2).
2. **DDFF12**: Download the dataset following the instructions provided [here](https://github.com/fuy34/DFV).
3. **ARKitScenes**: Download the dataset following the instructions provided [here](https://github.com/cake-lab/Mobile-AR-Depth-Estimation).

Set up the configuration file `config.yaml` in the `configs` directory. Pre-configured files for each dataset are available in the `configs` directory, where you can specify paths, model settings, and other hyperparameters. Here’s an example configuration:

```yaml
data:
  class_path: dataloader.dataset.NYUDataModule # Path to your dataloader module in dataset.py
  init_args:
    nyuv2_data_root: "path/to/NYUv2" # Path to the specific dataset
    img_size: [480, 640] # Adjust based on your DataModule requirements
    remove_white_border: True
    num_workers: 0 # Set to 0 if using synthetic data
    use_labels: True

model:
  invert_depth: True # Set to True if the model outputs inverted depth

ckpt_path: checkpoints/checkpoint.ckpt
```

Specify the configuration file in the `test.sh` script:

```bash
python cli_run.py test --config configs/config_file_name.yaml
```

Then, execute the evaluation with:

```bash
cd scripts
sh evaluate.sh
```

---

### 🏋️ Training

#### Install Synthetic CUDA Package

Install the required CUDA-based package for image synthesis:

```bash
python utils/synthetic/gauss_psf/setup.py install
```

This installs the package necessary for synthesizing images.

#### Configuration for Training

Set up the configuration file `config.yaml` in the `configs` directory, specifying the dataset path, batch size, and other training parameters. Below is a sample configuration for training with the NYUv2 dataset:

```yaml
model:
  invert_depth: True
  # learning rate
  lr: 3e-4 # Adjust as needed
  # weight decay
  wd: 0.001 # Adjust as needed

data:
  class_path: dataloader.dataset.NYUDataModule # Path to your dataloader module in dataset.py
  init_args:
    nyuv2_data_root: "path/to/NYUv2" # Dataset path
    img_size: [480, 640] # Adjust for NYUDataModule
    remove_white_border: True
    batch_size: 24 # Adjust based on available memory
    num_workers: 0 # Set to 0 if using synthetic data
    use_labels: True
ckpt_path: null
```

Specify the configuration file in the `train.sh` script:

```bash
python cli_run.py train --config configs/config_file_name.yaml
```

Execute the training command:

```bash
cd scripts
sh train.sh
```

## 📖 Citation

If our work assists you in your research, please cite it as follows:

```Bibtex
@misc{ganj2024hybriddepthrobustmetricdepth,
      title={HybridDepth: Robust Metric Depth Fusion by Leveraging Depth from Focus and Single-Image Priors},
      author={Ashkan Ganj and Hang Su and Tian Guo},
      year={2024},
      eprint={2407.18443},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.18443},
}
```
