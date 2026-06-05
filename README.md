# Comparative Study of UNet-based Architectures for Liver Tumor Segmentation

[![Python](https://img.shields.io/badge/Python-3.11.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red.svg)](https://pytorch.org)

---

## Abstract

Segmentation of liver structures in multi-phase contrast-enhanced computed tomography (CECT) plays a crucial role in computer-aided diagnosis and treatment planning. In this study, we investigate the performance of UNet-based architectures for liver tumor segmentation, evaluating ResNet, Transformer-based, and State-space (Mamba) backbones initialized with pretrained weights.

Our comparative analysis reveals that despite the theoretical advantages of modern architectures in modeling long-range dependencies, ResNet-based models demonstrated superior sample efficiency on this dataset. To further improve segmentation quality, we introduced attention mechanisms into the backbone, finding that the **Convolutional Block Attention Module (CBAM)** yields the optimal configuration. The **ResNetUNet3+ with CBAM** achieved the highest nominal performance with a Dice score of **0.755** and IoU of **0.662**, while also delivering the most precise boundary delineation (lowest HD95 of 77.911). Furthermore, Grad-CAM visualizations were employed to enhance model interpretability, providing explainable insights into the network's clinical decision-making process.

---

## Repository Structure

The main codebase is organized as follows, separating the core deep learning logic from dataset configurations and submodules:

```markdown
├── causal-conv1d/           # Submodule for causal convolutions (dependency for Mamba)
├── code/                    # Core source code directory
│   ├── augmentations/       # Data augmentation scripts (e.g., ctaugment.py)
│   ├── configs/             # YAML configurations for Transformer/Mamba backbones
│   ├── dataloaders/         # Dataset loading, preprocessing, and splitting logic
│   ├── networks/            # Model architectures (UNet, UNet3+, ResNetUNet3+, Mamba variants)
│   ├── pretrained_ckpt/     # Directory for pre-trained backbone weights
│   ├── utils/               # Loss functions, evaluation metrics, and helper utilities
│   ├── config.py            # Main config for the project
│   ├── train_2D.py          # Main training entry point
|   ├── val_2D.py            # Main evaluation function
│   ├── test_2D_fully.py     # Main testing script
│   └── test_grad_cam.py     # Script for generating Grad-CAM visualizations
├── mamba/                   # Submodule for Mamba state-space models
├── requirements.txt         # Python dependencies required to run the code
└── README.md                # This documentation file

```

---

## Setup & Usage (Kaggle / Colab Notebooks)

The experiments were originally conducted on Kaggle using an Nvidia Tesla T4 GPU. You can easily reproduce the results in Kaggle or Google Colab environments by following the notebook workflow below.

### 1. Environment Setup

In your notebook cell, clone the repository and install the necessary dependencies:

```python
# Clone the repository with its submodules
!git clone https://github.com/VA0211/cect_seg_unet.git
%cd cect_seg_unet

# Install standard dependencies
!pip install -r requirements.txt

# Install State-Space Model dependencies (Mamba & Causal-Conv1d)
!cd causal-conv1d && pip install .
!cd mamba && pip install .

```

### 2. Dataset Preparation

This project utilizes the **Primary Liver Cancer CECT Imaging Dataset** [Luo et al., 2025]. It contains 278 liver cancer cases and 83 non-liver cancer cases with 3D multi-phase CECT scans (Plain, Arterial, Venous, and Delayed phases).

Download the dataset from [ScienceDB](https://doi.org/10.57760/sciencedb.12207).

*Note: The dataloader automatically handles 2D slice extraction, standardization, resizing to 256x256, and augmentations.*

### 3. Training the Model

Execute the training script directly from a notebook cell. You can pass arguments to select the architecture and hyperparameters:

```python
!python code/train_2D.py --batch_size 4 --epochs 100

```

*(Check `train_2D.py` for specific `argparse` flags to select the model architecture).*

### 4. Evaluation & Visualization

To evaluate a trained model on the testing set and calculate quantitative metrics (Dice, IoU, HD95, Accuracy, Precision, Sensitivity, Specificity):

```python
!python code/test_2D_fully.py

```

To generate **Grad-CAM** localization heatmaps highlighting the model's decision-making process:

```python
!python code/test_grad_cam.py

```

---

## Main Results

Performance comparison of different backbones on the testing set. The proposed **ResNetUNet3+ with CBAM** achieved the best overall stability and boundary precision.

| Model | Param | Dice (DSC) | HD95 | IoU | Accuracy | Specificity |
| --- | --- | --- | --- | --- | --- | --- |
| UNet | 1.8M | 0.506 | 224.710 | 0.491 | 0.786 | 0.789 |
| EfficientUNet | 13.2M | 0.601 | 165.982 | 0.504 | 0.838 | 0.840 |
| SwinUNet | 27.6M | 0.617 | 129.808 | 0.512 | 0.874 | 0.876 |
| MambaUNet | 19.1M | 0.646 | 130.916 | 0.548 | 0.875 | 0.877 |
| MambaUNet3+ | 36.1M | 0.710 | 112.901 | 0.617 | 0.890 | 0.892 |
| TransformerUNet3+ | 53.3M | 0.656 | 159.379 | 0.565 | 0.843 | 0.845 |
| ResNetUNet3+ (baseline) | 31.1M | 0.746 | 88.082 | 0.657 | 0.915 | 0.916 |
| **ResNetUNet3+ with CBAM** | **31.2M** | **0.755** | **77.911** | **0.662** | **0.925** | **0.926** |

*(Note: See the paper for full standard deviation ranges and statistical significance testing results. Cases where the model failed to predict any tumor mask were excluded from the HD95 average to prevent undefined infinity values).*
