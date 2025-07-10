import argparse
import sys
import os
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from networks.net_factory import net_factory
from config import get_config

# Step 1: Parse custom args (non-Swin/Mamba config args)
custom_parser = argparse.ArgumentParser()
custom_parser.add_argument('--ct_path', type=str, required=True, help="Path to .nii CT image")
custom_parser.add_argument('--mask_path', type=str, default=None, help="Optional: Path to .nii mask")
custom_parser.add_argument('--slice_index', type=int, required=True, help="Slice index to visualize")
custom_parser.add_argument('--model_path', type=str, required=True, help="Path to model .pth file")
custom_parser.add_argument('--net_type', type=str, default='mambaunet', help="Model type")
custom_parser.add_argument('--num_classes', type=int, default=2, help="Number of output classes")
custom_parser.add_argument('--patch_size', type=int, default=256, help="Input image patch size")
custom_args, remaining_args = custom_parser.parse_known_args()

# Step 2: Setup for loading VMamba config
sys.argv = [sys.argv[0]] + remaining_args
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='')
parser.add_argument('--opts', nargs='+', default=[])
FLAGS = parser.parse_args()
FLAGS.num_classes = custom_args.num_classes
FLAGS.patch_size = custom_args.patch_size
if custom_args.net_type == 'mambaunet':
    FLAGS.cfg = "../code/configs/vmamba_tiny.yaml"  # Adjust path as needed
config = get_config(FLAGS)

# Step 3: Load model
net = net_factory(net_type=custom_args.net_type, in_chns=1, class_num=custom_args.num_classes)
net.load_state_dict(torch.load(custom_args.model_path, map_location='cpu'))
net.cuda()
net.eval()

# Step 4: Load image and optional mask
ct_nii = nib.load(custom_args.ct_path)
ct_volume = ct_nii.get_fdata()
slice_idx = custom_args.slice_index
ct_slice = ct_volume[:, :, slice_idx]

if custom_args.mask_path:
    mask_nii = nib.load(custom_args.mask_path)
    mask_volume = mask_nii.get_fdata()
    mask_slice = mask_volume[:, :, slice_idx]
else:
    mask_slice = None

# Step 5: Normalize & Resize input
orig_h, orig_w = ct_slice.shape
ct_slice = (ct_slice - np.mean(ct_slice)) / (np.std(ct_slice) + 1e-8)
ct_resized = zoom(ct_slice, (custom_args.patch_size / orig_h, custom_args.patch_size / orig_w), order=1)

input_tensor = torch.tensor(ct_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

# Step 6: Inference
with torch.no_grad():
    output = net(input_tensor)
    pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()
    pred_resized = zoom(pred, (orig_h / custom_args.patch_size, orig_w / custom_args.patch_size), order=0)

# Step 7: Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(ct_slice, cmap='gray')
plt.title("CT Slice")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(pred_resized, cmap='jet')
plt.title("Model Prediction")
plt.axis('off')

if mask_slice is not None:
    plt.subplot(1, 3, 3)
    plt.imshow(mask_slice, cmap='jet')
    plt.title("Ground Truth")
    plt.axis('off')

plt.tight_layout()
plt.show()
