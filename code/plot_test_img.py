import argparse
import os
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from networks.net_factory import net_factory  # Make sure this returns your model


def load_nifti_slice(path, slice_idx, output_size=(256, 256)):
    nii = nib.load(path)
    data = nii.get_fdata()
    if data.ndim == 4:
        data = data[:, :, :, 0]
    slice_img = data[:, :, slice_idx]
    slice_img = (slice_img - np.mean(slice_img)) / (np.std(slice_img) + 1e-8)
    resized = zoom(slice_img, (output_size[0] / slice_img.shape[0], output_size[1] / slice_img.shape[1]), order=1)
    return resized


def plot_prediction(ct_tensor, pred_tensor, mask_tensor=None):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3 if mask_tensor is not None else 2, 1)
    plt.title("CT Slice")
    plt.imshow(ct_tensor.squeeze(0), cmap='gray')
    plt.axis('off')

    if mask_tensor is not None:
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(mask_tensor.squeeze(0), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
    else:
        plt.subplot(1, 2, 2)
        plt.title("Prediction")

    plt.imshow(pred_tensor.squeeze(0), cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ct_path', type=str, required=True, help="Path to CT .nii or .nii.gz file")
    parser.add_argument('--mask_path', type=str, default=None, help="Optional: Path to mask .nii file")
    parser.add_argument('--slice_index', type=int, required=True, help="Slice index to visualize")
    parser.add_argument('--model_path', type=str, required=True, help="Path to .pth model checkpoint")
    parser.add_argument('--net_type', type=str, default='unet', help="Model type used in net_factory")
    parser.add_argument('--num_classes', type=int, default=2, help="Number of output classes")
    args, unknown = parser.parse_known_args()  # <-- this fixes unrecognized arg crash

    # Load image slice
    ct_slice = load_nifti_slice(args.ct_path, args.slice_index)  # (256, 256)
    ct_tensor = torch.tensor(ct_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()  # [1, 1, H, W]

    # Load mask if available
    mask_tensor = None
    if args.mask_path:
        mask_slice = load_nifti_slice(args.mask_path, args.slice_index)
        mask_tensor = torch.tensor(mask_slice, dtype=torch.uint8).unsqueeze(0)

    # Load model
    model = net_factory(net_type=args.net_type, in_chns=1, class_num=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    model.eval()

    # Run prediction
    with torch.no_grad():
        output = model(ct_tensor)
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu()

    # Plot result
    plot_prediction(ct_tensor.cpu().squeeze(), pred, mask_tensor)


if __name__ == '__main__':
    main()
