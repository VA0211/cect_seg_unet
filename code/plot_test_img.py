import argparse
import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from networks.net_factory import net_factory


def normalize_and_resize(image_slice, target_size=(256, 256)):
    image_slice = (image_slice - np.mean(image_slice)) / (np.std(image_slice) + 1e-8)
    h, w = image_slice.shape
    image_resized = zoom(image_slice, (target_size[0] / h, target_size[1] / w), order=1)
    return image_resized, (h, w)


def restore_size(prediction, original_size):
    return zoom(prediction, (original_size[0] / prediction.shape[0], original_size[1] / prediction.shape[1]), order=0)


def plot_prediction(image, prediction, label=None):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Input Image")
    plt.axis("off")

    if label is not None:
        plt.subplot(1, 3, 2)
        plt.imshow(label, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def run_inference_on_slice(ct_path, mask_path, slice_index, model_path, net_type="unet", num_classes=2):
    net = net_factory(net_type=net_type, in_chns=1, class_num=num_classes)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    ct = nib.load(ct_path).get_fdata()
    ct_slice = ct[:, :, slice_index]
    image_resized, original_size = normalize_and_resize(ct_slice)

    input_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        output = net(input_tensor)
        prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).numpy()

    prediction_resized = restore_size(prediction, original_size)

    # Load and process ground truth if provided
    label_slice = None
    if mask_path and os.path.exists(mask_path):
        mask = nib.load(mask_path).get_fdata()
        label_slice = mask[:, :, slice_index]

    plot_prediction(ct_slice, prediction_resized, label_slice)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ct_path', type=str, required=True, help='Path to the CT .nii.gz file')
    parser.add_argument('--mask_path', type=str, default=None, help='Path to the mask .nii.gz file (optional)')
    parser.add_argument('--slice_index', type=int, default=50, help='Slice index to visualize')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained .pth model')
    parser.add_argument('--net_type', type=str, default='unet', help='Model type, e.g., unet, mambaunet, etc.')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    args = parser.parse_args()

    run_inference_on_slice(
        ct_path=args.ct_path,
        mask_path=args.mask_path,
        slice_index=args.slice_index,
        model_path=args.model_path,
        net_type=args.net_type,
        num_classes=args.num_classes
    )
