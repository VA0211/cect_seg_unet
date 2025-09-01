import argparse
import os
import shutil
import pandas as pd
import h5py
import cv2
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt

from networks.net_factory import net_factory
from dataloaders.dataset import LiverTumorSliceDataset
from torch.utils.data import DataLoader

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=140,
                    help='labeled data')
# parser.add_argument('--model_weight_path', type=str,
#                     default='', help='model_weight_path')

def save_liver_and_tumor_masks(image, tumor_mask=None, alpha=0.4, pred_tumor_mask=None, beta=0.5, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    if tumor_mask is not None:
        plt.imshow(tumor_mask, cmap='Reds', alpha=alpha)
    if pred_tumor_mask is not None:
        plt.imshow(pred_tumor_mask, cmap='Blues', alpha=beta)
    # plt.title("CT Slice with Tumor Mask")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def calculate_metric_percase(pred, gt, num_classes=2):
    """
    pred, gt: numpy arrays of shape [H, W] or [D, H, W]
    num_classes: int, total number of classes including background
    """
    metrics = {
        "dice": [],
        "hd95": [],
        "iou": [],
        "acc": [],
        "prec": [],
        "sens": [],
        "spec": [],
    }

    for c in range(1, num_classes):  # skip background class 0
        pred_c = (pred == c).astype(np.uint8)
        gt_c = (gt == c).astype(np.uint8)

        if pred_c.sum() == 0 and gt_c.sum() == 0:
            metrics["dice"].append(1.0)
            metrics["hd95"].append(0.0)
            metrics["iou"].append(1.0)
            metrics["acc"].append(1.0)
            metrics["prec"].append(1.0)
            metrics["sens"].append(1.0)
            metrics["spec"].append(1.0)
            continue
        if pred_c.sum() == 0 or gt_c.sum() == 0:
            metrics["dice"].append(0.0)
            metrics["hd95"].append(999.0)
            metrics["iou"].append(0.0)
            metrics["acc"].append(0.0)
            metrics["prec"].append(0.0)
            metrics["sens"].append(0.0)
            metrics["spec"].append(0.0)
            continue

        try:
            dice = metric.binary.dc(pred_c, gt_c)
            hd95 = metric.binary.hd95(pred_c, gt_c)
            iou = metric.binary.jc(pred_c, gt_c)
            acc = (pred_c == gt_c).sum() / pred_c.size
            prec = metric.binary.precision(pred_c, gt_c)
            sens = metric.binary.sensitivity(pred_c, gt_c)
            spec = metric.binary.specificity(pred_c, gt_c)
        except:
            dice, hd95, iou, acc, prec, sens, spec = [0.0] * 7

        metrics["dice"].append(dice)
        metrics["hd95"].append(hd95)
        metrics["iou"].append(iou)
        metrics["acc"].append(acc)
        metrics["prec"].append(prec)
        metrics["sens"].append(sens)
        metrics["spec"].append(spec)

    # Return per-metric mean across all foreground classes
    return np.array([np.mean(metrics[k]) for k in metrics])

def test_single_volume(image, label, net, classes=2, patch_size=[256, 256]):
    image = image.squeeze().cpu().numpy()   # shape: [H, W]
    label = label.squeeze().cpu().numpy()   # shape: [H, W]

    x, y = image.shape
    image_resized = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
    input_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float().cuda()

    net.eval()
    with torch.no_grad():
        out = net(input_tensor)
        pred = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0).cpu().numpy()
        pred = zoom(pred, (x / patch_size[0], y / patch_size[1]), order=0)

    return pred, calculate_metric_percase(pred, label, num_classes=classes)

# ----------------------------
# Helper: find last conv layer
# ----------------------------
def get_last_conv_layer(model):
    """Find the last Conv2d in the model for Grad-CAM."""
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found in model")


# ----------------------------
# Helper: run Grad-CAM
# ----------------------------
def run_gradcam(net, input_tensor, target_category=1):
    """
    Run Grad-CAM on segmentation model.

    Args:
        net: trained nn.Module
        input_tensor: torch.Tensor of shape (1,1,H,W)
        target_category: class index to highlight (default=1 for foreground/tumor)
    """
    target_layer = get_last_conv_layer(net)
    cam = GradCAM(model=net, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

    targets = [SemanticSegmentationTarget(target_category, None)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # (B,H,W)
    return grayscale_cam[0]


# ----------------------------
# Your Inference function
# ----------------------------
def Inference(FLAGS, net, test_loader, img_size, test_save_path, test_single_volume, save_liver_and_tumor_masks):
    """
    Run inference and save both predictions and Grad-CAM heatmaps.
    """
    csv_data = '/kaggle/input/cect-liver-2/file_check.csv'
    cect_root_dirs = ["/kaggle/input/cect-liver-1", "/kaggle/input/cect-liver-2"]
    mask_dir = "/kaggle/input/cect-liver-2/mask_files/mask_files"

    # unet, mamabunet
    # img_size = 256
    # swimunet
    img_size = 224

    db_test = LiverTumorSliceDataset(
        metadata_csv=csv_data,
        cect_root_dirs=cect_root_dirs,
        mask_dir=mask_dir,
        split="test",  # <--- use test split
        val_ratio=0.2,
        test_ratio=0.1,
        random_seed=42,
        output_size=(img_size, img_size),
        augment=False
    )

    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    snapshot_path = f"/kaggle/working/model_test/{FLAGS.model}"
    test_save_path = f"{snapshot_path}_predictions_plot/"
    os.makedirs(test_save_path, exist_ok=True)

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(f"/kaggle/input/cect_model_weight/pytorch/default/{FLAGS.labeled_num}", f'{FLAGS.model}_best_model.pth')
    print(save_mode_path)
    net.load_state_dict(torch.load(save_mode_path))
    net.eval()
    print(f"Model loaded from: {save_mode_path}")

    all_metrics = []
    metric_names = ["Dice", "HD95", "IoU", "Accuracy", "Precision", "Sensitivity", "Specificity"]

    for i, batch in enumerate(tqdm(test_loader)):
        image = batch["image"]
        label = batch["label"]

        # ------------------------
        # 1. Run prediction
        # ------------------------
        pred, metric_i = test_single_volume(image, label, net, classes=FLAGS.num_classes, patch_size=[img_size, img_size])
        all_metrics.append(metric_i)

        # Plot and save the result
        img_np = image.squeeze().cpu().numpy()
        label_np = label.squeeze().cpu().numpy()
        save_file = os.path.join(test_save_path, f"pred/test_{i}.png")
        save_liver_and_tumor_masks(img_np, tumor_mask=label_np, pred_tumor_mask=pred, save_path=save_file)

        # ------------------------
        # 2. Run Grad-CAM
        # ------------------------
        x, y = img_np.shape
        img_resized = zoom(img_np, (img_size / x, img_size / y), order=0)

        input_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float()
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        # Grad-CAM for class=1 (adjust depending on dataset)
        grayscale_cam = run_gradcam(net, input_tensor, target_category=1)

        # Convert grayscale CT to RGB for overlay
        rgb_img = np.float32(cv2.cvtColor((img_resized * 255).astype(np.uint8),
                                          cv2.COLOR_GRAY2RGB)) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Save Grad-CAM visualization
        cam_save_path = os.path.join(test_save_path, f"gradcam/grad_test_{i}.png")
        cv2.imwrite(cam_save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    df = pd.DataFrame(all_metrics, columns=metric_names)
    # df.insert(0, "Case", [f"sample_{i}" for i in range(len(df))])  # case ID

    # Save per-sample metrics
    csv_save_path = os.path.join(test_save_path, "test_results.csv")
    df.to_csv(csv_save_path, index=False)
    print(f"Saved test metrics to {csv_save_path}")

    # Compute mean of each metric
    mean_metrics = df[metric_names].mean()
    print("Mean Metrics:\n", mean_metrics)

    return mean_metrics.to_dict()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    # print((metric[0]+metric[1]+metric[2])/3)
