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

def save_truth_tumor_masks(image, tumor_mask=None, alpha=0.4, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    if tumor_mask is not None:
        plt.imshow(tumor_mask, cmap='Reds', alpha=alpha)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

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

def get_target_conv_layer(model, offset=0):
    """
    Find a specific Conv2d layer in the model by its reverse index.

    Args:
        model (torch.nn.Module): The model to search within.
        offset (int): The offset from the last Conv2d layer. 
                      0 for the last, 1 for the second-to-last, etc.

    Returns:
        torch.nn.Module: The target Conv2d layer.
        
    Raises:
        ValueError: If no Conv2d layer is found at the specified offset.
    """
    conv_counter = 0
    target_layer = None
    
    # Iterate through the model's modules in reverse order
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            if conv_counter == offset:
                target_layer = module
                return target_layer
            conv_counter += 1
            
    # If the loop completes and the layer wasn't found
    raise ValueError(f"Could not find a Conv2d layer with offset {offset}. "
                     f"Only {conv_counter} Conv2d layers were found.")

# ----------------------------
# Helper: run Grad-CAM
# ----------------------------
def run_gradcam(net, input_tensor, pred, target_category=1):
    """
    Run Grad-CAM on segmentation model.

    Args:
        net: trained nn.Module
        input_tensor: torch.Tensor of shape (1,1,H,W)
        pred: np.ndarray (H,W), predicted segmentation mask
        target_category: class index to highlight (default=1 for foreground/tumor)
    """
    # target_layer = get_last_conv_layer(net)
    target_layer = get_target_conv_layer(net, offset=1)
    cam = GradCAM(model=net, target_layers=[target_layer])

    targets = [SemanticSegmentationTarget(target_category, pred)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # (B,H,W)
    return grayscale_cam[0]


# ----------------------------
# Your Inference function
# ----------------------------
def Inference(FLAGS):
    """
    Run inference and save both predictions and Grad-CAM heatmaps.
    """
    csv_data = '/kaggle/input/cect-liver-2/file_check.csv'
    cect_root_dirs = ["/kaggle/input/cect-liver-1", "/kaggle/input/cect-liver-2"]
    mask_dir = "/kaggle/input/cect-liver-2/mask_files/mask_files"

    # unet, mamabunet
    img_size = 256
    # swimunet
    # img_size = 224

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

    pred_save_path = os.path.join(test_save_path, "pred")
    gradcam_save_path = os.path.join(test_save_path, "gradcam")
    # truth_save_path =  os.path.join(test_save_path, "ground_truth")
    os.makedirs(pred_save_path, exist_ok=True)
    os.makedirs(gradcam_save_path, exist_ok=True)
    # os.makedirs(truth_save_path, exist_ok=True)

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(f"/kaggle/input/cect_model_weight/pytorch/default/{FLAGS.labeled_num}", f'{FLAGS.model}_aspp_best_model.pth')
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
        save_file = os.path.join(pred_save_path, f"test_{i}.png")
        # save_truth_file = os.path.join(truth_save_path, f"test_{i}.png")
        save_liver_and_tumor_masks(img_np, tumor_mask=label_np, pred_tumor_mask=pred, save_path=save_file)
        # save_truth_tumor_masks(img_np, tumor_mask=label_np, save_path=save_truth_file)

        # ------------------------
        # 2. Run Grad-CAM
        # ------------------------
        x, y = img_np.shape
        img_resized = zoom(img_np, (img_size / x, img_size / y), order=0)

        input_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float()
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        # Grad-CAM for class=1 (adjust depending on dataset)
        grayscale_cam = run_gradcam(net, input_tensor, pred, target_category=1)

        # Convert grayscale CT to RGB for overlay
        rgb_img = np.float32(cv2.cvtColor((img_resized * 255).astype(np.uint8),
                                          cv2.COLOR_GRAY2RGB)) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Save Grad-CAM visualization
        cam_save_path = os.path.join(gradcam_save_path, f"grad_test_{i}.png")
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
