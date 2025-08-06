import argparse
import os
import shutil
import pandas as pd
import h5py
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


# def test_single_volume(case, net, test_save_path, FLAGS):
#     h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
#     image = h5f['image'][:]
#     label = h5f['label'][:]
#     prediction = np.zeros_like(label)
#     for ind in range(image.shape[0]):
#         slice = image[ind, :, :]
#         x, y = slice.shape[0], slice.shape[1]
#         slice = zoom(slice, (256 / x, 256 / y), order=0)
#         input = torch.from_numpy(slice).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             if FLAGS.model == "unet_urds":
#                 out_main, _, _, _ = net(input)
#             else:
#                 out_main = net(input)
#             out = torch.argmax(torch.softmax(
#                 out_main, dim=1), dim=1).squeeze(0)




#             out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / 256, y / 256), order=0)
#             prediction[ind] = pred

#     first_metric = calculate_metric_percase(prediction == 1, label == 1)
#     second_metric = calculate_metric_percase(prediction == 2, label == 2)
#     third_metric = calculate_metric_percase(prediction == 3, label == 3)

#     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#     img_itk.SetSpacing((1, 1, 10))
#     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#     prd_itk.SetSpacing((1, 1, 10))
#     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#     lab_itk.SetSpacing((1, 1, 10))
#     sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
#     sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
#     sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
#     return first_metric, second_metric, third_metric

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

# def Inference(FLAGS):
#     with open(FLAGS.root_path + '/test.list', 'r') as f:
#         image_list = f.readlines()
#     image_list = sorted([item.replace('\n', '').split(".")[0]
#                          for item in image_list])
#     # snapshot_path = "../model/{}_{}_labeled/{}".format(
#     snapshot_path = "../model/{}_{}/{}".format(        
#         FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
#     # test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
#     test_save_path = "../model/{}_{}/{}_predictions/".format(        
#         FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
#     if os.path.exists(test_save_path):
#         shutil.rmtree(test_save_path)
#     os.makedirs(test_save_path)
#     net = net_factory(net_type=FLAGS.model, in_chns=1,
#                       class_num=FLAGS.num_classes)
#     save_mode_path = os.path.join(
#         snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
#     print(save_mode_path)
#     net.load_state_dict(torch.load(save_mode_path))
#     print("init weight from {}".format(save_mode_path))
#     net.eval()

#     first_total = 0.0
#     second_total = 0.0
#     third_total = 0.0
#     for case in tqdm(image_list):
#         first_metric, second_metric, third_metric = test_single_volume(
#             case, net, test_save_path, FLAGS)
#         first_total += np.asarray(first_metric)
#         second_total += np.asarray(second_metric)
#         third_total += np.asarray(third_metric)
#     avg_metric = [first_total / len(image_list), second_total /
#                   len(image_list), third_total / len(image_list)]
#     return avg_metric

def Inference(FLAGS):

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

    snapshot_path = f"{FLAGS.exp}_{FLAGS.labeled_num}/{FLAGS.model}"
    test_save_path = f"{snapshot_path}_predictions_plot/"
    os.makedirs(test_save_path, exist_ok=True)

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    # if FLAGS.model_weight_path == '':
    #     save_mode_path = os.path.join(snapshot_path, f'{FLAGS.model}_best_model.pth')
    # else:
    #     save_mode_path = FLAGS.model_weight_path
    save_mode_path = os.path.join(snapshot_path, f'{FLAGS.model}_best_model.pth')
    print(save_mode_path)
    net.load_state_dict(torch.load(save_mode_path))
    net.eval()
    print(f"Model loaded from: {save_mode_path}")

    all_metrics = []
    metric_names = ["Dice", "HD95", "IoU", "Accuracy", "Precision", "Sensitivity", "Specificity"]

    # for i, (image, label) in enumerate(tqdm(test_loader)):
    for i, batch in enumerate(tqdm(test_loader)):
        image = batch["image"]
        label = batch["label"]
        pred, metric_i = test_single_volume(image, label, net, classes=FLAGS.num_classes, patch_size=[img_size, img_size])
        all_metrics.append(metric_i)

        # Plot and save the result
        img_np = image.squeeze().cpu().numpy()
        label_np = label.squeeze().cpu().numpy()
        save_file = os.path.join(test_save_path, f"pred_test_{i}.png")
        save_liver_and_tumor_masks(img_np, tumor_mask=label_np, pred_tumor_mask=pred, save_path=save_file)

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
