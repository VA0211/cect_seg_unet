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
from dataloaders.dataset import LiverTumorSliceDataset, LiverTumorSliceDatasetPatient, LiverTumorPatientSliceDataset, NpyDataset
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
parser.add_argument('--data_root', type=str, default='/kaggle/input/cect-npy-full/npy_dataset',
                    help='patient-level npy dataset ({split}_{patient}_{phase}_{slice}.npy)')
parser.add_argument('--img_size', type=int, default=256,
                    help='network input size (224 for Swin-based models)')
parser.add_argument('--eval_size', type=int, default=256,
                    help='resolution at which test metrics are computed')
parser.add_argument('--decoder_attn', type=str, default='none',
                    help='unet3plus_resnet_attn only: none | se | cbam | mcbam')
parser.add_argument('--aspp', type=int, default=0,
                    help='unet3plus_resnet_attn only: 1 to apply ASPP at the bottleneck')
parser.add_argument('--save_png', type=int, default=0,
                    help='1 to save an overlay image for every test slice')
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
    img_size = FLAGS.img_size

    # Patient-level test split; slices are kept in sorted file order (shuffle=False)
    db_test = NpyDataset(FLAGS.data_root, split='test', augment=False,
                         output_size=(FLAGS.eval_size, FLAGS.eval_size))
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    files = db_test.file_list
    print(f"[TEST] {len({f.split('_')[1] for f in files})} patients, {len(files)} slices")

    snapshot_path = f"{FLAGS.exp}_{FLAGS.labeled_num}/{FLAGS.model}"
    test_save_path = f"{snapshot_path}_predictions_plot/"
    os.makedirs(test_save_path, exist_ok=True)

    attn = None if FLAGS.decoder_attn.lower() == 'none' else FLAGS.decoder_attn
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes,
                      decoder_attn=attn, apply_aspp=bool(FLAGS.aspp))
    save_mode_path = os.path.join(snapshot_path, f'{FLAGS.model}_best_model.pth')
    print(save_mode_path)
    net.load_state_dict(torch.load(save_mode_path))
    net.eval()
    print(f"Model loaded from: {save_mode_path}")

    all_metrics = []
    empty_pred = []
    metric_names = ["Dice", "HD95", "IoU", "Accuracy", "Precision", "Sensitivity", "Specificity"]

    for i, batch in enumerate(tqdm(test_loader)):
        image = batch["image"]
        label = batch["label"]
        pred, metric_i = test_single_volume(image, label, net, classes=FLAGS.num_classes, patch_size=[img_size, img_size])
        all_metrics.append(metric_i)
        empty_pred.append(int((pred == 1).sum() == 0))

        if FLAGS.save_png:
            img_np = image.squeeze().cpu().numpy()
            label_np = label.squeeze().cpu().numpy()
            save_file = os.path.join(test_save_path, f"pred_test_{i}.png")
            save_liver_and_tumor_masks(img_np, tumor_mask=label_np, pred_tumor_mask=pred, save_path=save_file)

    df = pd.DataFrame(all_metrics, columns=metric_names)
    df.insert(0, "file", files)
    df.insert(1, "patient_id", [f.split('_')[1] for f in files])
    df.insert(2, "phase", [f.split('_')[2] for f in files])
    df["empty_pred"] = empty_pred
    # HD95 is undefined for empty predictions (stored as 999 by calculate_metric_percase)
    df.loc[df["empty_pred"] == 1, "HD95"] = np.nan

    csv_save_path = os.path.join(test_save_path, "test_results.csv")
    df.to_csv(csv_save_path, index=False)
    print(f"Saved test metrics to {csv_save_path}")

    mean_metrics = df[metric_names].mean()
    print("Slice-level mean (HD95 over non-empty predictions):\n", mean_metrics)
    print(f"Empty predictions: {int(df['empty_pred'].sum())} / {len(df)} ({100 * df['empty_pred'].mean():.1f}%)")
    per_patient = df.groupby("patient_id")[metric_names].mean()
    print("Patient-level mean:\n", per_patient.mean())

    return mean_metrics.to_dict()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    # print((metric[0]+metric[1]+metric[2])/3)
