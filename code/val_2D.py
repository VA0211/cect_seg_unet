import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         return dice, hd95
#     else:
#         return 0, 0

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


# def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
#     image, label = image.squeeze(0).cpu().detach(
#     ).numpy(), label.squeeze(0).cpu().detach().numpy()
#     prediction = np.zeros_like(label)
#     for ind in range(image.shape[0]):
#         slice = image[ind, :, :]
#         x, y = slice.shape[0], slice.shape[1]
#         slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
#         input = torch.from_numpy(slice).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(
#                 net(input), dim=1), dim=1).squeeze(0)
#             out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#             prediction[ind] = pred
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(
#             prediction == i, label == i))
#     return metric_list

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

    return calculate_metric_percase(pred, label, num_classes=classes)

def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
