"""Export test-set predictions (all slices) and Grad-CAM maps (selected slices) of a trained model.

Output .npz: files, preds (np.packbits of the 256x256 binary masks, axis=-1), dice (per slice),
cam_files, cams (float16, 256x256), cam_target ('pred' or 'gt' when the prediction is empty).
Predictions are produced exactly as in test_2D_fully.py (nearest-neighbour resize to --img_size and back).

Usage:
python export_predictions.py --model unet3plus_resnet_attn --decoder_attn cbam --aspp 0 --img_size 256 \
    --weights cbam_best_model.pth --data_root .../npy_dataset --cam_files example_pool.txt --out cbam.npz
"""
import argparse
import os

import numpy as np
import torch
from scipy.ndimage import zoom
from tqdm import tqdm

from dataloaders.dataset import NpyDataset
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--decoder_attn', type=str, default='none')
parser.add_argument('--aspp', type=int, default=0)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--data_root', type=str, default='/kaggle/input/cect-npy-full/npy_dataset')
parser.add_argument('--cam_files', type=str, default='', help='text file with one test file name per line')
parser.add_argument('--cam_offset', type=int, default=2, help='Grad-CAM target: n-th Conv2d from the end')
parser.add_argument('--out', type=str, required=True)
FLAGS, _ = parser.parse_known_args()


def predict(net, image, size):
    """image: (H, W) float32 -> binary (H, W) mask, as test_single_volume in test_2D_fully.py."""
    x, y = image.shape
    inp = zoom(image, (size / x, size / y), order=0)
    with torch.no_grad():
        out = net(torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().cuda())
        pred = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0).cpu().numpy()
    return zoom(pred, (x / size, y / size), order=0).astype(np.uint8)


def dice(pred, gt):
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(pred, gt).sum() / s


def target_conv(net, offset):
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)]
    return convs[-1 - offset]


def main():
    attn = None if FLAGS.decoder_attn.lower() == 'none' else FLAGS.decoder_attn
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes,
                      decoder_attn=attn, apply_aspp=bool(FLAGS.aspp))
    net.load_state_dict(torch.load(FLAGS.weights, map_location='cuda'))
    net.eval()

    db = NpyDataset(FLAGS.data_root, split='test', augment=False, output_size=(256, 256))
    files = db.file_list
    preds, dices = [], []
    for i in tqdm(range(len(db)), mininterval=30):
        s = db[i]
        img = s['image'].squeeze().numpy()
        gt = s['label'].numpy().astype(np.uint8)
        p = predict(net, img, FLAGS.img_size)
        preds.append(np.packbits(p.astype(bool), axis=-1))
        dices.append(dice(p, gt))
    print(f'{FLAGS.model}: mean slice Dice = {np.mean(dices):.4f} over {len(dices)} slices '
          f'(should match test_results.csv)')

    cam_files, cams, cam_target = [], [], []
    if FLAGS.cam_files:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
        wanted = [l.strip() for l in open(FLAGS.cam_files) if l.strip()]
        index = {f: k for k, f in enumerate(files)}
        try:
            cam = GradCAM(model=net, target_layers=[target_conv(net, FLAGS.cam_offset)])
            for f in wanted:
                s = db[index[f]]
                img = s['image'].squeeze().numpy()
                gt = s['label'].numpy().astype(np.uint8)
                p = np.unpackbits(preds[index[f]], axis=-1)[:, :256]
                mask, tgt = (p, 'pred') if p.sum() > 0 else (gt, 'gt')
                size = FLAGS.img_size
                inp = zoom(img, (size / 256, size / 256), order=0)
                m_in = zoom(mask, (size / 256, size / 256), order=0).astype(np.float32)
                g = cam(input_tensor=torch.from_numpy(inp)[None, None].float().cuda(),
                        targets=[SemanticSegmentationTarget(1, m_in)])[0]
                cams.append(zoom(g, (256 / size, 256 / size), order=1).astype(np.float16))
                cam_files.append(f)
                cam_target.append(tgt)
        except Exception as e:  # Grad-CAM is optional; keep the predictions
            print('Grad-CAM failed:', repr(e))

    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.out)), exist_ok=True)
    np.savez_compressed(FLAGS.out, files=np.array(files), preds=np.stack(preds), dice=np.array(dices),
                        cam_files=np.array(cam_files), cams=np.stack(cams) if cams else np.zeros((0, 256, 256)),
                        cam_target=np.array(cam_target))
    print('saved', FLAGS.out, f'({len(cams)} Grad-CAM maps)')


if __name__ == '__main__':
    main()
