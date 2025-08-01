import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.vision_mamba import MambaUnet as VIM_seg

from config import get_config

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator, BaseDataSets_Synapse, LiverTumorSliceDataset
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

parser.add_argument(
    '--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')


parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=140,
                    help='labeled data')
args = parser.parse_args()


config = get_config(args)

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    # labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

    # model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)




    model = VIM_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    # model.load_from(config)




    # db_train = BaseDataSets(base_dir=args.root_path, split="train", num=labeled_slice, transform=transforms.Compose([
    #     RandomGenerator(args.patch_size)
    # ]))
    # db_val = BaseDataSets(base_dir=args.root_path, split="val")

    csv_data = '/kaggle/input/cect-liver-2/file_check.csv'
    cect_root_dirs=["/kaggle/input/cect-liver-1", "/kaggle/input/cect-liver-2"]
    mask_dir="/kaggle/input/cect-liver-2/mask_files/mask_files"

    # Train set
    db_train = LiverTumorSliceDataset(
        metadata_csv=csv_data,
        cect_root_dirs=cect_root_dirs,
        mask_dir=mask_dir,
        split="train",
        val_ratio=0.2,
        test_ratio=0.1,
        random_seed=42,
        output_size=(256, 256),
        augment=True
    )

    # Validation set
    db_val = LiverTumorSliceDataset(
        metadata_csv=csv_data,
        cect_root_dirs=cect_root_dirs,
        mask_dir=mask_dir,
        split="val",
        val_ratio=0.2,
        test_ratio=0.1,
        random_seed=42,
        output_size=(256, 256),
        augment=False
    )

    # Test set
    # db_test = LiverTumorSliceDataset(
    #     metadata_csv=csv_data,
    #     cect_root_dirs=cect_root_dirs,
    #     mask_dir=mask_dir,
    #     split="test",
    #     val_ratio=0.2,
    #     test_ratio=0.1,
    #     random_seed=42,
    #     output_size=(256, 256),
    #     augment=False
    # )


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = []
                for i_batch, sampled_batch in enumerate(valloader):
                    # metric_i = test_single_volume(
                    #     sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, patch_size=args.patch_size)
                    # metric_i = test_single_volume(sampled_batch["image"][0, 0].cpu().numpy(),
                    #                               sampled_batch["label"][0].cpu().numpy(),
                    #                               model, classes=num_classes, patch_size=args.patch_size)
                    # metric_list += np.array(metric_i)
                    metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model,
                        classes=num_classes,
                        patch_size=args.patch_size
                    )
                    metric_list.append(metric_i)
                # metric_list = metric_list / len(db_val)
                # for class_i in range(num_classes-1):
                #     writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                #                       metric_list[class_i, 0], iter_num)
                #     writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                #                       metric_list[class_i, 1], iter_num)

                # performance = np.mean(metric_list, axis=0)[0]

                # mean_hd95 = np.mean(metric_list, axis=0)[1]
                # writer.add_scalar('info/val_mean_dice', performance, iter_num)
                # writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                metric_list = np.array(metric_list)
                mean_metrics = np.mean(metric_list, axis=0)

                metric_names = ["dice", "hd95", "iou", "acc", "prec", "sens", "spec"]
                for i, name in enumerate(metric_names):
                    writer.add_scalar(f'info/val_{name}', mean_metrics[i], iter_num)

                performance = mean_metrics[0]  # Dice
                mean_hd95 = mean_metrics[1]
                mean_iou = mean_metrics[2]

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f mean_iou : %f' % (iter_num, performance, mean_hd95, mean_iou))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
