# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Single_coarse_single_fine, Single_coarse_stereo_fine
from utils import stereo_validation, validationSingle
from val_data import  K12_testloader, rain_cityscape_dataset
import cv2
import dill
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
from torch.backends import cudnn
import semantic_seg.network
from semantic_seg.datasets import cityscapes, kitti
from semantic_seg.config import infer_cfg

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-semantic', help='If -semantic set, use semantic attetion', action='store_true', default=False)
parser.add_argument('-single', help='If -single set, use single scale', action='store_true', default=False)
parser.add_argument('-single_single', help='If -single set, use single scale', action='store_true', default=False)
parser.add_argument('-single_stereo', help=' If -single_stereo set, use single scale for coarsenet and stereo fine net ', action='store_true', default=False)
parser.add_argument('-share', help=' If -single_stereo set, use single scale for coarsenet and stereo_share fine net ', action='store_true', default=True)
parser.add_argument('-imgmulti', help='If -imgmulti set, use image multi-scale', action='store_true', default=False)
parser.add_argument('-nocf_multi', help='If -nocf_multi set, use multi-scale without coarse-fine', action='store_true', default=False)
args = parser.parse_args()

network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = 'indoor'
semantic = args.semantic
single = args.single
single_single = args.single
imgmulti = args.imgmulti
single_stereo = args.single_stereo
fine_share = args.share


print('--- Hyper-parameters for testing ---')
print('val_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\nnum_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'
      .format(val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss, category))

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if args.semantic:
    infer_cfg(train_mode=False)
    arch = 'semantic_seg.network.deepv3.DeepWV3Plus'
    dataset_cls = kitti
    semantic_extract_net = semantic_seg.network.get_net(arch, dataset_cls, criterion=None)
    #semantic_extract_net = nn.DataParallel(semantic_extract_net, device_ids=device_ids)

    # --- Load semantic model --- #
    try:
        ckpt_path = './kitti_best.pth'
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage, pickle_module=dill)
        state_dict = ckpt['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            #name = 'module.' + k  # add `module.`
            name = k[7:]
            new_state_dict[name] = v
        semantic_extract_net.load_state_dict(new_state_dict)
        #semantic_extract_net.load_state_dict(ckpt['state_dict'])
        print('--- semantic net weight loaded ---')
    except:
        print('--- no semantic weight loaded ---')
    # --- frozen all params of semantic network --- #
    for param in semantic_extract_net.parameters():
        param.requires_grad = False
    semantic_extract_net.eval()

# --- Define the network --- #
if args.single_single:
    print("single single model")
    net = Single_coarse_single_fine(semantic_extract_model=semantic_extract_net, height=1, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)#######
elif args.single_stereo:
    print("single stereo model")
    net = Single_coarse_stereo_fine(semantic_extract_model=semantic_extract_net, height=1, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate, share=fine_share)#######

print(device)
net = net.to(device)

ckpt_path = './checkpoints/checkpoint_K15_fine_share_epoch_50'
ckpt = torch.load(ckpt_path)
net.load_state_dict(ckpt['net'])
print('--- backbone weight loaded ---')

# --- Set category-specific hyper-parameters  --- #
val_data_dir = './K15/test'
val_data_loader = DataLoader(K12_testloader(val_data_dir, single_stereo=single_stereo), batch_size=1, shuffle=False, num_workers=0, drop_last=True)

# --- Use the evaluation model in testing --- #
net.eval()
print('--- Testing starts! ---')
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
if single_stereo:
    val_psnr, val_ssim, val_psnr2, val_ssim2 = stereo_validation(net, val_data_loader, val_data_dir, device)
else:
    val_psnr, val_ssim = validationSingle(net, val_data_loader, val_data_dir, device)

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
print(val_psnr)
print(val_ssim)



