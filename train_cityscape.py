# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data import K12_dataset, rain_cityscape_dataset
from model import Single_coarse_single_fine, Single_coarse_stereo_fine
from utils import to_psnr, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
from torchvision.transforms import Compose, ToTensor, Resize
plt.switch_backend('agg')

import cv2
import dill
import numpy as np
from collections import OrderedDict
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
from torch.backends import cudnn
import semantic_seg.network
from semantic_seg.datasets import cityscapes, kitti
from semantic_seg.config import infer_cfg

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    Save the tensor in CV2 format
    :param input_tensor: the tensor name to save
    :param filename: the file name to save
    """
    input_tensor = input_tensor[:1]
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor.squeeze()
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-learning_rate', help='Set the learning rate', default=0.0002, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[240, 240], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=4, type=int)
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-levels', help='Set multi-scale levels of the network', default=3, type=int)
parser.add_argument('-semantic', help='If -semantic set, use semantic attetion', action='store_true', default=False)
parser.add_argument('-single', help='If -single set, use single scale', action='store_true', default=False)
parser.add_argument('-single_single', help='If -single set, use single scale for both coarsenet and fine net ', action='store_true', default=False)
parser.add_argument('-single_stereo', help=' If -single_stereo set, use single scale for coarsenet and stereo fine net ', action='store_true', default=False)
parser.add_argument('-share', help=' If -single_stereo set, use single scale for coarsenet and stereo_share fine net ', action='store_true', default=False)


args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = 'indoor'
semantic = args.semantic
single = args.single
single_single = args.single_single
single_stereo = args.single_stereo
fine_share = args.share

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\n'
      'num_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #

num_epochs = 200
train_data_dir = './rain_cityscape_train'
gt_root = './rain_cityscape_gt' 

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
print(device_ids)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define semantic network --- #
if args.semantic:
    infer_cfg(train_mode=False)
    arch = 'semantic_seg.network.deepv3.DeepWV3Plus'
    dataset_cls = kitti
    semantic_extract_net = semantic_seg.network.get_net(arch, dataset_cls, criterion=None)
    semantic_extract_net = semantic_extract_net.to(device)
    # semantic_extract_net = nn.DataParallel(semantic_extract_net, device_ids=device_ids)

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
        print('--- semantic net weight loaded ---')
    except:
        print('--- no semantic weight loaded ---')
    # --- frozen all params of semantic network --- #
    for param in semantic_extract_net.parameters():
        param.requires_grad = False
    semantic_extract_net.eval()

# --- Define the backbone network --- #
if args.single_single:
    print("single single model")
    net = Single_coarse_single_fine(semantic_extract_model=semantic_extract_net, height=1, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)#######
elif args.single_stereo:
    print("single stereo model")
    net = Single_coarse_stereo_fine(semantic_extract_model=semantic_extract_net, height=1, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate, share=fine_share)#######

net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Load the network weight --- #
try:
    ckpt_path = './'
    ckpt = torch.load(ckpt_path)
    net.load_state_dict(ckpt['net'])
    epoch = ckpt['epoch'] + 1
    print('--- backbone weight loaded ---')
except:
    epoch = 0
    print('--- no weight loaded --- ')


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network = loss_network.to(device)
loss_network.eval()

# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


# --- Load training data and validation/test data --- #

train_data_loader = DataLoader(rain_cityscape_dataset(crop_size=crop_size, img_root=train_data_dir, gt_root=gt_root), batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=True)


start_time = time.time()
print(net.__class__)
while epoch < num_epochs:
    psnr_list = []
    
    adjust_learning_rate(optimizer, epoch, category=category)

    for batch_id, train_data in enumerate(train_data_loader):
        if single_stereo:
            haze, haze2, gt, gt2 = train_data
            haze = haze.to(device)
            gt = gt.to(device)
            haze2 = haze2.to(device)
            gt2 = gt2.to(device)
            # --- Zero the parameter gradients --- #
            optimizer.zero_grad()

            dehaze, dehaze2 = net(haze, haze2)

            total_loss = 0
            # --- calculare total loss --- 
            smooth_loss = F.smooth_l1_loss(dehaze, gt)
            perceptual_loss = loss_network(dehaze, gt)
            smooth_loss2 = F.smooth_l1_loss(dehaze2, gt2)
            perceptual_loss2 = loss_network(dehaze2, gt2)

            total_loss = smooth_loss + lambda_loss * perceptual_loss + smooth_loss2 + lambda_loss * perceptual_loss2

            total_loss.backward()
            optimizer.step()

            if not (batch_id % 100):
                if fine_share:
                    print("share")
                else:
                    print("no share")
                print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))
                print('total_loss = {0}'.format(total_loss))
                print("PSNR: ", end=" ")
        else:
            haze, gt = train_data
            haze = haze.to(device)
            gt = gt.to(device)

            # --- Zero the parameter gradients --- #
            optimizer.zero_grad()

            # --- Forward + Backward + Optimize --- #
            net.train()
            dehaze, coarse_out = net(haze)

            # --- Calculate Total loss --- #
            total_loss = 0
            loss = []
            if single_single:
                smooth_loss = F.smooth_l1_loss(coarse_out, gt)
                perceptual_loss = loss_network(coarse_out, gt)
                if epoch > 100:
                    smooth_loss += F.smooth_l1_loss(dehaze, gt)
                    perceptual_loss += loss_network(dehaze, gt)
                
                total_loss = smooth_loss + lambda_loss * perceptual_loss    
            else:
                for i in range(args.levels):
                    _, _, hi, wi = dehaze[i].size()
                    gt_img = F.interpolate(gt, size=[hi, wi])
                    smooth_loss = F.smooth_l1_loss(dehaze[i], gt_img)
                    perceptual_loss = loss_network(dehaze[i], gt_img)
                    loss.append(smooth_loss + lambda_loss * perceptual_loss)
                    total_loss += loss[i]

            total_loss.backward()
            optimizer.step()
            

            if not (batch_id % 400):
                print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))
                print('total_loss = {0}'.format(total_loss))
                print("PSNR: ", end=" ")
                print('coarse: ', to_psnr(coarse_out, gt))
                print('fine : ', to_psnr(dehaze, gt))
 
    # --- Calculate the average training PSNR in one epoch --- #
    epoch += 1  
    if epoch % 10 == 0:
        state = {'net':net.state_dict(), 'epoch':epoch}
        if fine_share:
            torch.save(state, './checkpoint_raincityscape_monocular_coarseloss_fine_epoch_{}'.format(epoch))
        else:
            torch.save(state, './checkpoint_raincityscape_monocular_coarseloss_fine_epoch_{}'.format(epoch))

    

