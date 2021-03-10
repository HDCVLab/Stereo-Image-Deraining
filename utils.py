# --- Imports --- #
import time
import torch
import torch.nn.functional as F
from math import log10
from skimage import measure
import numpy as np
import cv2

def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list
  

def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)
    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [measure.compare_ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]
    return ssim_list


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    Save the tensor in CV2 format
    :param input_tensor: tensor saved
    :param filename : filename saved
    """
    input_tensor = input_tensor[:1]
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor.squeeze()
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


def validationSingle(net, val_data_loader, device, category, save_tag=False):
    psnr_list = []
    ssim_list = []  
    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            haze, gt, name = val_data
            haze = haze.cuda()
            gt = gt.cuda()
            dehaze, coarse_out = net(haze)

        # This is optional : 
        # save_image_tensor2cv2(dehaze,  './rain_cityscape_val_result_200/' + name[0])

        if(batch_id % 100 == 0):
            print("processed %d images" % batch_id)
        
    ret_psnr = sum(psnr_list) / len(psnr_list)
    ret_ssim = sum(ssim_list) / len(ssim_list)
    return ret_psnr, ret_ssim

def stereo_validation(net, val_data_loader, save_dir, device):
    psnr_list = []
    ssim_list = []
    psnr_list2 = []
    ssim_list2 = []

    for batch_id, data in enumerate(val_data_loader):
        with torch.no_grad():
            haze, haze2, gt, gt2, name, name2 = data

            haze = haze.cuda()
            haze2 = haze2.cuda()
            gt = gt.cuda()
            gt2 = gt2.cuda()

            dehaze ,dehaze2 = net(haze, haze2)
        psnr_list.append(to_psnr(dehaze, gt))
        ssim_list.append(to_ssim_skimage(dehaze, gt))
        psnr_list2.append(to_psnr(dehaze2, gt2))
        ssim_list2.append(to_ssim_skimage(dehaze2, gt2))

        # save result
        # save_image_tensor2cv2(dehaze, save_dir + '/image_2_3_rain50/' + name[0].split('/')[-1])
        # save_image_tensor2cv2(dehaze2, save_dir + '/image_3_2_rain50/' + name2[0].split('/')[-1])

        if batch_id % 300 == 0:
            print("processed %d images" % batch_id)

    return np.mean(psnr_list), np.mean(ssim_list), np.mean(psnr_list2), np.mean(ssim_list2)

def adjust_learning_rate(optimizer, epoch, category, lr_decay=0.5):
    # --- Decay learning rate --- #
    step = 20 if category == 'indoor' else 2

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
