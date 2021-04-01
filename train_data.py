# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
import os


# --- Training dataset --- #

def get_dataset(dir):
    if not os.path.isdir(dir):
        raise Exception('check' + dir)
    image_list = os.listdir(dir)
    images = []
    for img in image_list:
        images.append(dir + '/' + img)
    return images
    
class K12_dataset(data.Dataset):
    """Some Information about K12(K15)_dataset"""

    def __init__(self, root, crop_size, single_stereo=False):
        super(K12_dataset, self).__init__()
        # ---- root == './' ---
        self.gt_images = get_dataset(root + '/image_2_3_norain')
        self.rain_images = get_dataset(root + '/image_2_3_rain50')
        self.gt_images2 = get_dataset(root + '/image_3_2_norain')
        self.rain_images2 = get_dataset(root + '/image_3_2_rain50')
        self.crop_size = crop_size
        self.root = root
        self.single_stereo = single_stereo



    def __getitem__(self, index):
        crop_width, crop_height = self.crop_size
        gt_img = Image.open(self.gt_images[index]).convert('RGB')
        haze_img = Image.open(self.rain_images[index]).convert('RGB')
        if self.single_stereo:
            gt_img2 = Image.open(self.gt_images2[index]).convert('RGB')
            haze_img2 = Image.open(self.rain_images2[index]).convert('RGB')

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        if self.single_stereo:
            haze_crop_img2 = haze_img2.crop((x, y, x + crop_width, y + crop_height))
            gt_crop_img2 = gt_img2.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)
        if self.single_stereo:
            haze2 = transform_haze(haze_crop_img2)
            gt2 = transform_gt(gt_crop_img2)

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        if self.single_stereo:
            return haze, haze2, gt, gt2
            
        return haze, gt


    def __len__(self):
        return len(self.gt_images)


class rain_cityscape_dataset(data.Dataset):
    def __init__(self, img_root, gt_root,  crop_size):
        super(rain_cityscape_dataset, self).__init__()
        self.img_root = img_root
        self.gt_root = gt_root
        self.crop_size = crop_size
        self.rain_images, self.gt_images = self.make_city_dataset(img_root, gt_root)


    def make_city_dataset(self, img_root, gt_root):
        names = os.listdir(img_root)
        print("get cities : ", names)
        img_list = []
        gt_list = []
        for name in names:
            imgs = os.listdir(img_root + '/' + name)
            for img in imgs:
                img_list.append(name + '/' + img)
                gt = img.split("_")
                # print(gt)
                gt = '_'.join(gt[:4]) + '.png'
                gt_list.append(name + '/' + gt)
        print("---- img number :{} ----".format(len(img_list)))
        return img_list, gt_list 

    def __getitem__(self, index):
        crop_width, crop_height = self.crop_size
        gt_img = Image.open(self.gt_root + '/' + self.gt_images[index]).convert('RGB')
        haze_img = Image.open(self.img_root + '/' + self.rain_images[index]).convert('RGB')

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

            
        return haze, gt


    def __len__(self):
        return len(self.gt_images)
    


