import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np

# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, crop_size,val_data_dir,val_filename):
        super().__init__()
        val_list = val_data_dir + val_filename
        with open(val_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input','gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.crop_size = crop_size[0]
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        
        try:
            input_img = Image.open(self.val_data_dir + input_name.replace('jpg', 'png')).convert('RGB')
        except:
            input_img = Image.open(self.val_data_dir + input_name.replace('png', 'jpg')).convert('RGB')
        try:
            gt_img = Image.open(self.val_data_dir + gt_name.replace('jpg', 'png')).convert('RGB')
        except:
            gt_img = Image.open(self.val_data_dir + gt_name.replace('png', 'jpg')).convert('RGB')
        # input_img = input_img.resize((wd_new,ht_new), Image.LANCZOS)
        # gt_img = gt_img.resize((wd_new, ht_new), Image.LANCZOS)
        input_img = input_img.resize((self.crop_size ,self.crop_size ), Image.LANCZOS)
        gt_img = gt_img.resize((self.crop_size , self.crop_size ), Image.LANCZOS)
        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)

        return input_im, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
