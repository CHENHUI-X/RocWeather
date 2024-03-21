
import torch
import argparse

from torch.utils.data import DataLoader
from  torchsummary import summary
from utils.val_data_functions import ValData
from utils.utils import PSNR , SSIM , load_best_model,  save_img
import numpy as np
import random
from tqdm import tqdm
from models.RecWeather import RecWeather
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for visualization')
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str,default='./checkpoint')
parser.add_argument('--time_str', help='timestamp', default=None, type=str)
parser.add_argument('-seed', help='set random seed', default=666, type=int)


args = parser.parse_args()

crop_size = args.crop_size
val_batch_size = 1 
exp_name = args.exp_name
time_str = args.time_str
assert  time_str , 'Must specify the time stamp of the model !'

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

val_data_dir = './data/test/'
val_filename = 'test.txt'
val_data_name = 'allweather'
output_dir = "./data/test/pred/"
os.makedirs(output_dir, exist_ok=True)
    
def SaveImages():
    # --- Load validation/test data --- #
    val_data_loader = DataLoader(ValData(crop_size,val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False)

    # --- Gpu device --- #
    if torch.cuda.is_available():
        GPU = True
        device = torch.device("cuda:0")
    else:
        GPU = False
        device = torch.device("cpu")
        print("Using CPU device")

    net = RecWeather().to(device).eval()
    net = load_best_model(net, exp_name = exp_name , time_str = time_str )# GPU or CPU

    # -----Some parameters------
    total_step = 0
    step = 0
    lendata = len(val_data_loader)
    psnr = PSNR()
    ssim = SSIM()
    eval_psnr = []
    eval_ssim = []
    pred_images_list = []
    img_names_list = []
    loop = tqdm(val_data_loader, desc="--- Progress bar : ")
    with torch.no_grad():
        for batch_id, val_data in enumerate(loop):
            input_image, gt, img_names = val_data
            input_image = input_image.to(device)
            gt = gt.to(device)
            # save image
            pred_image = net(input_image).to('cpu')
            step_psnr, step_ssim = \
                psnr.to_psnr(pred_image.detach(), gt.detach()), ssim.to_ssim(pred_image.detach(), gt.detach())
            eval_psnr.append(step_psnr.to('cpu'))
            eval_ssim.append(step_ssim.to('cpu'))
            save_img(img_names, pred_image)
            return 
            
        print('='*50)
        print(
            '--- The {0} dataset psnr is : {1:.3f} , ssim is : {2:.3f} , and processed image have saved .'.format(
                val_data_name , np.mean(eval_psnr), np.mean(eval_ssim)
            )
        )
        print('='*50)


def get_activation(name,activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def visualization_map():
    activation = {}
    val_data_loader = DataLoader(ValData(crop_size,val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False)
    # --- Gpu device --- #
    if torch.cuda.is_available():
        GPU = True
        device = torch.device("cuda:0")
    else:
        GPU = False
        device = torch.device("cpu")
        print("Using CPU device")

    net = RecWeather().to(device).eval()
    summary(net, input_size=(3, 256, 256), device='cuda')
    net = load_best_model(net, exp_name = exp_name , time_str = time_str )# GPU or CPU
    
    #register hook
    # model_children = list(net.children())
    # model_children[0][0].register_forward_hook(get_activation('Task_Attention')) # register the output
    # print(len(model_children))
    # for layer in model_children : print(layer)
    
    # loop = tqdm(val_data_loader, desc="--- Progress bar : ")
    # with torch.no_grad():
    #     for batch_id, val_data in enumerate(loop):
    #         input_image, gt, img_names = val_data
    #         input_image = input_image.to(device)
    #         gt = gt.to(device)
    #         # save image
    #         pred_image = net(input_image).to('cpu')
    #         return 


if __name__ == '__main__':
    # SaveImages()
    visualization_map()