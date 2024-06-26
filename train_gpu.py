import os
from torch.profiler import profile, record_function, ProfilerActivity

os.environ[ 'CUDA_VISIBLE_DEVICES' ] = "0,1,2,3,4,5,6,7"
import datetime
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.switch_backend( 'agg' )

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.train_data_functions import TrainData
from utils.val_data_functions import ValData

from utils.utils import PSNR, SSIM, validation_gpu, Logger, synthetic_loss
from torchvision.models import convnext_tiny ,vgg16
from utils.perceptual import LossNetwork
import torchvision
import numpy as np
import random
from tqdm import tqdm
# from transweather_model import SwingTransweather
from models.RecWeather import RecWeather
# from models.srformer import SRFormer
# from models.FocalResWeather import SwingTransweather

# ================================ Parse hyper-parameters  ================================= #
parser = argparse.ArgumentParser( description = 'Hyper-parameters for network' )
parser.add_argument( '--learning_rate', help = 'Set the learning rate', default = 1e-4, type = float )
parser.add_argument( '--weight_decay', help = 'weight decay', default = 1e-4, type = float )

parser.add_argument( '--crop_size', help = 'Set the crop_size', default = [ 256, 256 ], nargs = '+', type = int )
parser.add_argument( '--train_batch_size', help = 'Set the training batch size', default = 64, type = int )
parser.add_argument( '--epoch_start', help = 'Starting epoch number of the training', default = 0, type = int )

parser.add_argument(
    '--alpha_loss', help = 'Set the alpha in loss function for perceptual_loss', default = 0.04, type = float
)
parser.add_argument( '--beta_loss', help = 'Set the beta in loss function for ssim_loss', default = 0.05, type = float )
parser.add_argument(
    '--gamma_loss', help = 'Set the gamma in loss function for identity_loss', default = 0.05, type = float
)

parser.add_argument( '--val_batch_size', help = 'Set the validation/test batch size', default = 64, type = int )
parser.add_argument(
    '--exp_name', help = 'directory for saving the networks of the experiment', type = str
    , default = 'checkpoint'
)
parser.add_argument( '--seed', help = 'set random seed', default = 666, type = int )
parser.add_argument( '--num_epochs', help = 'number of epochs', default = 2, type = int )
parser.add_argument( '--isapex', help = 'Automatic Mixed-Precision', default = 0, type = int )
parser.add_argument( "--pretrained", help = 'whether have a pretrained model', type = int, default = 0 )
parser.add_argument(
    "--isresume", help = 'if you have a pretrained model , you can continue train it ', type = int
    , default = 0
)
parser.add_argument(
    "--time_str", help = 'where the logging file and tensorboard you want continue', type = str
    , default = None
)
parser.add_argument( "--step_size", help = 'step size of step lr scheduler', type = int, default = 5 )
parser.add_argument( "--step_gamma", help = 'gamma of step lr scheduler', type = float, default = 0.99 )

# ================================ Set parameter  ================================= #
args = parser.parse_args()
learning_rate = args.learning_rate
weight_decay = args.weight_decay
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
alpha_loss = args.alpha_loss
beta_loss = args.beta_loss
gamma_loss = args.gamma_loss

val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
pretrained = args.pretrained
isresume = args.isresume
time_str = args.time_str
isapex = args.isapex
step_size = args.step_size
step_gamma = args.step_gamma


# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
#                        max_lr = 5e-4, # Upper learning rate boundaries in the cycle for each parameter group
#                        div_factor = 5,
#                        final_div_factor = 1e2,
#                        steps_per_epoch = len(train_data_loader.dataset) // train_batch_size, # The number of steps per epoch to train for.
#                        epochs = num_epochs, # The number of epochs to train for.
#                        anneal_strategy = 'cos') # Specifies the annealing strategy


# ================================ Set seed  ================================= #
seed = args.seed
if seed is not None:
    np.random.seed( seed )
    torch.manual_seed( seed )
    torch.cuda.manual_seed( seed )
    random.seed( seed )
    print( 'Seed:\t{}'.format( seed ) )

# =============  Load training data and validation/test data  ============ #

train_data_dir = './data/train/'
val_data_dir = './data/test/'
### The following file should be placed inside the directory "./data/train/"
labeled_name = 'train.txt'
### The following files should be placed inside the directory "./data/test/"
# val_filename = 'val_list_rain800.txt'
# val_filename1 = 'raindroptesta.txt'
val_filename = 'test.txt'

train_data_loader = DataLoader(
    TrainData( crop_size, train_data_dir, labeled_name ), batch_size = train_batch_size,
    shuffle = True
)
val_data_loader = DataLoader(
    ValData( crop_size, val_data_dir, val_filename ), batch_size = val_batch_size, shuffle = False,
    num_workers = 0
)

# ================== Define the model nad  loss network  ===================== #
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
net = RecWeather().to( device ) 
# net = torch.compile( net )  # for torch 2.0

# vgg_model = vgg16(pretrained=True).features[:16]
# # download model to  C:\Users\CHENHUI/.cache\torch\hub\checkpoints\vgg16-397923af.pth
# # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
# for param in vgg_model.parameters():
#     param.requires_grad = False
# loss_network = LossNetwork(vgg_model).to(device)
# loss_network.eval()

# perceptnet = convnext_tiny(weights=torchvision.models.convnext.ConvNeXt_Tiny_Weights.DEFAULT).features
# perceptnet = vgg16(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1).features[:16]
perceptnet = vgg16(pretrained = True).features[:16]
# download model to  C:\Users\CHENHUI/.cache\torch\hub\checkpoints\vgg16-397923af.pth
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in perceptnet.parameters():
    param.requires_grad = False
loss_network = LossNetwork( perceptnet ).to( device )
loss_network.eval()

# ==========================  Build optimizer  ========================= #
optimizer = torch.optim.AdamW( net.parameters(), lr = learning_rate )

# ================== Build learning rate scheduler  ===================== #
optimizer.param_groups[0]["learning_rate"] = learning_rate 
optimizer.param_groups[0]["weight_decay"] = weight_decay

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                       max_lr = 5e-4, # Upper learning rate boundaries in the cycle for each parameter group
                       div_factor = 5,
                       final_div_factor = 1e2,
                       steps_per_epoch = len(train_data_loader.dataset) // train_batch_size, # The number of steps per epoch to train for.
                       epochs = num_epochs, # The number of epochs to train for.
                       anneal_strategy = 'cos') # Specifies the annealing strategy

# ================== Previous PSNR and SSIM in testing  ===================== #
psnr = PSNR()
ssim = SSIM()

# ================  Amp, short for Automatic Mixed-Precision ================
if isapex:
    use_amp = True
    print( f" Let's using  Automatic Mixed-Precision to speed traing !" )
    scaler = torch.cuda.amp.GradScaler( enabled = use_amp )


# ================== Resume training from checkpoint  ===================== #
if pretrained:
    
    assert time_str is not None, 'Must specify a model timestamp'
    try:
        print( '--- Loading model weight... ---' )
        # original saved file with DataParallel
        best_state_dict = torch.load( './{}/{}/best_model.pth'.format( exp_name ,time_str), map_location = device )
        # state_dict = {
        #     "net": net.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     "epoch": epoch,
        #     'scheduler': scheduler.state_dict()
        # }
        net.load_state_dict( best_state_dict[ 'net' ] )
        print( '--- Loading model successfully! ---' )
        pytorch_total_params = sum( p.numel() for p in net.parameters() if p.requires_grad )
        print( "Total_params: {}".format( pytorch_total_params ) )
        old_val_loss, old_val_psnr, old_val_ssim = validation_gpu(
            net, val_data_loader, device = device,
            perceptual_loss_network=loss_network,
            ssim=ssim, psnr=psnr, alpha=alpha_loss, beta=beta_loss, gamma=gamma_loss
        )
        print( ' old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format( old_val_psnr, old_val_ssim ) )
        del best_state_dict

        if isresume:
            # 接着最后一轮开始训练
            last_state_dict = torch.load( './{}/{}/latest_model.pth'.format( exp_name ,time_str) )
            net.load_state_dict( last_state_dict[ 'net' ] )
            optimizer.load_state_dict( last_state_dict[ 'optimizer' ] )
            if isapex:
                scaler.load_state_dict( last_state_dict[ 'amp_scaler' ] )
            epoch_start = last_state_dict[ 'epoch' ]  # Do not need + 1
            step_start = last_state_dict[ 'step' ]
            scheduler.load_state_dict( last_state_dict[ 'scheduler' ] )
            print( f" Let's continue training the model from epoch {epoch_start} !" )

            step_logger = Logger( timestamp = time_str, filename = f'train-step.txt' ).initlog()
            epoch_logger = Logger( timestamp = time_str, filename = f'train-epoch.txt' ).initlog()
            val_logger = Logger( timestamp = time_str, filename = f'val-epoch.txt' ).initlog()
            writer = SummaryWriter( f'logs/tensorboard/{time_str}' )  # tensorboard writer
        else:
            # 否则就是 有 pretrain 的 model，只不过用这个best model上重新训练
            # 就需要新的logging
            curr_time = datetime.datetime.now()
            time_str = datetime.datetime.strftime( curr_time, r'%Y_%m_%d_%H_%M_%S' )
            step_logger = Logger( timestamp = time_str, filename = f'train-step.txt' ).initlog()
            epoch_logger = Logger( timestamp = time_str, filename = f'train-epoch.txt' ).initlog()
            val_logger = Logger( timestamp = time_str, filename = f'val-epoch.txt' ).initlog()

            writer = SummaryWriter( f'logs/tensorboard/{time_str}' )  # tensorboard writer
        del last_state_dict

        torch.cuda.empty_cache()

    except:
        raise FileNotFoundError

else:  # 如果没有pretrained的model，那么就新建logging
    old_val_psnr, old_val_ssim = 0.0, 0.0
    print( '- ' * 50 )
    print(
        'Do not continue training an already pretrained model , '
        'if you need , please specify the parameter ** pretrained | isresume | time_str ** .\n'
        'Now will be train the model from scratch ! '
    )

    # -----Logging------
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime( curr_time, '%Y_%m_%d_%H_%M_%S' )
    os.makedirs( './{}/{}/'.format( exp_name ,time_str) ,exist_ok=True )
    step_logger = Logger( timestamp = time_str, filename = f'train-step.txt' ).initlog()
    epoch_logger = Logger( timestamp = time_str, filename = f'train-epoch.txt' ).initlog()
    val_logger = Logger( timestamp = time_str, filename = f'val-epoch.txt' ).initlog()
    writer = SummaryWriter( f'logs/tensorboard/{time_str}' )  # tensorboard writer
    # -------------------
    step_start = 0
    
    
# =============  Gpu device and nn.DataParallel  ============ #
if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        device_ids = [ Id for Id in range( torch.cuda.device_count() ) ]
        net = nn.DataParallel( net, device_ids = device_ids )
        loss_network = nn.DataParallel( loss_network, device_ids = device_ids )
        print( '-' * 50 )
        print( f'Train model on {torch.cuda.device_count()} GPU with multi threads !' )
    else:
        print( '-' * 50 )
        print( f'Train model on {torch.cuda.device_count()} GPU !' )

# ================================  Set parameters and save them and Synchronize all processes =============================== #

step = 0
if step_start: step = step + step_start
lendata = len( train_data_loader )
num_epochs = num_epochs + epoch_start
pytorch_total_params = sum( p.numel() for p in net.parameters() if p.requires_grad )
parameter_logger = Logger( timestamp = time_str, filename = f'parameters.txt', mode = 'w+' ).initlog()
print( '--- Hyper-parameters for training...' )
parameter = '--- seed: {}\n' \
            '--- learning_rate: {}\n' \
            '--- total_epochs: {}\n' \
            '--- total_params: {}\n' \
            '--- crop_size: {}\n' \
            '--- train_batch_size: {}\n' \
            '--- val_batch_size: {}\n' \
            '--- alpha: {}\n' \
            '--- beta: {}\n' \
            '--- gamma: {}\n' \
            '--- lrscheduler_step_size: {}\n' \
            '--- lrscheduler_step_gamma: {}\n'.format(
    seed, learning_rate, num_epochs, pytorch_total_params,
    crop_size, train_batch_size, val_batch_size,
    alpha_loss,beta_loss,gamma_loss, step_size, step_gamma
)
print( parameter )
parameter_logger.writelines( parameter )
parameter_logger.close()
print( '=' * 25, ' Begin training model ! ', '=' * 25, )


for epoch in range( epoch_start, num_epochs ):  # default epoch_start = 0
    start_time = time.time()
    epoch_loss = 0
    epoch_psnr = 0
    epoch_ssim = 0
    # adjust_learning_rate(optimizer, epoch)
    loop = tqdm( train_data_loader, desc = "Progress bar : " )
    # -------------------------------------------------------------------------------------------------------------
    for batch_id, train_data in enumerate( loop ):

        input_image, gt, imgid = train_data
        input_image = input_image.to( device )
        # print(input_image.shape)
        gt = gt.to( device )
        # --- Zero the parameter gradients --- #
        optimizer.zero_grad( set_to_none = True )  # set_to_none = True here can modestly improve performance

        # --- Forward + Backward + Optimize --- #
        if isapex:
            with torch.autocast( device_type = 'cuda', dtype = torch.float16, enabled = use_amp ):
                net.to( device ).train()
                pred_image = net( input_image )
                gt_pred = net( gt )

                gt_pred.to( device )
                pred_image.to( device )

                loss = synthetic_loss(
                    pred_image = pred_image,
                    gt = gt,
                    gt_pred = gt_pred,
                    plnet = loss_network, ssim = ssim, alpha = alpha_loss, beta = beta_loss, gamma = gamma_loss

                )
            scaler.scale( loss ).backward()
            scaler.step( optimizer )
            scheduler.step()  # Adjust learning rate for every batch
            scaler.update()
        else:
            net.to( device ).train()
            pred_image = net( input_image )
            gt_pred  = net( gt )

            gt_pred.to( device )
            pred_image.to( device )

            loss = synthetic_loss(
                pred_image = pred_image,
                gt = gt,
                gt_pred = gt_pred,
                plnet = loss_network, ssim = ssim, alpha = alpha_loss, beta = beta_loss, gamma = gamma_loss

            )
            loss.backward()
            optimizer.step()
            scheduler.step()  # Adjust learning rate for every batch
            

        step_psnr, step_ssim = \
            psnr.to_psnr( pred_image.detach(), gt.detach() ), ssim.to_ssim( pred_image.detach(), gt.detach() )

        loop.set_postfix(
            { 'Epoch': f'{epoch + 1} / {num_epochs}', 'Step': f'{step + 1}',
            'Steploss': '{:.4f}'.format( loss.item() )
            }
        )

        writer.add_scalar( 'TrainingStep/step-loss', loss.item(), step + 1 )
        writer.add_scalar( 'TrainingStep/step-PSNR', step_psnr, step + 1 )
        writer.add_scalar( 'TrainingStep/step-SSIM', step_ssim, step + 1 )
        writer.add_scalar(
            'TrainingStep/lr', scheduler.get_last_lr()[ 0 ], step + 1
        )  # logging lr for every step
        step_logger.writelines(
            f'Epoch: {epoch + 1} / {num_epochs} - Step: {step + 1}'
            + ' - steploss: {:.4f} - stepPSNR: {:.4f} - stepSSIM: {:.4f}\n'.format(
                loss.item(), step_psnr, step_ssim
            )
        )
        if step % 50 == 0: step_logger.flush()
        epoch_loss += loss.item()
        epoch_psnr += step_psnr
        epoch_ssim += step_ssim
        step = step + 1

    epoch_loss /= lendata
    epoch_psnr /= lendata
    epoch_ssim /= lendata

    print(
        '----Epoch: [{}/{}], EpochAveLoss: {:.4f}, EpochAvePSNR: {:.4f}, EpochAveSSIM: {:.4f}----'
        .format( epoch + 1, num_epochs, epoch_loss, epoch_psnr, epoch_ssim )
    )
    writer.add_scalar( 'TrainingEpoch/epoch-loss', epoch_loss, epoch + 1 )
    writer.add_scalar( 'TrainingEpoch/epoch-PSNR', epoch_psnr, epoch + 1 )
    writer.add_scalar( 'TrainingEpoch/epoch-SSIM', epoch_ssim, epoch + 1 )

    epoch_logger.writelines(
        'Epoch [{}/{}], EpochAveLoss: {:.4f}, EpochAvePSNR: {:.4f} EpochAveSSIM: {:.4f}\n'.format(
            epoch + 1, num_epochs, epoch_loss, epoch_psnr, epoch_ssim
        )
    )

    epoch_logger.flush()

    if ((epoch + 1) % 5 == 0) or (epoch == num_epochs - 1):
        # --- Save the  parameters --- #
        model_to_save = net.module if hasattr( net, "module" ) else net
        ## Take care of distributed/parallel training
        '''
        If you have an error about load model in " Missing key(s) in state_dict: " , 
        maybe you can  reference this url :
        https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/7
        '''
        checkpoint = {
            "net": model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch + 1,
            'step': step + 1,
            'scheduler': scheduler.state_dict(),
            'amp_scaler': scaler.state_dict() if isapex else None
        }
        torch.save( checkpoint, './{}/{}/latest_model.pth'.format( exp_name ,time_str) )

        # --- Use the evaluation model in testing --- #
        val_loss, val_psnr, val_ssim = validation_gpu(
            net, val_data_loader, device = device,
            perceptual_loss_network = loss_network,
            ssim = ssim, psnr = psnr, alpha = alpha_loss, beta = beta_loss, gamma = gamma_loss
        )

        print( '--- ValLoss : {:.4f} , Valpsnr : {:.4f} , Valssim : {:.4f}'.format( val_loss, val_psnr, val_ssim ) )
        writer.add_scalar( 'Validation/loss', val_loss, epoch + 1 )
        writer.add_scalar( 'Validation/PSNR', val_psnr, epoch + 1 )
        writer.add_scalar( 'Validation/SSIM', val_ssim, epoch + 1 )
        #  logging
        val_logger.writelines(
            'Epoch [{}/{}], ValEpochAveLoss: {:.4f}, ValEpochAvePSNR: {:.4f} ValEpochAveSSIM: {:.4f}\n'.format(
                epoch + 1, num_epochs, val_loss, val_psnr, val_ssim
            )
        )
        # val_psnr1, val_ssim1 = validation(net, val_data_loader1, device, exp_name)
        # val_psnr2, val_ssim2 = validation(net, val_data_loader2, device, exp_name)

        one_epoch_time = time.time() - start_time
        # print("Rain 800")
        # print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, exp_name)
        # print("Rain Drop")
        # print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr1, val_ssim1, exp_name)
        # print("Test1")
        # print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr2, val_ssim2, exp_name)

        # --- update the network weight --- #

        if val_psnr >= old_val_psnr:
            torch.save( checkpoint, './{}/{}/best_model.pth'.format( exp_name ,time_str) )

            print( 'Update the best model !' )
            old_val_psnr = val_psnr

        # Note that we find the best model based on validating with raindrop data.

step_logger.close()
epoch_logger.close()
val_logger.close()
writer.close()
print( 'Training Program Is Finished !' )
# 打印按 GPU 时间排序的性能分析结果