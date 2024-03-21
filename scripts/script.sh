# download dataset
pip3 install gdown
gdown --id 1v1z7NRyF9wD6wAlZBbphBZgTuIs8zOas
unzip -o -d ./ Allweather_subset.zip

#========================================================================================================
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9 
tensorboard --port 6007 --logdir /root/Projects/RocWeather/logs/tensorboard 
# 注意需要根目录
#========================================================================================================
# train net  with nn.DataParallel
python3 train_gpu.py --train_batch_size=1 --val_batch_size=1 --num_epochs=50
# visualization model and metrics
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9 
tensorboard --port 6007 --logdir /root/Projects/RocWeather/logs/tensorboard 
# 注意需要根目录

# resume model
python3 train_gpu.py --train_batch_size 8 --val_batch_size 8 --num_epochs 2 --pretrained 1 --isresume 1 --time_str 2023_11_17_14_13_18

#========================================================================================================

#========================================================================================================
# train net  with nn.DataParallel


python3 -m torch.distributed.launch --nproc_per_node 4 train_ddp.py --train_batch_size 8 --val_batch_size 8 --num_epochs 50


python3 -m torch.distributed.launch --nproc_per_node 4 train_ddp.py --pretrained 1 --isresume 1 --num_epochs 10 --time_str 2023_12_16_17_44_12 --train_batch_size 8 --val_batch_size 8  



'''

import os

def rename_images(path):
    for filename in os.listdir(path):
        old_path = os.path.join(path, filename)
        new_filename = filename.replace("rain", "clean")
        new_path = os.path.join(path, new_filename)
        
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')

# 指定要修改的路径
directory_path = "data/test/input"

# 调用函数进行重命名
rename_images(directory_path)

'''

