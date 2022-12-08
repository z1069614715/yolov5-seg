import os, shutil, random
import numpy as np

postfix = 'jpg'
base_path = '../dataset/labelme_dataset'
dataset_path = '../dataset/custom_dataset'
val_size, test_size = 0.1, 0.2

os.makedirs(dataset_path, exist_ok=True)
os.makedirs(f'{dataset_path}/images', exist_ok=True)
os.makedirs(f'{dataset_path}/images/train', exist_ok=True)
os.makedirs(f'{dataset_path}/images/val', exist_ok=True)
os.makedirs(f'{dataset_path}/images/test', exist_ok=True)
os.makedirs(f'{dataset_path}/labels/train', exist_ok=True)
os.makedirs(f'{dataset_path}/labels/val', exist_ok=True)
os.makedirs(f'{dataset_path}/labels/test', exist_ok=True)

path_list = np.array([i.split('.')[0] for i in os.listdir(base_path) if 'txt' in i])
random.shuffle(path_list)
train_id = path_list[:int(len(path_list) * (1 - val_size - test_size))]
val_id = path_list[int(len(path_list) * (1 - val_size - test_size)):int(len(path_list) * (1 - test_size))]
test_id = path_list[int(len(path_list) * (1 - test_size)):]

for i in train_id:
    shutil.copy(f'{base_path}/{i}.{postfix}', f'{dataset_path}/images/train/{i}.{postfix}')
    shutil.copy(f'{base_path}/{i}.txt', f'{dataset_path}/labels/train/{i}.txt')

for i in val_id:
    shutil.copy(f'{base_path}/{i}.{postfix}', f'{dataset_path}/images/val/{i}.{postfix}')
    shutil.copy(f'{base_path}/{i}.txt', f'{dataset_path}/labels/val/{i}.txt')

for i in test_id:
    shutil.copy(f'{base_path}/{i}.{postfix}', f'{dataset_path}/images/test/{i}.{postfix}')
    shutil.copy(f'{base_path}/{i}.txt', f'{dataset_path}/labels/test/{i}.txt')