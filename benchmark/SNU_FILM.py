import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import cv2
import math
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_small', type=str)
parser.add_argument('--path', type=str, default="./data/SNU-FILM", required=True)
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--num_key_points', default=0.5, type=float)
args = parser.parse_args()

'''==========import from our code=========='''
sys.path.append('.')

import config as cfg
# from Trainer_base import Model
from Trainer_x4k import Model
from benchmark.utils.padder import InputPadder
from benchmark.utils.pytorch_msssim import ssim_matlab


'''==========Model setting=========='''
TTA = True
down_scale = 0.8
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=16,
        depth=[2, 2, 2, 4],
        num_key_points=args.num_key_points
    )
    print(f'Testing num_key_points: {args.num_key_points}')
else:
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_base'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=32,
        depth=[2, 2, 2, 6],
        num_key_points=args.num_key_points
    )
    print(f'Testing num_key_points: {args.num_key_points}')
model = Model(-1)
model.load_model(args.exp_name)
model.eval()
model.device()

print(f'=========================Starting testing=========================')
print(f'Dataset: SNU_FILM   Model: {model.name}   TTA: {TTA}')
path = args.path
level_list = ['top-half-motion-sufficiency_test-hard.txt', 'top-half-motion-sufficiency_test-extreme.txt']

for test_file in level_list:
    psnr_list, ssim_list = [], []
    file_list = []

    with open(os.path.join(path, test_file), "r") as f:
        for line in f:
            line = line.strip()
            file_list.append(line.split(' '))

    for line in tqdm(file_list):
        I0_path = os.path.join('./', line[0])
        I1_path = os.path.join('./', line[1])
        I2_path = os.path.join('./', line[2])
        I0 = cv2.imread(I0_path)
        I1_ = cv2.imread(I1_path)
        I2 = cv2.imread(I2_path)
        I0 = (torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
        I1 = (torch.tensor(I1_.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
        I2 = (torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
        padder = InputPadder(I0.shape, divisor=80)
        I0, I2 = padder.pad(I0, I2)
        I1_pred = model.hr_inference(I0, I2, TTA, down_scale=down_scale, fast_TTA=TTA)[0]
        I1_pred = padder.unpad(I1_pred)
        ssim = ssim_matlab(I1, I1_pred.unsqueeze(0)).detach().cpu().numpy()

        I1_pred = I1_pred.detach().cpu().numpy().transpose(1, 2, 0)
        I1_ = I1_ / 255.
        psnr = -10 * math.log10(((I1_ - I1_pred) * (I1_ - I1_pred)).mean())

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print('Testing level:' + test_file[:-4])
    print('Avg PSNR: {} SSIM: {}'.format(np.mean(psnr_list), np.mean(ssim_list)))

    with open(f"./log/{args.exp_name}/log_snu.txt", "a") as f:
        f.write(f'Testing level: {test_file[:-4]}\n')
        f.write(f'Avg PSNR: {np.mean(psnr_list)} SSIM: {np.mean(ssim_list)}\n')
