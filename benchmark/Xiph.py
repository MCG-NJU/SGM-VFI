import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import cv2
import skimage
import torch
import argparse
import warnings
import math
import numpy as np
from tqdm import tqdm
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer_x4k import Model
# import config_base as cfg
# from Trainer_base import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_small', type=str)
parser.add_argument('--path', type=str, default="./data/xiph", required=True)
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--num_key_points', default=0.5, type=float)
args = parser.parse_args()


'''==========Model setting=========='''
TTA = True
down_scale = 0.5
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=16,
        depth=[2, 2, 2, 4],
        num_key_points=args.num_key_points
    )
    print(f'Testing num_key_points: {args.num_key_points}')
else:
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
    )

model = Model(-1)
model.load_model(args.exp_name)
model.eval()
model.device()
file_list = []
with open(f"./xiph/top-half-selected-gap2.txt", "r") as f:
    for line in f:
        line = line.strip()
        file_list.append(line.split(' '))

print(f'=========================Starting testing=========================')
print(f'Dataset: Xiph   Model: {model.name}   TTA: {TTA}')
path = args.path
with torch.no_grad():
    for strCategory in ['resized','cropped']:
        fltPsnr, fltSsim = [], []
        for line in tqdm(file_list):
            npyFirst = cv2.imread(line[0])
            npySecond = cv2.imread(line[2])
            npyReference = cv2.imread(line[1])
            if strCategory == 'resized':
                npyFirst = cv2.resize(src=npyFirst, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                npySecond = cv2.resize(src=npySecond, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                npyReference = cv2.resize(src=npyReference, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

            elif strCategory == 'cropped':
                npyFirst = npyFirst[540:-540, 1024:-1024, :]
                npySecond = npySecond[540:-540, 1024:-1024, :]
                npyReference = npyReference[540:-540, 1024:-1024, :]

            tenFirst = torch.FloatTensor(np.ascontiguousarray(npyFirst.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).unsqueeze(0).cuda()
            tenSecond = torch.FloatTensor(np.ascontiguousarray(npySecond.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).unsqueeze(0).cuda()

            padder = InputPadder(tenFirst.shape)
            tenFirst, tenSecond = padder.pad(tenFirst, tenSecond)

            npyEstimate = padder.unpad(model.hr_inference(tenFirst, tenSecond, TTA=TTA, down_scale=down_scale, fast_TTA=False).clamp(0.0, 1.0).cpu())[0]
            npyEstimate1 = (npyEstimate.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

            psnr = skimage.metrics.peak_signal_noise_ratio(image_true=npyReference, image_test=npyEstimate1, data_range=255)
            ssim = skimage.metrics.structural_similarity(im1=npyReference, im2=npyEstimate1, data_range=255, multichannel=True)

            fltPsnr.append(psnr)
            fltSsim.append(ssim)

        if strCategory == 'resized':
            print('\n---2K---')
            with open(f"./log/{args.exp_name}/log_xiph.txt", "a") as f:
                f.write(f'Xiph-2K:\n')
                f.write(f'Avg PSNR: {np.mean(fltPsnr)} SSIM: {np.mean(fltSsim)}\n')
        else:
            print('\n---4K---')
            with open(f"./log/{args.exp_name}/log_xiph.txt", "a") as f:
                f.write(f'Xiph-4K:\n')
                f.write(f'Avg PSNR: {np.mean(fltPsnr)} SSIM: {np.mean(fltSsim)}\n')
        print('Avg psnr:', np.mean(fltPsnr))
        print('Avg ssim:', np.mean(fltSsim))

