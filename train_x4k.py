import os
import argparse
import random
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import math
import json
import time

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from dataset import VimeoDataset
from X4K_dataset import get_train_data, get_test_data
from config import *
from Trainer_x4k import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 100

def get_learning_rate(step):
    warmup = 1000
    if step < warmup:
        mul = step / warmup
        return 2e-4 * mul
    else:
        mul = np.cos((step - warmup) / (epochs * args.step_per_epoch - warmup) * math.pi) * 0.5 + 0.5
        return (2e-4 - 2e-5) * mul + 2e-5


def random_rescale(img0, img1, gt):
    rand = random.uniform(0, 1)
    if rand < 0.5:
        scale_factor = 1
    elif 0.5 <= rand < 0.75:
        scale_factor = 0.5
    else:
        scale_factor = 0.25
    img0 = F.interpolate(img0, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    img1 = F.interpolate(img1, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    gt = F.interpolate(gt, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    return img0, img1, gt


def train(model, local_rank):
    if local_rank == 0:
        writer = SummaryWriter(f'log/{MODEL_CONFIG["LOGNAME"]}/train/vis')
    train_data, sampler = get_train_data(args, 32, local_rank)
    args.step_per_epoch = train_data.__len__()
    val_data = get_test_data(args, 2, True)
    print('training...')
    start_epoch, nr_eval, step = 0, 0, 0
    time_stamp = time.time()
    cur_psnr = evaluate(model, val_data, nr_eval, local_rank)
    if local_rank <= 0:
        print(f'initial psnr: {cur_psnr}')
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch) if local_rank > 0 else None
        for i, (imgs, timestep) in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            imgs = imgs.to(device, non_blocking=True) / 255.
            timestep = timestep.view(-1, 1, 1, 1)
            timestep = timestep.to(device, non_blocking=True)
            img0, img1, gt = imgs[:, :, 0], imgs[:, :, 1], imgs[:, :, 2]
            img0, img1, gt = random_rescale(img0, img1, gt)
            imgs = torch.cat((img0, img1), 1)
            learning_rate = get_learning_rate(step)
            _, loss = model.update(imgs, gt, learning_rate, timestep, training=True)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss', loss, step)
            if local_rank <= 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, loss))
            step += 1
        nr_eval += 1
        if nr_eval % 2 == 0:
            cur_psnr = evaluate(model, val_data, nr_eval, local_rank)
        model.save_model(local_rank, epoch)
        dist.barrier()


def evaluate(model, val_data, nr_eval, local_rank):
    if local_rank == 0:
        writer_val = SummaryWriter(f'log/{MODEL_CONFIG["LOGNAME"]}/val/vis')
    psnr = []
    for _, (imgs, timestep, _, _) in enumerate(val_data):
        imgs = imgs.to(device, non_blocking=True) / 255.
        timestep = timestep.to(device, non_blocking=True)
        timestep = timestep.view(-1, 1, 1, 1)
        img0, img1, gt = imgs[:, :, 0], imgs[:, :, 1], imgs[:, :, 2]
        imgs = torch.cat((img0, img1), 1)
        with torch.no_grad():
            pred, _ = model.update(imgs, gt, timestep=timestep, training=False)
        for j in range(gt.shape[0]):
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
    psnr = np.array(psnr).mean()
    if local_rank == 0:
        print(str(nr_eval), psnr)
        writer_val.add_scalar('psnr', psnr, nr_eval)
        log_stats = {"epoch": nr_eval, "psnr": psnr}
        with open(f"./log/{MODEL_CONFIG['LOGNAME']}/log.txt", "a") as f:
            f.write(json.dumps(log_stats) + "\n")
    return psnr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--num_thrds', default=8, type=int, help='dataloader num threads')
    parser.add_argument('--img_ch', default=3, type=int, help='image channels')
    parser.add_argument("--need_patch", action="store_true", default=False, help="if need patch")
    parser.add_argument('--patch_size', default=512, type=int, help='if need patch, patch size')
    parser.add_argument('--train_data_path', type=str, help='data path of X4K')
    parser.add_argument('--val_data_path', type=str, help='data path of X4K')
    parser.add_argument("--wandb_log", action="store_true", default=False, help="use wandb to log")
    parser.add_argument("--ckpt_name", default=None, type=str, help="ckpt path")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else -1

    if local_rank != -1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(local_rank)  # NOTE: `Model` is not an nn.Module()
    if local_rank <= 0:
        n_parameters = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
        print(f'Number of parameters: {n_parameters}')
    train(model, local_rank)
