import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


def RGBframes_np2Tensor(imgIn, channel):
    ## input : T, H, W, C
    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0, axis=3,
                       keepdims=True) + 16.0

    # to Tensor
    ts = (3, 0, 1, 2)  ############# dimension order should be [C, T, H, W]
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)

    return imgIn


def frames_loader_train(args, candidate_frames, frameRange):
    frames = []
    for frameIndex in frameRange:
        frame = cv2.imread(candidate_frames[frameIndex])
        frames.append(frame)
    (ih, iw, c) = frame.shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    if args.need_patch:  ## random crop
        ps = args.patch_size
        ix = random.randrange(0, iw - ps + 1)
        iy = random.randrange(0, ih - ps + 1)
        frames = frames[:, iy:iy + ps, ix:ix + ps, :]

    if random.random() < 0.5:  # random horizontal flip
        frames = frames[:, :, ::-1, :]

    # No vertical flip

    rot = random.randint(0, 3)  # random rotate
    frames = np.rot90(frames, rot, (1, 2))

    """ w/o [-1,1] normalize """
    frames = RGBframes_np2Tensor(frames, args.img_ch)

    return frames


def frames_loader_test(args, I0I1It_Path, validation):
    frames = []
    for path in I0I1It_Path:
        frame = cv2.imread(path)
        frames.append(frame)
    (ih, iw, c) = frame.shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)

    if validation:
        ps = 512
        ix = (iw - ps) // 2
        iy = (ih - ps) // 2
        frames = frames[:, iy:iy + ps, ix:ix + ps, :]

    """ w/o [-1,1] normalize """
    frames = RGBframes_np2Tensor(frames, args.img_ch)

    return frames


def make_2D_dataset_X_Train(dir):
    framesPath = []
    # Find and loop over all the clips in root `dir`.
    for scene_path in sorted(glob.glob(os.path.join(dir, '*', ''))):
        sample_paths = sorted(glob.glob(os.path.join(scene_path, '*', '')))
        for sample_path in sample_paths:
            frame65_list = []
            for frame in sorted(glob.glob(os.path.join(sample_path, '*.png'))):
                frame65_list.append(frame)
            framesPath.append(frame65_list)

    print("The number of total training samples : {} which has 65 frames each.".format(
        len(framesPath)))  ## 4408 folders which have 65 frames each
    return framesPath


def make_2D_dataset_X_Test(dir, multiple, t_step_size):
    """ make [I0,I1,It,t,scene_folder] """
    """ 1D (accumulated) """
    testPath = []
    t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
    for type_folder in sorted(glob.glob(os.path.join(dir, '*', ''))):  # [type1,type2,type3,...]
        for scene_folder in sorted(glob.glob(os.path.join(type_folder, '*', ''))):  # [scene1,scene2,..]
            frame_folder = sorted(glob.glob(scene_folder + '*.png'))  # 32 multiple, ['00000.png',...,'00032.png']
            for idx in range(0, len(frame_folder), t_step_size):  # 0,32,64,...
                if idx == len(frame_folder) - 1:
                    break
                for mul in range(multiple - 1):
                    I0I1It_paths = []
                    I0I1It_paths.append(frame_folder[idx])  # I0 (fix)
                    I0I1It_paths.append(frame_folder[idx + t_step_size])  # I1 (fix)
                    I0I1It_paths.append(frame_folder[idx + int((t_step_size // multiple) * (mul + 1))])  # It
                    I0I1It_paths.append(t[mul])
                    I0I1It_paths.append(scene_folder.split(os.path.join(dir, ''))[-1])  # type1/scene1
                    testPath.append(I0I1It_paths)
    return testPath


class X_Train(Dataset):
    def __init__(self, args, max_t_step_size):
        self.args = args
        self.max_t_step_size = max_t_step_size

        self.framesPath = make_2D_dataset_X_Train(self.args.train_data_path)
        self.nScenes = len(self.framesPath)

        # Raise error if no images found in train_data_path.
        if self.nScenes == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.args.train_data_path + "\n"))

    def __getitem__(self, idx):
        t_step_size = random.randint(2, self.max_t_step_size)
        t_list = np.linspace((1 / t_step_size), (1 - (1 / t_step_size)), (t_step_size - 1))


        candidate_frames = self.framesPath[idx]

        # random temporal gap for fixed timestep interpolation (t=0.5)
        firstFrameIdx = random.randint(0, 62)
        max_len = 64 - firstFrameIdx
        interIdx = random.randint(1, int(max_len // 2))
        interFrameIdx = firstFrameIdx + interIdx
        t_value = 0.5

        if (random.randint(0, 1)):
            frameRange = [firstFrameIdx, firstFrameIdx + interIdx*2, interFrameIdx]
        else:  ## temporally reversed order
            frameRange = [firstFrameIdx + interIdx*2, firstFrameIdx, interFrameIdx]
            t_value = 1.0 - t_value
        assert frameRange[0] <= 64 and frameRange[1] <= 64 and frameRange[2] <= 64, f'frameRange: {frameRange}, firstFrameIdx: {firstFrameIdx}, interIdx: {interIdx}, interFrameIdx: {interFrameIdx}'
        frames = frames_loader_train(self.args, candidate_frames,
                                     frameRange)

        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0)

    def __len__(self):
        return self.nScenes


class X_Test(Dataset):
    def __init__(self, args, multiple, validation):
        self.args = args
        self.multiple = multiple
        self.validation = validation
        if validation:
            self.testPath = make_2D_dataset_X_Test(self.args.val_data_path, multiple, t_step_size=32)
        else:
            self.testPath = make_2D_dataset_X_Test(self.args.test_data_path, multiple, t_step_size=32)

        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            if validation:
                raise (RuntimeError("Found 0 files in subfolders of: " + self.args.val_data_path + "\n"))
            else:
                raise (RuntimeError("Found 0 files in subfolders of: " + self.args.test_data_path + "\n"))

    def __getitem__(self, idx):
        I0, I1, It, t_value, scene_name = self.testPath[idx]

        I0I1It_Path = [I0, I1, It]

        frames = frames_loader_test(self.args, I0I1It_Path, self.validation)

        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]

        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.nIterations


def get_train_data(args, max_t_step_size=None, local_rank=-1):
    data_train = X_Train(args, max_t_step_size)
    sampler = DistributedSampler(data_train) if local_rank != -1 else None
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, drop_last=True,
                                             num_workers=int(args.num_thrds), pin_memory=True, sampler=sampler)
    return dataloader, sampler


def get_test_data(args, multiple, validation):
    data_test = X_Test(args, 2, validation)  # 'validation' for validation while training for simplicity
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1, drop_last=True, pin_memory=True)  # 4K
    return dataloader
