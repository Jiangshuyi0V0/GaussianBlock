from copy import deepcopy
from functools import cached_property
from PIL import Image

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import ToTensor, Compose, Resize

from utils import path_exists, get_files_from
from utils.image import IMG_EXTENSIONS
from utils.path import DATASETS_PATH


class BMVSDataset(TorchDataset):
    name = 'bmvs'
    raw_img_size = (576, 768)
    n_channels = 3

    def __init__(self, split, img_size, tag, **kwargs):
        kwargs = deepcopy(kwargs)
        self.split = split
        if kwargs.pop('if_bin_render', False):
            self.data_path = path_exists(DATASETS_PATH / 'BlendedMVS' / tag / 'bin_image')
        else:
            self.data_path = path_exists(DATASETS_PATH / 'BlendedMVS' / tag / 'image')
        self.input_files = get_files_from(self.data_path, IMG_EXTENSIONS, recursive=True, sort=True)
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        N = len(self.input_files)
        self.view_ids = kwargs.pop('view_ids', list(range(N)))
        self.on_disk = kwargs.pop('on_disk', False)

        cam = np.load(self.data_path.parent / 'cameras.npz')
        proj_mat = [(cam[f'world_mat_{i}'] @ cam[f'scale_mat_{i}'])[:3, :4] for i in range(N)]
        self.KRT = [pytorch3d_KRT_from_proj(p, image_size=self.raw_img_size) for p in proj_mat]

        self.pc_gt = torch.zeros(1, 3)

        if self.on_disk:
            self.imgs = [self.transform(Image.open(f).convert('RGB')) for f in self.input_files]

    def __len__(self):
        if self.split == 'train':
            return len(self.view_ids)
        elif self.split == 'val':
            return min(5, len(self.view_ids))
        else:
            return min(10, len(self.view_ids))

    def __getitem__(self, i):
        idx = self.view_ids[i]
        if self.on_disk:
            imgs = self.imgs[idx]
        else:
            imgs = self.transform(Image.open(self.input_files[idx]).convert('RGB'))
        K, R, T = self.KRT[idx]
        out = {'imgs': imgs, 'K': K, 'R': R, 'T': T}
        indices = torch.randperm(len(self.pc_gt))[:int(1e5)]
        pc = self.pc_gt[indices]
        return out, {'points': pc}

    @cached_property
    def transform(self):
        return Compose([Resize(self.img_size), ToTensor()])


def pytorch3d_KRT_from_proj(P, image_size):
    K, R, T = map(torch.from_numpy, opencv_KRT_from_proj(P))

    R = R.T
    T = - R @ T

    # Retype the image_size correctly and flip to width, height.
    if isinstance(image_size, (tuple, list)):
        image_size = torch.Tensor(image_size)[None]
    image_size_wh = image_size.to(R).flip(dims=(1,))

    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = torch.stack([K[0, 0], K[1, 1]], dim=-1) / scale
    p0_pytorch3d = -(K[:2, 2] - c0) / scale
    K_pytorch3d = torch.zeros(K.shape)
    K_pytorch3d[0, 0] = focal_pytorch3d[..., 0]
    K_pytorch3d[1, 1] = focal_pytorch3d[..., 1]
    K_pytorch3d[:2, 2] = p0_pytorch3d
    K_pytorch3d[2:, 2:] = 1 - torch.eye(2)

    R_pytorch3d = R.clone().T
    T_pytorch3d = T.clone()
    R_pytorch3d[:, :2] *= -1
    T_pytorch3d[:2] *= -1
    return K_pytorch3d, R_pytorch3d, T_pytorch3d


def opencv_KRT_from_proj(P):
    K_raw, R, T = cv2.decomposeProjectionMatrix(P)[:3]
    K = np.eye(4, dtype=np.float32)
    K[:3, :3] = K_raw / K_raw[2, 2]
    R = R.T
    T = (T[:3] / T[3])[:, 0]
    return K, R, T
