import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Callable

import re
import random
import nibabel as nib
import torchio as tio



def convert_to_attenuation(mu_water_80keV: float = 0.0193,
                           min_hu: float = -1000,
                           max_hu: float = 2000) -> torch.Tensor:
    def transform(x: torch.Tensor) -> torch.Tensor:
        x = torch.clip(x, min_hu , max_hu)    
        mu = mu_water_80keV * (x / 1000 + 1)
        return mu
    return transform


def add_noise_sino(b: torch.Tensor,
                   I: float = 1e5):
        yi = torch.poisson(I * torch.exp( -b ) )
        b_tilde = torch.log(I / yi )
        b_tilde[ b_tilde <= 0] = 0
        b_tilde[ b_tilde==float('inf') ] = 0

        return b_tilde, yi
    

def convert_m1_p1_to_attenuation(mu_water_80keV: float = 0.0193,
                                  min_hu: float = -1000,
                                  max_hu: float = 2000):
    
    def inverse_transform(x: torch.Tensor) -> torch.Tensor:
        mu_min = mu_water_80keV * (min_hu / 1000 + 1)
        mu_max = mu_water_80keV * (max_hu / 1000 + 1)
        x = (x + 1) * (mu_max - mu_min) / 2 + mu_min
        return x

    return inverse_transform

def get_minus_one_one_norm_hu_transform(min_hu: float = -1000,
                                        max_hu: float = 2000):
    def transform(x: torch.Tensor) -> torch.Tensor:
        x = torch.clip(x, min_hu, max_hu)
        x = 2 * ((x - min_hu) / (max_hu - min_hu)) - 1
        return x
    return transform

# def get_minus_one_one_norm_hu_transform(min_hu: float = -1000,
#                                         max_hu: float = 2000):
#     def transform(x: torch.Tensor) -> torch.Tensor:
#         x = torch.clip(x, min_hu, max_hu)
#         x = 2 * ((x - min_hu) / (max_hu - min_hu)) - 1
#         return x
#     return transform


def get_minus_one_one_norm_hu_transform(min_hu: float = -1000,
                                        max_hu: float = 2000):
    def transform(x: torch.Tensor) -> torch.Tensor:
        x = torch.clip(x, min_hu, max_hu)
        x = 2 * ((x - min_hu) / (max_hu - min_hu)) - 1
        return x
    return transform


transform_train_head = tio.Compose([
    get_minus_one_one_norm_hu_transform(min_hu=-1000, max_hu=2000)
    ,
    tio.RandomAffine(
        scales=(0.9, 1.1),
        degrees=(15, 15, 15),
        translation=(10, 10, 10),
        p=0.75  # probability to apply this transform
    ),
    
])

class CTDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 transform : Callable = None):

        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = self._gather_files(root_dir)

    def _gather_files(self, root_dir):
        file_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".npy"):
                    file_paths.append(os.path.join(root, file))
        return sorted(file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """ Loads and returns a sample from the dataset using memory-mapped files. """
        file_path = self.file_paths[idx]
        vol = self._load_file(file_path)
        if self.transform:
            vol = self.transform(vol)
        return vol

    def _load_file(self, file_path: str) -> torch.Tensor:
        data = np.load(file_path, mmap_mode='r').astype(np.float32)
        data = torch.from_numpy(data).unsqueeze(0)
        return data

def get_minus_one_one_norm_att_transform(
        mu_water:float = 0.0193,
        min_hu: float = -1000,
        max_hu: float = 1500):
    
    def transform(x: torch.Tensor) -> torch.Tensor:    
        mu_min = mu_water * (1 + min_hu / 1000)
        mu_max = mu_water * (1 + max_hu / 1000)
        x = torch.clip(x, mu_min, mu_max)
        x = 2 * ((x - mu_min) / (mu_max - mu_min)) - 1
        return x
    return transform


def get_head_ct_dataloaders(root_dir: str,
                            batch_size: int = 1,
                            shuffle: bool = True):
    dataset = CTDataset(root_dir=root_dir, transform=convert_to_attenuation())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader