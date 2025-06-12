import numpy as np
import torch
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import random
from collections import defaultdict
import glob
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib 
import os
import matplotlib.pyplot as plt
from models import Ray2Ray
import torchmetrics
from utils import save_model, load_model, cartesian_to_spherical, spherical_to_cartesian, \
normalize_objectdistance, unnormalize_output, normalize_points, load_scalers, pixelization


import numpy as np
import torch
import os

def load_data(data_dir, device=None):

    train_folder = os.path.join(data_dir, 'train')
    test_folder = os.path.join(data_dir, 'test')

    def _load_data(folder_path):
        sources, gaussian_pixelized, outs, man_psfs = None, None, None, None
        

        sources_path = os.path.join(folder_path, 'sources.npy')
        sources = torch.from_numpy(np.load(sources_path)).to(device)
        gaussian_pixelized_path = os.path.join(folder_path, 'gaussian_pixelized.npy')
        gaussian_pixelized = torch.from_numpy(np.load(gaussian_pixelized_path)).to(device)

        outs_path = os.path.join(folder_path, 'outs.npy')
        outs = torch.from_numpy(np.load(outs_path)).to(device)

        man_psfs_path = os.path.join(folder_path, 'man_psfs.npy')
        man_psfs = torch.from_numpy(np.load(man_psfs_path)).to(device)


        return sources, gaussian_pixelized, outs, man_psfs

    train_data = _load_data(train_folder)
    test_data = _load_data(test_folder)
    return train_data, test_data, train_data[0].shape[1]


def create_dataloaders(source_train, outpix_train, source_test, outpix_test, train_ratio=0.8, batch_size=1, shuffle=True, test = False):
    if not test:
        train_dataset = TensorDataset(source_train, outpix_train)
        gen = torch.Generator(device=source_train.device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, generator=gen)
    else:
        train_loader=None
    test_dataset = TensorDataset(source_test, outpix_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
