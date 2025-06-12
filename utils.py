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
import os



def save_model(model, optimizer, epoch, loss,scheduler = None, path="model.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler
        'loss': loss
    }, path)
    print(f"Model saved to {path}")

def load_model(model, optimizer=None, scheduler=None, path="model.pth", test=False):
    print(path)
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if test:
        print(f"Model loaded from {path} for testing.")
        return model

    optimizer = None if optimizer == None else optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler =  None if scheduler == None else scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, scheduler, epoch, loss

def cartesian_to_spherical(cartesian_coords):
    r = np.linalg.norm(cartesian_coords, axis=-1)
    theta = np.arccos(cartesian_coords[..., 2] / r)  # cos(theta) = z / r
    phi = np.arctan2(cartesian_coords[..., 1], cartesian_coords[..., 0])  # atan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], axis=-1)


def normalize_objectdistance(objd, min_objd = 0, max_objd = 2500):
    return (objd - min_objd) / (max_objd - min_objd)

#modification is needed. currently its just work for object with size 200. The min_max is not used here.
#The final supporting areas is not decided so Its limited to one size. copy paster from normalized 
#big Waaaaaaaaaaaaaaaaaaaaaaaaarrrrrrrrrrrniiiiiiiiiiiiing
def unnormalize_output(rays):
    origin = rays[...,:2]
    direction = rays[...,2:4]
    screen_size = torch.floor(200 / torch.sqrt(torch.tensor(2)))
    min_max = (-screen_size // 2, screen_size // 2)
    original_xy = ((origin + 1) * (min_max[1] - min_max[0])/2) + min_max[0]
    cartesian_direction = spherical_to_cartesian( rays[...,2:3]*torch.pi,  rays[...,3:4]*torch.pi)
    return original_xy, cartesian_direction

#It should be cleaned
def normalize_points(points, min_max_scaler=None, standard_scaler=None, save_scalers=False, in_out = True, min_max_file =  'min_max_scaler_in.pkl'):
    if (not in_out):
        screen_size = np.floor(200 / np.sqrt(2))
        min_max = (-screen_size//2, screen_size//2)

        points_xy = 2* (points[...,:2] - min_max[0]) / (min_max[1]-min_max[0]) -1
        points[...,:2] = points_xy

    else:
        if min_max_scaler is None:
            min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
            points = min_max_scaler.fit_transform(points)
        else:

            points = min_max_scaler.transform(points)

        if (save_scalers and in_out):
            # Save the scalers for future use
            joblib.dump(min_max_scaler, min_max_file)

    return torch.tensor(points, dtype=torch.float32), min_max_scaler , None

def load_scalers(min_max_file ):
    min_max_scaler_in = joblib.load(min_max_file)
    return min_max_scaler_in

def pixelization(points, num_samples=1, n=(201, 201), screen=20):
    """
    Pixelization function with extended grid and border removal.

    Args:
        points (torch.Tensor): Existing samples of shape (batch_size, n_points, 2).
        num_samples (int): Number of additional samples to generate around existing points.
        n (tuple): Grid resolution (without border).
        screen (int): Unused parameter, kept for compatibility.
        noise_std (float): Standard deviation of Gaussian noise for new samples.

    Returns:
        torch.Tensor: Pixelized result with border pixels removed.
    """
    batch_size = points.shape[0]  
    all_points = points.unsqueeze(1) 
    center = torch.mean(all_points, dim=2, keepdim=True)
    n_ext = (n[0] + 2, n[1] + 2)
    # Pixelization grid (extended)
    pixel_size_x = 2.0 / (n_ext[0] - 2)
    pixel_size_y = 2.0 / (n_ext[1] - 2)
    # pixel_size_x = 2.0/ n_ext_div[0]
    # pixel_size_y=  2.0/ n_ext_div[1]


    x_vals = torch.linspace(-1 - pixel_size_x + pixel_size_x / 2, 1 + pixel_size_x - pixel_size_x / 2, n_ext[0])
    y_vals = torch.linspace(-1 - pixel_size_y + pixel_size_y / 2, 1 + pixel_size_y - pixel_size_y / 2, n_ext[1])
    grid_x, grid_y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    grid_samples = torch.stack([grid_x, grid_y], dim=-1)  # Shape (n_ext[0], n_ext[1], 2)

    mask = torch.ones(n_ext[0], n_ext[1])

    mask[0, :] = 1e6
    mask[-1, :] = 1e6
    mask[:, 0] = 1e6
    mask[:, -1] = 1e6

    std_dev_x = pixel_size_x / (1.448*1.6)
    std_dev_y = pixel_size_y /  (1.448*1.6)

    # std_dev_x = pixel_size_x /2
    # std_dev_y = pixel_size_y /2
    # Compute squared distances
    grid_samples_flat = grid_samples.view(-1, 1, 2).unsqueeze(0)  # Shape (1, n_ext*n_ext, 1, 2)
    squared_distances = torch.sum(((all_points - grid_samples_flat) ** 2), dim=-1) 
    distance_center = torch.norm(grid_samples_flat - center, dim= -1) # --> (1,n_ext**2, 1)
    #chnage from 1000 to smaller values for more optimized calculation
    limit_mask =  (distance_center <= 1000*pixel_size_x).float() 

    squared_distances_x = squared_distances / (2 * (std_dev_x ** 2))
    sigms = (torch.relu(1 * (abs(all_points[..., 0])-1))) + (torch.relu(1 *(abs(all_points[..., 1])-1))) 
    # print('------------------------------')
    sigms_out = sigms/ (sigms+1e-6)
    sigms_in = torch.relu(-1* (sigms)+1e-6)/torch.relu(torch.tensor(1e-6))
    sigms_end =  sigms_in

    squared_distance_out = (squared_distances_x * sigms_out)
    squared_distances = squared_distances_x * sigms_end
    gaussians = (torch.exp(-(squared_distances_x) ** 2))
    result_ext = gaussians.sum(dim=-1) + torch.sum(sigms *mask.reshape(-1).unsqueeze(0).unsqueeze(-1), dim =-1) # Sum over all points
    result_ext = result_ext * mask.reshape(-1).unsqueeze(0)
    result_ext = torch.log(result_ext + 1) / torch.log(torch.tensor(points.shape[1] + 1))
    result_ext = result_ext.view(result_ext.shape[0], n_ext[0],n_ext[1])
    result_ext = result_ext.reshape(result_ext.shape[0], -1)
    return result_ext

def display_images_in_grid(images, gt_images, rows=10, cols=1, name='test.png'):
    fig, axes = plt.subplots(rows, cols * 3, figsize=(6, 5))
    # print(images[0], rows)
    for i in range(rows):

        axes[i, 0].imshow(images[i], cmap='gray')
        if i == 0:
            axes[i, 0 ].set_title("Ray2Ray PSF")

        axes[i, 0].axis('off')  # Hide axes

        # Display GT in the second column
        axes[i, 1].imshow(gt_images[i], cmap='gray')
        if i == 0:
            axes[i, 1].set_title("Odak")
        axes[i, 1].axis('off')  # Hide axes

        axes[i, 2].imshow(abs(gt_images[i] - images[i]), cmap='gray')
        if i == 0:
            axes[i, 2].set_title("difference")
        axes[i, 2].axis('off')  # Hide axes

        plt.tight_layout()
        plt.savefig(name)
        
import time 

def angular_loss(v1, v2):
    v1 = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True)
    v2 = v2 / torch.linalg.norm(v2, dim=-1, keepdim=True)

    dot = torch.sum(v1 * v2, dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)  # ensures numerical stability
    angle_rad = torch.acos(dot)
    angle_deg = torch.rad2deg(angle_rad)

    return torch.mean(angle_deg)

def normalize_points_drawing(points):
    screen_size = torch.floor(200 / torch.sqrt(torch.tensor(2.0)))
    min_max = (-screen_size//2, screen_size//2)
    points_xy = 2* (points[...,:2] - min_max[0]) / (min_max[1]-min_max[0]) -1
    points[...,:2] = points_xy
    return points