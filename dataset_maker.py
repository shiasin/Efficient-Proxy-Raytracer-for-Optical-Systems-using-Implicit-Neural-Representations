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
import argparse
from utils import save_model, load_model, cartesian_to_spherical, spherical_to_cartesian, \
normalize_objectdistance, unnormalize_output, normalize_points, load_scalers, pixelization


def read_data(base_directory, save_scalers=False, load_test=False, min_max_file=None):
    all_data = {}

    if load_test:
        min_max_scaler_in = load_scalers(min_max_file)

    else:
        min_max_scaler_in = None
        save_scalers = True

    point_map = {}
    sources = []
    outs = []
    gaussian_pixelized = []
    non_man_psfs = []
    man_psfs = []
    for folder in os.listdir(base_directory):
        if not folder.startswith("data_"):
            continue
        # very kitanai, in this step it is not important so continue with kitanai. 
        object_distance = torch.tensor(normalize_objectdistance(float(folder.split('_')[-1])), dtype = torch.float32)
        folder_path = os.path.join(base_directory, folder)
        source_path = os.path.join(folder_path, 'source_rays.npy')
        out_path = os.path.join(folder_path, 'out_rays.npy')
        non_man_psf_path =  os.path.join(folder_path, 'psfs.npy')
        psf_directory = os.path.join(folder_path, 'psfs')

        if not (os.path.exists(source_path) and os.path.exists(out_path) and os.path.exists(psf_directory)):
            print(f"Skipping {folder}: Missing required files.")
            continue

        source_rays = torch.from_numpy(np.load(source_path)).to(torch.float32)
        
        out_rays = torch.from_numpy(np.load(out_path)).to(torch.float32)

        non_man_psf = torch.from_numpy(np.load(non_man_psf_path)).to(torch.float32)
        unnormalized_source_points = source_rays[:, 0, :3].numpy()
        unnormalized_out_points = out_rays[:, 0, :3].numpy()

        source_rays[:, 0, :3], min_max_scaler_in, _ = normalize_points(
            source_rays[:, 0, :3].numpy(), min_max_scaler_in, save_scalers = save_scalers, min_max_file = min_max_file)

        out_rays[:, 0, :3], _, _ = normalize_points(
            out_rays[:, 0, :3].numpy(), save_scalers = save_scalers, in_out=False, min_max_file = min_max_file)
        out_rays[:,0, 2] = object_distance

        source_r, source_tetha, source_phi = cartesian_to_spherical(source_rays[:, 1, :3].numpy())
        out_r, out_tetha, out_phi = cartesian_to_spherical(out_rays[:, 1, :3].numpy())
        print(out_rays[:, 1, :2].min(), out_rays[:, 1, :2].max(), source_rays[:, 1, :2].min(), source_rays[:, 1, :2].max())

        source_rays[:, 1, 0] = torch.tensor(source_tetha) / torch.pi

        source_rays[:, 1, 1] = torch.tensor(source_phi)/torch.pi

        out_rays[:, 1, 0] = torch.tensor(out_tetha) /torch.pi
        
        out_rays[:, 1, 1] = torch.tensor(out_phi)/torch.pi

        source_rays[:, 1, 2] =  torch.tensor(source_r)
        out_rays[:, 1, 2] = torch.tensor(out_r)
        # print(spherical_to_cartesian(out_rays[:, 1,:1]*torch.pi,out_rays[:,1,1:2]*torch.pi))
        #lets put the wavelength here for now divided by 1000. Wavelength is addded to all rays for later update
        source_rays[:, 0, 2] = 0
        source_rays[:, 0, -1:] =  source_rays[:, 0, -1:] / 1000
        parent_positions = torch.cat([source_rays[:, 0, :3], source_rays[:, 0, -1:]], dim=-1)

        unique_parents, inverse_indices = torch.unique(parent_positions, dim=0, return_inverse=True)

        unnormalized_parents = min_max_scaler_in.inverse_transform(unique_parents[...,:-1].numpy())  # Reverse min-max scaling
        categorized_source = defaultdict(list)
        categorized_out = defaultdict(list)
        # pay attention to the order
        rays_num = torch.randperm(source_rays.shape[0]).to(source_rays.device)

        for i, idx in enumerate(inverse_indices):
            categorized_source[idx.item()].append(source_rays[i])
            categorized_out[idx.item()].append(out_rays[i])
        categorized_source = {key: torch.stack(value) for key, value in categorized_source.items()}
        categorized_out = {key: torch.stack(value) for key, value in categorized_out.items()}

        psf_files = glob.glob(os.path.join(psf_directory, "*psf_*.png"))
        categorized_psfs = {}
        for psf_file in psf_files:
            filename = os.path.basename(psf_file)
            parts = filename.replace('.png', '').split('_')

            try:
                px_x, px_y = map(float, parts[-2:])
            except ValueError:
                continue

            for i, parent_pos in enumerate(unnormalized_parents):
                parent_x, parent_y = parent_pos[:2]
                if np.isclose(parent_x, px_x) and np.isclose(parent_y, px_y):
                    img = Image.open(psf_file).convert("L")
                    img_array = np.array(img).astype(np.float32)

                    img_min, img_max = img_array.min(), img_array.max()
                    if img_max > img_min:
                        img_array = (img_array - img_min) / (img_max - img_min)
                    else:
                        img_array = np.zeros_like(img_array)

                    img_tensor = torch.from_numpy(img_array).flatten()
                    categorized_psfs[i] = img_tensor
                    break

        min_rays = min(tensor.shape[0] for tensor in list(categorized_source.values()))
        sampled_tensors = [torch.randperm(tensor.shape[0],device=tensor.device)[:min_rays] for tensor in categorized_source.values()]
 
        categorized_source =  torch.stack([tensor[sampled_tensors[i]] for i, tensor in enumerate(categorized_source.values())])

        source = torch.cat([categorized_source[:, :, 0, :2], categorized_source[:, :, 1, :2], categorized_source[:,:,0, -1:]], dim=-1)
        categorized_out = torch.stack([tensor[sampled_tensors[i]] for i, tensor in enumerate(categorized_out.values())])

        out = torch.cat([categorized_out[:, :, 0, :2], categorized_out[:, :, 1, :2]], dim=-1)

        gtemp = []
        #instead of using psfs in dataset, I am using  pixelization with gaussian here to have same method when comparing; 
        for i in range(out.shape[0]):
            rand_rays = torch.randperm(out.shape[1], device=out.device)[0:1024]
            res = pixelization(torch.tensor(out[i,  rand_rays, :2].unsqueeze(0)), n=(201, 201))
            gtemp.append(res)

        gaussian_pixelized.append(torch.cat(gtemp))
        sources.append(source)
        outs.append(out)
        val  = torch.log(torch.tensor(non_man_psf.reshape(non_man_psf.shape[0],-1))+ torch.tensor(1.0)) / torch.log(torch.tensor(categorized_out.shape[1]+1.0))
        non_man_psfs.append(val)
        man_psfs.append(torch.stack(list(categorized_psfs.values())))
        
    return torch.tensor(torch.cat(sources)), torch.tensor(torch.cat(gaussian_pixelized)), torch.tensor(torch.cat(outs)), torch.tensor(torch.cat(man_psfs))


def dataset_maker(base_directory, output_dir, train_ratio=0.8, save_scalers=False, load_test=False, min_max_file='min_max_scaler_in.pkl'):

    print(f"Starting dataset_maker for base_directory: {base_directory}")
    print(f"Output directory: {output_dir}")

    sources, gaussian_pixelized, outs, man_psfs = read_data(
        base_directory, save_scalers, load_test, min_max_file
    )

    if sources is None or sources.shape[0] == 0:
        print("No data found or loaded. Exiting dataset_maker.")
        return

    n = sources.shape[0]

    perm = torch.randperm(n) # Permutation of indices
    train_size = int(train_ratio * n)
    train_indices, test_indices = perm[:train_size], perm[train_size:]

    sources_train = sources[train_indices]
    gaussian_pixelized_train = gaussian_pixelized[train_indices]
    outs_train = outs[train_indices]
    man_psfs_train = man_psfs[train_indices]

    sources_test = sources[test_indices]
    gaussian_pixelized_test = gaussian_pixelized[test_indices]
    outs_test = outs[test_indices]
    man_psfs_test = man_psfs[test_indices]
    
    def _save_data_to_folder(sources, gaussian_pixelized, outs, man_psfs, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        try:
            np.save(os.path.join(output_folder, 'sources.npy'), sources.cpu().numpy())
            np.save(os.path.join(output_folder, 'gaussian_pixelized.npy'), gaussian_pixelized.cpu().numpy())
            np.save(os.path.join(output_folder, 'outs.npy'), outs.cpu().numpy())
            np.save(os.path.join(output_folder, 'man_psfs.npy'), man_psfs.cpu().numpy())
            print("All data saved successfully!")
        except Exception as e:
            print(f"Error saving data: {e}")


    train_output_folder = os.path.join(output_dir, 'train')
    test_output_folder = os.path.join(output_dir, 'test')

    _save_data_to_folder(sources_train, gaussian_pixelized_train, outs_train, man_psfs_train, train_output_folder)

    _save_data_to_folder(sources_test, gaussian_pixelized_test, outs_test, man_psfs_test, test_output_folder)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_directory', type=str, required=True,
                        help='Path to the directory containing raw data (e.g., "./raw_data").')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where "train" and "test" subfolders will be created for saving processed data.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data to use for the training set (e.g., 0.8 for 80%% train, 20%% test).')
    parser.add_argument('--save_scalers', action='store_true',
                        help='If set, the MinMax scaler will be saved during data processing.')
    parser.add_argument('--load_test', action='store_true',
                        help='If set, test data will be loaded using pre-saved scalers (check your read_data implementation for details).')
    parser.add_argument('--min_max_file', type=str, default='min_max_scaler_in.pkl',
                        help='Path to the MinMax scaler pickle file (used for saving or loading scalers).')

    args = parser.parse_args()

    dataset_maker(
        base_directory=args.base_directory,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        save_scalers=args.save_scalers,
        load_test=args.load_test,
        min_max_file=args.min_max_file
    )

if __name__ == '__main__':
    main()