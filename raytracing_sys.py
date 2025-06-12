import torch
from elements import Sensor, PlaneSurface , SphereSurface, ProjectionSys, Render # Import the Sensor class from elements.py
import json
import numpy as np
import os
from collections import defaultdict
import glob
from PIL import Image
import torch.nn.functional as F
import time
import random
import argparse
import ast

#This should be updated
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def read_lens_json(file_path, wave_length=[589]):
    with open(file_path, 'r') as f:
        data = json.load(f)

    sensor_dict = data.get("sensor", {})

    lens_dict = data.get("lenses", {})
    aperture_dict = data.get("aperture", {})

    object_dict = data.get("target", {})
    focal  = data.get("focal_length", 0)
    mode = data.get('mode', 'row')
    wavelengths = data.get('wavelengths', [589])
    # wavelengths = wave_length
    return sensor_dict, lens_dict, aperture_dict, object_dict, mode,focal, wavelengths



def raytracing(lens_file, output_directory, image_size_px=[201, 201], kernel_size=(201, 201), wave_length = [589]):
    sensor_dict, lens_dict, aperture_dict, object_dict, rmode, focal, wavelengths  = read_lens_json(lens_file, wave_length=wave_length)
    com_output_directory = (
        f"{output_directory}_{lens_dict[0]['surface1']['radius_of_curvature']}_"
        f"{lens_dict[0]['surface2']['radius_of_curvature']}_"
        f"{lens_dict[0]['tc']}_"
        f"{lens_dict[0]['te']}_"
        f"{lens_dict[0]['diameter']}"
    )
    
    projection_systems = [ProjectionSys(sensor_dict, lens_dict, aperture_dict, rmode, object_dict, focal, wavelength_nm=wv) for wv in wavelengths]
    # final_intersections, final_rays, image_distance, object_distance = [projection_system.forward() for projection_system in projection_systems]
    start_time = time.time()
    all_wvs = [list(projection_system.forward()) for projection_system in projection_systems]
    end_time = time.time()
    _, _, image_distance, object_distance = all_wvs[0]
    final_intersections, final_rays = [], []
    for wv in all_wvs:
        final_intersections.append(wv[0])
        final_rays.append(wv[1])
    final_intersections = torch.cat(final_intersections, dim=0)
    # print(final_intersections.shape)
    # exit()
    final_rays = torch.cat(final_rays, dim=0)
    new_dir = os.path.join(os.getcwd(), output_directory)
    os.chdir(new_dir)
    # exit()
    scene_directory = os.path.join(com_output_directory, f"data_{image_distance}_{object_distance}")
    os.makedirs(scene_directory, exist_ok=True)


    source_rays = final_rays[:, :, 3:]
    np.save(os.path.join(scene_directory, 'source_rays.npy'), source_rays[~torch.isnan(source_rays).any(dim=(1, 2))].cpu().numpy())
    out_rays = final_rays[:, :, :3]
    # exit()
    print('-----------------------------')

    np.save(os.path.join(scene_directory, 'out_rays.npy'), out_rays[~torch.isnan(source_rays).any(dim=(1, 2))].cpu().numpy())

    psfs_directory = os.path.join(scene_directory, "psfs")
    os.makedirs(psfs_directory, exist_ok=True)

    render = Render(image_size_mm=(torch.floor(torch.tensor(object_dict.get('diameter'))/ torch.sqrt(torch.tensor(2.0))), \
                                   torch.floor(torch.tensor(object_dict.get('diameter'))/ torch.sqrt(torch.tensor(2.0)))), \
                    image_size_px=image_size_px, kernel_size=kernel_size,
                    centered=False, positional=True, output_directory=psfs_directory)

    rendered_images, non_man_psf = render(final_intersections, final_rays, image_distance, object_distance)
    for i, image in enumerate(rendered_images):
        image.save_image()

    if non_man_psf != None:
        np.save(os.path.join(scene_directory, 'psfs.npy'), non_man_psf.cpu().numpy())

    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lens_file', type=str, required=True,
                        help='Path to the lens JSON file. ex. ./lenses/one-lens/lens01.json')

    parser.add_argument('--output_directory', type=str, required = True,
                        help='Directory to save the generated dataset. ex , IamT')

    parser.add_argument('--image_size_px', type=str, # Use str and then ast.literal_eval
                        default='(201, 201)',
                        help='Image size in pixels as a tuple (width, height). E.g., "(201, 201)". It is not being used in our code.')

    parser.add_argument('--kernel_size', type=str, # Use str and then ast.literal_eval
                        default='(31, 31)',
                        help='Kernel size as a tuple (width, height). E.g., "(31, 31)"')

    args = parser.parse_args()

    try:
        image_size_px_tuple = ast.literal_eval(args.image_size_px)
        kernel_size_tuple = ast.literal_eval(args.kernel_size)
    except (ValueError, SyntaxError) as e:
        parser.error(f"Invalid tuple format for image_size_px or kernel_size: {e}")

    if not isinstance(image_size_px_tuple, tuple) or not isinstance(kernel_size_tuple, tuple):
        parser.error("image_size_px and kernel_size must be provided as tuples, e.g., '(201, 201)'")


    raytracing(lens_file=args.lens_file,
                  output_directory=args.output_directory,
                  image_size_px=image_size_px_tuple,
                  kernel_size=kernel_size_tuple)

if __name__ == '__main__':
    main()