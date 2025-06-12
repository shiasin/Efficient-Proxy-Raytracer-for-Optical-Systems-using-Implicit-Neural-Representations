import argparse
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
from load_data import load_data, create_dataloaders
from utils import save_model, load_model, cartesian_to_spherical, spherical_to_cartesian, \
normalize_objectdistance, unnormalize_output, normalize_points, load_scalers, pixelization, display_images_in_grid, angular_loss, normalize_points_drawing


def test(data_directory, data_folder, model_path, l=4, h=128, t= 300, min_max_scaler = 'min_max_scaler_in.pkl', separated_test = None, device ="cuda:2"):
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    
    data_directory =  os.path.join(os.getcwd(), data_directory)
    
    # for x in os.listdir(data_directory):
    #     # if '_50.6' not in x:
    #     #     continue
    d = os.path.join(data_directory, data_folder)
    train_data, test_data, num_rays = load_data(d, device=device)
        # source, non_man_psfs, outs, man_psfs = read_data(d, save_scalers=False, load_test=True, min_max_scaler = min_max_scaler)

    train_loader, test_loader = create_dataloaders(None, None, test_data[0], test_data[2], batch_size=2, test=True)
    points_on = [] 
    model = Ray2Ray(hidden_size = 128, num_layers=4)
    model = load_model(model, path=model_path, test=True)
    criterion2 = lambda x,y: torch.mean(abs(x-y)**2)
    criterion1 = lambda x,y: torch.mean(abs(x-y))
    model.eval()
    loss1= 0
    loss2= 0
    error2_mm= 0
    error1_mm = 0
    dir_loss = 0
    with torch.no_grad():
        count = 0
        for i in test_loader:
            if count == 0 :
                dummy = torch.randn_like(i[0])
                _ = model(dummy)
            gt = i[1]

            out = model(i[0])
            origin_in , dir_in  =  unnormalize_output(gt) 
            origin_out, dir_out =  unnormalize_output(out)
            points_on.append(i[0][:,0:1,:2])

            error2_mm += criterion2(origin_in, origin_out)
            error1_mm += criterion1(origin_in, origin_out)
            dir_loss += angular_loss(dir_in, dir_out)

            print(error2_mm, error1_mm, dir_loss)
            loss2 +=(1*criterion2(i[1][:,:,:2], out[:,:,:2]) +criterion2(i[1][:,:,2:4], out[:,:,2:4])).item()
            loss1 += (1*criterion1(i[1][:,:,:2], out[:,:,:2])+criterion1(i[1][:,:,2:4], out[:,:,2:4])).item()

            ones = torch.ones(origin_in.shape[:-1] + (1,))

            draw_in = torch.cat([origin_in, ones], axis=-1)+ t* dir_in.squeeze()

            draw_out = torch.cat([origin_out, ones], axis=-1) + t* dir_out.squeeze()     
            # model_res = pixelization(normalize_points_drawing(draw_out[:, :, :2]))
            # gt_res = pixelization(normalize_points_drawing(draw_in[:, :, :2]))
            # display_images_in_grid(model_res.cpu().numpy().reshape(model_res.shape[0], 203, 203), 
            #                         gt_res.cpu().numpy().reshape(gt_res.shape[0], 203, 203), rows=gt.shape[0], 
            #                         name = f'ray2set{count}.pdf')
            count = count + 1
        print(error1_mm/count, error2_mm/count, dir_loss/count, count, loss1/count, loss2/count)
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root_dir', type=str, default='data', required=True,
                        help='your input data- what you have used for training this lens. structure of files should be follow the raytracing_dataset structure.')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Exact folder name of the lens. ex; myd')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file (.pth or similar). This argument is required.')
    parser.add_argument('--t', type=int, default=300,
                        help='traveling from the target plane at 1.')
    
    parser.add_argument('--h', type=int, default=128,
                        help='neuron size')
        
    parser.add_argument('--l', type=int, default=4,
                        help='layers')
    parser.add_argument('--min_max_scaler', type=str, default='min_max_scaler_in.pkl',
                        help='scaler file. should be same with the one used for training')
    
    parser.add_argument('--device', type=str,
                        default='cuda:2',
                        help='cuda/cpu')


    args = parser.parse_args()

    test(data_directory=args.data_root_dir,
         data_folder = args.data_folder,
         model_path=args.model_path,
         t=args.t,
         min_max_scaler=args.min_max_scaler,
         device = args.device,
         l= args.l,
         h= args.h
         )

if __name__ == '__main__':
    main()
    
    