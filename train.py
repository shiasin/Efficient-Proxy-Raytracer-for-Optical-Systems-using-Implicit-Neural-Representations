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
from utils import save_model, pixelization
import argparse 
import json



seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def traint(data_directory, train_ratio=0.8, batch_size=3, num_layers = 8, num_nodes = 256, num_epochs=7002,\
           weight_path = 'model.path', rayrayoutput='rayrayoutput.png', save_dir='./saved_images2', min_max_file='min_max_scaler_in.pkl',\
          device="cuda:2"):
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    # Read data
    # source, non_man_psfs, outs, man_psfs = load_data(data_directory, train_ratio = train_ratio, save_scalers=True, min_max_file=min_max_file)
    train_data, test_data, num_rays = load_data(data_directory, device)

    train_loader, test_loader = create_dataloaders(train_data[0], train_data[2], test_data[0], test_data[2], batch_size=1)

    model = Ray2Ray(num_layers=num_layers, hidden_size = num_nodes)
    model.train()

    # try with constant learning rate
    initial_lr = 5e-4
    final_lr = 1e-6

    gamma = (final_lr / initial_lr) ** (1 / num_epochs)

    optimizer = torch.optim.AdamW(list(model.parameters()), lr=initial_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma= gamma)

    os.makedirs(save_dir, exist_ok=True)

    criterion = lambda x,y: torch.mean(abs(x-y))
    criterion2 = lambda x,y: torch.mean(abs(x-y)**2)
    losses = []

    for epoch in range(num_epochs):
        mean_loss = []
        
        model.train()
        for batch in train_loader:
            random_rays = torch.randint(0, num_rays, (200,))
            inputs = batch[0][:, random_rays, :].to(device)  #(x, y, tetha, phi, wv)
            targets = batch[1][:, random_rays, :].to(device)

            optimizer.zero_grad()

            outputs = model(inputs) 
            
            results = outputs

            cra1p = 1* criterion(results[...,:2], targets[...,:2])
            cra1d = 1* criterion(results[...,2:], targets[...,2:])
            # print(results, targets)
            cra2p = 1* criterion2(results[...,:2], targets[...,:2])
            cra2d = 1* criterion2(results[...,2:], targets[...,2:])
            
            loss =  cra1p + 1*cra1d
            mean_loss.append(loss.item())
            loss.backward()

            optimizer.step()

        losses.append(sum(mean_loss)/ len(mean_loss))


        if epoch % 1000 == 0:
            model.eval()
            random_sample = random.choice(train_loader.dataset)
            random_test = random.choice(test_loader.dataset)

            inputs_val = random_sample[0].unsqueeze(0)
            inputs_test = random_test[0].unsqueeze(0)

            target_val = random_sample[1]
            target_test = random_test[1]
            with torch.no_grad():
                outputs_val = model(inputs_val)
                output_test = model(inputs_test)
                val_loss = criterion(outputs_val, target_val)
                test_loss =  criterion(output_test, target_test)

                val_loss2 = criterion2(outputs_val, target_val)
                test_loss2_p =  criterion2(output_test[...,:2], target_test[...,:2])
                test_loss2_d =  criterion2(output_test[..., 2:], target_test[..., 2:])

                val_wv = inputs_val[..., -1]
                test_wv = inputs_test[...,-1]

                output_image = pixelization(outputs_val[...,:2]).cpu().numpy().reshape(203, 203)
                test_out = pixelization(output_test[...,:2]).cpu().numpy().reshape(203, 203)
                ref_out = pixelization(target_val[...,:2].unsqueeze(0)).cpu().numpy().reshape(203, 203)
                ref_test = pixelization(target_test[...,:2].unsqueeze(0)).cpu().numpy().reshape(203, 203)

                diff = abs(ref_test - test_out)
                diff_eval = abs(ref_out - output_image)
                fig, axes = plt.subplots(1, 7, figsize=(18, 8))

                axes[0].imshow(output_image, cmap='gray')
                axes[0].set_title("Output Image")
                axes[0].axis("off")

                axes[1].imshow(ref_out, cmap='gray')
                axes[1].set_title("Reference Image")
                axes[1].axis("off")

                axes[2].imshow(test_out, cmap='gray')
                axes[2].set_title("Test Image")
                axes[2].axis("off")

                axes[3].imshow(ref_test, cmap='gray')
                axes[3].set_title("Test Reference")
                axes[3].axis("off")

                axes[4].imshow(diff_eval, cmap='gray')
                axes[4].set_title("Evaluation Difference")
                axes[4].axis("off")

                axes[5].imshow(diff, cmap='gray')
                axes[5].set_title("Test Difference")
                axes[5].axis("off")

                axes[6].plot(losses, marker='o', color='blue')
                axes[6].set_title("Loss Curve")
                axes[6].set_xlabel("Epoch")
                axes[6].set_ylabel("Loss")
                axes[6].grid(True)

                for i in range(6):
                    fig.colorbar(axes[i].images[0], ax=axes[i], fraction=0.05, pad=0.02)

                print('---------------------validation---------------------------')
                plt.suptitle(f"Epoch {epoch} - Validation l1 Output - Loss {val_loss.item()} - TLoss {test_loss.item()}\n Validation l2 Output - \
                             Loss {val_loss2.item()} - TLossp{test_loss2_p.item()} - TLossd{test_loss2_d.item()}")
                plt.tight_layout()

                plt.savefig(os.path.join(save_dir, f"{epoch}"+ rayrayoutput))

            print(f"Epoch {epoch}/{num_epochs}, Validation Loss: {val_loss.item():.8f}")
        if (epoch % (num_epochs-2)==0) and (epoch!=0):
            model.eval()

            epoch_added = weight_path.split('.pth')[0] + '_' + str(epoch)+ '.pth'
            save_model(model, optimizer, epoch,loss, scheduler = scheduler, path=epoch_added)
            avg_loss_p = 0
            avg_loss_d = 0
            counter = 0
            with torch.no_grad():
                for td in test_loader:
                    inputs = td[0] #(x, y, tetha, phi, wv)
                    targets = td[1]

                    outputs = model(inputs)
                    results = outputs

                    # print(inputs, 1* criterion2(results[...,:2], targets[...,:2]), 1* criterion2(results[...,2:], targets[...,2:]))
                    avg_loss_p += 1* criterion2(results[...,:2], targets[...,:2])
                    avg_loss_d += 1* criterion2(results[...,2:], targets[...,2:])
                    counter = counter + 1
                return avg_loss_p/counter, avg_loss_d/counter
        # Print training loss every 100 epochs
        if epoch % 500 == 0:
            print('-------------------------------------------------------')

            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.8f} cra1p: {cra1p.item():.8f} cra1d: {cra1d.item():.8f} cra2p: {cra2p.item():.8f} cra2d: {cra2d.item():.8f} ")
            # print(loss_pos.item(), loss_dir.item())
            print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"Gradient norm for {name}: {grad_norm}")
                    if grad_norm < 1e-6:
                        print(f"Warning: Vanishing gradient detected for {name}")
        scheduler.step()


def train_model(directory_path, filenamed, l, n, min_max_file, loss_dict = dict(), num_epochs=3002, device = "cuda:2"):
    filename = f"{filenamed}_{l}_{n}"
    print(f"Starting training: {filename}")

    avg_loss_p, avg_loss_d = traint(
        os.path.join(directory_path, filenamed),
        weight_path=os.path.join(os.getcwd(), 'IamT', filename + '.pth'),
        rayrayoutput=filename + '.png',
        save_dir='./IamT1',
        num_layers=l,
        num_nodes=n,
        num_epochs=num_epochs, 
        min_max_file = min_max_file,
        device = device
    )

    print(f"Finished training: {filename}, avg_loss_p: {avg_loss_p}, avg_loss_d: {avg_loss_d}")

    # Update the shared dictionary
    loss_dict[f"{l}_{n}"] = [avg_loss_p.item(), avg_loss_d.item()]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root_dir', type=str,
                        required=True,
                        help='very first root pass of the lens data, ex, processed_data')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Exact folder name of the lens. ex; myd')
    
    parser.add_argument('--scaler_file', type=str,
                        default='min_max_scaler_in.pkl',
                        help='Exact folder name of the lens.')
    
    parser.add_argument('--device', type=str,
                        default='cuda:2',
                        help='cuda/cpu')

    parser.add_argument('--epochs', type=int,
                        default='3002',
                        help='Directory to save the generated dataset.')
    
    args = parser.parse_args()

    directory_path = os.path.join(os.getcwd(), args.data_root_dir)

    #other arguments for more flexible training will be added 
    train_model(directory_path , args.data_folder, min_max_file = args.scaler_file, l=4, n=128, num_epochs= args.epochs, device = args.device)

if __name__ == "__main__":
    main()

