import json
from train import train_model
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
from load_data import read_data, create_dataloaders
from utils import save_model, pixelization
import argparse 
import json


def ablation(data_path):
    #put series of lenses in folder named ablation- it is possible to change l and n
    directory_path = os.path.join(os.getcwd(), 'ablation')
    layers = [2, 4, 6, 8]
    neurons = [64, 128, 256, 512] 

    manager = multiprocessing.Manager()
    loss_dict = manager.dict()
    for i, filenamed in enumerate(os.listdir(directory_path)):
        print(f"Processing file: {filenamed}")
        # if '34.9' not in filenamed:
        #     continue  
        json_filename = filenamed+ 'ablation'  +'.json'
        if os.path.exists(json_filename):
            with open(json_filename, 'r') as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}
        manager = multiprocessing.Manager()
        loss_dict = manager.dict(existing_data)

        for i in range(0, len(layers), 2):
            processes = []
            for n in neurons:
                p = multiprocessing.Process(target=train_model, args=(directory_path, filenamed, layers[i], n, loss_dict))
                p.start()
                processes.append(p)
                p = multiprocessing.Process(target=train_model, args=(directory_path, filenamed, layers[i+1], n, loss_dict))
                p.start()
                processes.append(p)
                
                print(i, n, i+1, n)
            
            for p in processes:
                p.join()
                
        # for l in layers:
        #     processes = []
        #     for n in neurons:
        #         p = multiprocessing.Process(target=train_model, args=(directory_path, filenamed, l, n, loss_dict))
        #         p.start()
        #         processes.append(p)
                
        #     for p in processes:
        #         p.join()

        final_loss_data = dict(existing_data)
        final_loss_data.update(dict(loss_dict))

        print("Final loss dictionary:", final_loss_data)

        with open(json_filename, 'w') as json_file:
            json.dump(final_loss_data, json_file, indent=4)