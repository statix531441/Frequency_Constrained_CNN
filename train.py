# Defaults
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# PyTorch
import torch
import torch.nn as nn
from torch.functional import F

# Local
from dataset import *
from models import *
from options import Options
from utils import *


# Utilities
import os
from tqdm import tqdm
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device, " used for training")

##################################################################

opt = Options(dataset='ECG', model='SmoothCNN', tag='')
opt.epochs = 200
opt.pct = 0.2


### Name and create model folder (Replace opt. with args. once argparse is implemented)
opt.model_folder = f"models/{opt.dataset}/{opt.split}/{opt.model}" + f"_pct{opt.pct}" + f"_{opt.tag}"
os.makedirs(opt.model_folder, exist_ok=True)

### Update opt with changes, Initialize directories and save options
opt.data_folder = f"data/{opt.dataset}/{opt.split}_{f'Noisy({opt.mean}-{opt.std})' if opt.add_noise else ''}"
os.makedirs(opt.data_folder, exist_ok=True)

opt.save_options(opt.model_folder)

### 🔐Load train.csv and test.csv
try:
    train_df = pd.read_csv(f'{opt.data_folder}/train.csv', header=None)
    test_df = pd.read_csv(f'{opt.data_folder}/test.csv', header=None)
except:
    print(f"Creating train, test split in {opt.data_folder}")
    train_df, test_df = create_split(opt)


### 🔐Select dataset and create dataloaders
Dataset = Datasets[opt.dataset]
train_set = Dataset(train_df)
test_set = Dataset(test_df)
train_loader = DataLoader(train_set, shuffle=True, batch_size=opt.batch_size)
test_loader = DataLoader(test_set, shuffle=True, batch_size=opt.batch_size)

### 🔐Select and load model
model = Models[opt.model](opt).to(device)

### Initialize loss, optimizer and train history
lossFn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_accuracy": []
}


### 🔐Train Loop 
for epoch in tqdm(range(opt.epochs)):

    train_loss, train_accuracy = fit(model, train_loader, lossFn, optimizer)
    _, _, test_accuracy = predict(model, test_loader)

    ############### Save Model ################
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if len(history['test_accuracy']) and test_accuracy >= max(history['test_accuracy']):
        torch.save(state, f"{opt.model_folder}/best.pth")
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_accuracy)
    history['test_accuracy'].append(test_accuracy)

    torch.save(history, f"{opt.model_folder}/history.pth")
    torch.save(state, f"{opt.model_folder}/latest.pth")
    ###########################################
