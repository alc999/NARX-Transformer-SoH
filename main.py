import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm, trange
import yaml
import os
import seaborn as sns
sns.set_theme()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import NARX_Transformer
from dataset import load_NASA

# Load the YAML configuration file
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

# # Access the variables
NUM_CYCLES = cfg['NUM_CYCLES']
NUM_PREDS = cfg['NUM_PREDS']
FEATURE_DIM1 = cfg['FEATURE_DIM1']
FEATURE_DIM2 = cfg['FEATURE_DIM2']
NUM_ATTENTION = cfg['NUM_ATTENTION']
EPOCHS = cfg['EPOCHS']
LEARNING_RATE = cfg['LEARNING_RATE']
BATCH_SIZE = cfg['BATCH_SIZE']

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_dataset, test_dataset = load_NASA(folder='NASA_DATA', num_cycles=NUM_CYCLES+NUM_PREDS-1, split_ratio=0.5, scale_data=True)

# Train/test split
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# NN model
model = NARX_Transformer(feature_dim1=FEATURE_DIM1, 
                         feature_dim2=FEATURE_DIM2, 
                         num_attention=NUM_ATTENTION, 
                         num_cycles=NUM_CYCLES, 
                         num_preds=NUM_PREDS).to(device)

# Loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_epoch = 0
best_loss = float('inf')
Loss_log = []
os.makedirs('models', exist_ok=True)

model.train()
t_range = trange(EPOCHS)
for epoch in t_range:
    train_losses = []
    for inputs, outputs in train_dataloader:
        inputs = inputs.float().to(device)
        outputs = outputs.float().to(device)
        predicted_outputs = model.pred_sequence(inputs, outputs)

        optimizer.zero_grad()
        loss = criterion(predicted_outputs[:,NUM_CYCLES-1:], outputs[:,NUM_CYCLES-1:])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    test_losses = []
    for inputs, outputs in test_dataloader:
        inputs = inputs.float().to(device)
        outputs = outputs.float().to(device)
        with torch.no_grad():
            predicted_outputs = model.pred_sequence(inputs, outputs)
            test_loss = criterion(predicted_outputs[:,NUM_CYCLES-1:], outputs[:,NUM_CYCLES-1:])
            test_losses.append(test_loss.item())
    Loss_log.append([np.mean(train_losses),np.mean(test_losses)])

    # Print the loss for monitoring after each epoch
    t_range.set_description(f"train loss: {np.mean(train_losses)}, test loss: {np.mean(test_losses)}")
    t_range.refresh()

    # Check if the current loss is the best so far
    if np.mean(test_losses) < best_loss:
        best_epoch = epoch
        best_loss = np.mean(test_losses)
        torch.save(model, f'models/trained_model_{best_loss:.6f}_{best_epoch}.pt')

Loss_log = np.array(Loss_log)
np.save(f'models/training_log_{NUM_CYCLES}_{NUM_PREDS}.npy', Loss_log)

plt.figure(figsize=(8,5))
plt.plot(Loss_log[:best_epoch, 0])
plt.plot(Loss_log[:best_epoch, 1])
plt.legend(["Train Loss","Test Loss"])
plt.grid("on")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()