import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm, trange
import yaml
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import CNN_Transformer
from dataset import load_NASA, BatteryDataset

# Load the YAML configuration file
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

# # Access the variables
NUM_CYCLES = cfg['NUM_CYCLES']
FEATURE_DIM = cfg['FEATURE_DIM']
EPOCHS = cfg['EPOCHS']
LEARNING_RATE = cfg['LEARNING_RATE']
BATCH_SIZE = cfg['BATCH_SIZE']

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
battery_dict = load_NASA(folder='NASA_DATA', scale_data=True)
battery_dict_no18 = {key: val for key, val in battery_dict.items() if key != 'B0018'}
dataset = BatteryDataset(battery_dict_no18, num_cycles=NUM_CYCLES)

# Train/test split
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# NN model
model = CNN_Transformer(feature_dim=FEATURE_DIM, num_cycles=NUM_CYCLES).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_loss = float('inf')
model.train()

t_range = trange(EPOCHS)
for epoch in t_range:
    train_losses = []
    for inputs, outputs in train_dataloader:
        inputs = inputs.float().to(device)
        outputs = outputs.float().to(device)
        predicted_outputs = model(inputs, outputs[:,:-1])

        optimizer.zero_grad()
        loss = criterion(predicted_outputs, outputs[:,-1].unsqueeze(-1))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    test_losses = []
    for inputs, outputs in test_dataloader:
        inputs = inputs.float().to(device)
        outputs = outputs.float().to(device)
        with torch.no_grad():
            predicted_outputs = model(inputs, outputs[:,:-1])
            test_loss = criterion(predicted_outputs, outputs[:,-1].unsqueeze(-1))
            test_losses.append(test_loss.item())

    # Print the loss for monitoring after each epoch
    t_range.set_description(f"train loss: {np.mean(train_losses)}, test loss: {np.mean(test_losses)}")
    t_range.refresh()

    # Check if the current loss is the best so far
    if np.mean(test_losses) < best_loss:
        best_loss = np.mean(test_losses)
        torch.save(model, 'trained_model.pt')