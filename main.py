import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import CNN_Transformer
from dataset import BatteryDataset

# Load data
charge_data = pd.read_csv('charge_data.csv')
data_x = np.load('data_x.npy')
data_y = np.load('data_y.npy')

sc_voltage = MinMaxScaler(feature_range=(-1,1))
sc_current = MinMaxScaler(feature_range=(-1,1))
sc_temp    = MinMaxScaler(feature_range=(-1,1))

data_x[:,:,0] = sc_voltage.fit_transform(data_x[:,:,0])
data_x[:,:,1] = sc_current.fit_transform(data_x[:,:,1])
data_x[:,:,2] = sc_temp.fit_transform(data_x[:,:,2])

battery_dict = {}
for name in np.unique(charge_data['Battery_id']):
    battery_dict[name] = {}
    battery_dict[name]['data'] = data_x[charge_data[charge_data['Battery_id']==name].index]
    battery_dict[name]['cap']  = data_y[charge_data[charge_data['Battery_id']==name].index]

    idx,_,_ = np.where(np.isnan(battery_dict[name]['data']))
    battery_dict[name]['data'] = np.delete(battery_dict[name]['data'], idx, axis=0)
    battery_dict[name]['cap'] = np.delete(battery_dict[name]['cap'], idx, axis=0)

dataset = BatteryDataset(battery_dict, num_cycles=3)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# NN model
model = CNN_Transformer()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define the number of epochs for training
num_epochs = 10

# Training loop
model.train()
for epoch in range(num_epochs):    
    for inputs, outputs in dataloader:
        # Convert inputs and outputs to PyTorch tensors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        outputs = torch.tensor(outputs, dtype=torch.float32)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        predicted_outputs = model(inputs, outputs[:-1])
        
        # Compute the loss
        loss = criterion(predicted_outputs, outputs[-1])
        
        # Backward pass
        loss.backward()
        
        # Update the weights
        optimizer.step()
        
    # Print the loss for monitoring after each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# After the training loop, you can save the trained model if needed
torch.save(model.state_dict(), 'trained_model.pth')