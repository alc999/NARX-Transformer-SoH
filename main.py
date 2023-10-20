import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import CNN_Transformer
from dataset import load_NASA, BatteryDataset

# Load data
battery_dict = load_NASA(folder='NASA_DATA', scale_data=True)
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