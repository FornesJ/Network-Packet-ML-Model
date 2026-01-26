import torch
import torch.nn as nn
from model.model_utils.normalization import L2ByteNorm

class CNN(nn.Module):
    def __init__(self, input_dim, filters, linear_sizes, flatten_dim, dropout, classes=24):
        super().__init__()
        self.filters = filters
        self.conv_layers = len(filters)
        self.embedding = L2ByteNorm(idx=13, dim=2)

        # add convolutional layers and maxpool
        conv_layers = []
        in_channels = input_dim
        for out_channels, kernel_size, pool in filters:
            # add Conv + ReLu + Dropout layer
            conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ))

            # add MaxPool layer
            if pool:
                conv_layers.append(nn.MaxPool1d(2))
            
            in_channels = out_channels # new in_channel is out channel
        self.conv = nn.ModuleList(conv_layers) 

        # flatten and layer norm
        self.flatten = nn.Flatten()
        self.ln1 = nn.LayerNorm(flatten_dim)

        # add fully connected layers
        layers = []
        in_dim = flatten_dim
        for h in linear_sizes:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_dim = h

        self.linear = nn.ModuleList(layers)
        self.ln2 = nn.LayerNorm(in_dim)
        self.output = nn.Linear(in_dim, classes)
    
    def forward(self, x):
        x = self.embedding(x)

        # Convolutional layers
        for conv in self.conv:
            x = conv(x)
        
        # Flatten and layer norm
        x = self.flatten(x)
        x = self.ln1(x)

        # Fully connected layers
        for layer in self.linear:
            x = layer(x)
        
        # Layer norm and output layer
        x = self.ln2(x)
        out = self.output(x)
    
        return out
        

class DPU_CNN(nn.Module):
    def __init__(self, input_dim, filters, flatten_dim, dropout):
        super().__init__()
        self.filters = filters
        self.conv_layers = len(filters)
        self.embedding = L2ByteNorm(idx=13, dim=2)

        # add convolutional layers and maxpool
        conv_layers = []
        in_channels = input_dim
        for out_channels, kernel_size, pool in filters:
            # add Conv + ReLu + Dropout layer
            conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ))

            # add MaxPool layer
            if pool:
                conv_layers.append(nn.MaxPool1d(2))
            
            in_channels = out_channels # new in_channel is out channel
        self.conv = nn.ModuleList(conv_layers)

        if len(self.filters) == 5:
            # flatten and layer norm
            self.flatten = nn.Flatten()
            self.ln1 = nn.LayerNorm(flatten_dim)
        else:
            # flatten and layer norm 
            self.flatten = None
            self.ln1 = None

    def forward(self, x):
        x = self.embedding(x)

        # Convolutional layers
        for conv in self.conv:
            x = conv(x)
        
        if len(self.filters) == 5:
            # Flatten + LayerNorm
            x = self.flatten(x)
            x = self.ln1(x)
        
        return x