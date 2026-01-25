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
    def __init__(self, i_size, filters, flatten_size, dropout, max_pool_last=True):
        super().__init__()
        self.max_pool_last = max_pool_last
        self.conv_layers = len(filters)

        # add convolutional layers and maxpool
        conv_layers = []
        in_channels = i_size
        for out_channels, kernel_size in filters:
            # add MaxPool layer
            if len(conv_layers) != 0:
                conv_layers.append(nn.MaxPool1d(2))
            
            # add Conv + ReLu + Dropout layer
            conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ))

            in_channels = out_channels # new in_channel is out channel
        
        if self.max_pool_last:
            conv_layers.append(nn.MaxPool1d(2))

        self.conv = nn.ModuleList(conv_layers) # conv layers

        if self.max_pool_last == False:
            # flatten and batch norm 
            self.flatten = nn.Flatten()
            self.bn1 = nn.BatchNorm1d(flatten_size)
        else:
            # flatten and batch norm 
            self.flatten = None
            self.bn1 = None

    def forward(self, x):
        # Convolutional layers
        for conv in self.conv:
            x = conv(x)
        
        if not self.max_pool_last:
            # Flatten + BatchNorm
            x = self.flatten(x)
            x = self.bn1(x)
        
        return x


class CNN_QP(CNN):
    def __init__(self, i_size, filters, linear_sizes, flatten_size, dropout, max_pool_last=False):
        super().__init__(i_size, filters, linear_sizes, flatten_size, dropout, max_pool_last)
        self.bn1 = nn.LayerNorm(flatten_size)
        self.bn2 = nn.LayerNorm(linear_sizes[-1])