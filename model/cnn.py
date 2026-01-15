import torch
import torch.nn as nn
import torch.ao.quantization as quant

class CNN(nn.Module):
    def __init__(self, i_size, filters, linear_sizes, flatten_size, dropout, max_pool_last=False):
        super().__init__()
        self.filters = filters
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

        self.conv = nn.ModuleList(conv_layers) # 

        # flatten and batch norm 
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm1d(flatten_size)

        # add fully connected layers
        layers = []
        in_dim = flatten_size
        for h in linear_sizes:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_dim = h
        self.linear = nn.ModuleList(layers)

        # batch norm and output layer
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.output = nn.Linear(in_dim, 24)
    
    def forward(self, x):
        # Convolutional layers
        for conv in self.conv:
            x = conv(x)
        
        # Flatten + BatchNorm
        x = self.flatten(x)
        x = self.bn1(x)

        # Fully connected layers
        for layer in self.linear:
            x = layer(x)
        
        # BN + Output layer
        x = self.bn2(x)
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

        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        # Convolutional layers
        for conv in self.conv:
            x = conv(x)
        
        # Flatten + BatchNorm
        x = self.flatten(x)
        x = self.dequant(x)
        x = self.bn1(x)
        x = self.quant(x)

        # Fully connected layers
        for layer in self.linear:
            x = layer(x)
        
        # BN + Output layer
        x = self.dequant(x)
        x = self.bn2(x)
        x = self.quant(x)
        out = self.output(x)
        out = self.dequant(x)
    
        return out