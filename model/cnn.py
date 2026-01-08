import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, i_size, filters, linear_sizes, flatten_size, dropout):
        super().__init__()

        # add convolutional layers and maxpool
        conv_layers = []
        in_channels = i_size
        for out_channels, kernel_size in filters:
            if len(conv_layers) != 0:
                conv_layers.append(nn.MaxPool1d(2))
            conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels
        self.conv = nn.ModuleList(conv_layers)

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
    
    def forward(self, x, feat=False, classify=True):
        feat_map = []
        for layer in self.conv:
            x = layer(x)
            if isinstance(layer, nn.Sequential):
                feat_map.append(x)

        last_feat = x

        if classify:
            x = self.flatten(x)
            x = self.bn1(x)

            for layer in self.linear:
                x = layer(x)
            
            x = self.bn2(x)
            out = self.output(x)
        
        if feat:
            return feat_map, out
        else:
            return last_feat, out
        
        


