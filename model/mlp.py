import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils.normalization import L2ByteNorm

class MLP(nn.Module):
    """
    class for MLP model
    """
    def __init__(self, i_size, hidden_sizes, dropout):
        super().__init__()
        """
        Constructor for MLP model
        Creates linear layers
        Args:
            i_size (int): input size to first layer
            hidden_sizes (list): list containing layer sizes to each hidden linear layer
            dropout (float): dropout rate in rnn layers and linear layers
        """
        # define linear layers
        layers = []
        in_dim = i_size
        self.hidden_sizes = hidden_sizes
        self.embedding = L2ByteNorm(idx=13)

        for h in self.hidden_sizes:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_dim = h
        
        self.linear = nn.ModuleList(layers)
        self.ln1 = nn.LayerNorm(in_dim) # add layer norm
        self.output = nn.Linear(in_dim, 24)

    def forward(self, x):
        """
        Forward method to model
        Args:
            x (torch.tensor): input tensor
        Returns:
            out (torch.tensor): prediction output from linear layers
        """
        x = self.embedding(x)

        for layer in self.linear:
            x = layer(x)

        # LayerNorm + Output layer
        x = self.ln1(x)
        out = self.output(x)

        return out


class DPU_MLP(nn.Module):
    def __init__(self, i_size, hidden_sizes, dropout):
        super().__init__()
        """
        Constructor for MLP model
        Creates linear layers
        Args:
            i_size (int): input size to first layer
            hidden_sizes (list): list containing layer sizes to each hidden linear layer
            dropout (float): dropout rate in rnn layers and linear layers
        """
        # define linear layers
        layers = []
        in_dim = i_size
        self.hidden_sizes = hidden_sizes
        self.embedding = L2ByteNorm(idx=13)

        for h in self.hidden_sizes:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_dim = h
        
        self.linear = nn.ModuleList(layers)

    def forward(self, x):
        x = self.embedding(x)
        
        for layer in self.linear:
            x = layer(x)
        
        return x

       