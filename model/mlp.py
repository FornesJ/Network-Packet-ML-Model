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

        for h in self.hidden_sizes:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_dim = h
        
        self.linear = nn.ModuleList(layers)
        self.bn = nn.BatchNorm1d(in_dim) # add batch norm
        self.output = nn.Linear(in_dim, 24)

    def forward(self, x):
        """
        Forward method to model
        Args:
            x (torch.tensor): input tensor
        Returns:
            out (torch.tensor): prediction output from linear layers
        """
        for layer in self.linear:
            x = layer(x)

        # BatchNorm + Output layer
        x = self.bn(x)
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

        for h in self.hidden_sizes:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_dim = h
        
        self.linear = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.linear:
            x = layer(x)
        
        return x
    

class QuantMLP(nn.Module):
    def __init__(self, i_size, hidden_sizes, dropout):
        super().__init__()
        self.ln1 = L2ByteNorm(idx=13)
        #self.quant = quant.QuantStub()
        # define linear layers
        layers = []
        in_dim = i_size
        self.hidden_sizes = hidden_sizes

        for h in self.hidden_sizes:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_dim = h
        
        self.linear = nn.ModuleList(layers)
        #self.bn = nn.BatchNorm1d(in_dim)
        self.output = nn.Linear(in_dim, 24)

    def forward(self, x):
        x = self.ln1(x)
        #x = self.quant(x)
        for layer in self.linear:
            x = layer(x)
        #x = self.dequant(x)   # dequant BEFORE logits
        x = self.output(x)
        return x
    

class ByteNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_head = F.normalize(x[:,:13]) * 127
        x_payload = F.normalize(x[:,13:]) * 127 # prep for int8 conversion
        x = torch.cat((x_head, x_payload), dim=1)
        return x

       