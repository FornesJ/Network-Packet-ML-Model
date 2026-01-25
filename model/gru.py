import torch
import torch.nn as nn
from model.model_utils.hidden_state import LastHidden
from model.model_utils.normalization import L2ByteNorm

class GRU(nn.Module):
    """
    class for GRU model
    """
    def __init__(self, i_size, h_size, n_layers, linear_sizes, dropout, device):
        super().__init__()
        """
        Constructor for GRU model
        Creates gru layers and linear layers
        Args:
            i_size (int): input size to first layer
            h_size (int): size in hidden layers
            linear_sizes (list): list containing layer sizes to each hidden linear layer
            dropout (float): dropout rate in rnn layers and linear layers
            device (string): device location of model
        """
        self.h_size = h_size
        self.n_layers = n_layers
        self.device = device
        self.embedding = L2ByteNorm(idx=13)

        rnn_dropout = dropout if n_layers > 1 else 0.0 # can not have dropout with only one rnn layer

        # define gru layers
        self.rnn = nn.GRU(
            input_size=i_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=True
        )

        self.lh = LastHidden(self.rnn.num_layers, self.rnn.hidden_size) # [B, 2*size]
        self.ln1 = nn.LayerNorm(2*self.h_size)

        # define linear layers
        layers = []
        in_dim = 2 * h_size

        for h in linear_sizes:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_dim = h
        
        self.linear = nn.ModuleList(layers)
        self.ln2 = nn.LayerNorm(in_dim)
        self.output = nn.Linear(in_dim, 24)

    def forward(self, x):
        """
        Forward method to model
        Args:
            x (torch.tensor): input tensor
        Returns:
            embeddings | hidden_states (torch.tensor): last hidden state | hidden states from each gru layers
            out (torch.tensor): prediction output from linear layers
        """
        x = self.embedding(x)
        
        h0 = torch.zeros(2*self.n_layers, x.shape[0], self.h_size).to(self.device)
               
        _, h0 = self.rnn(x, h0)  # sequence: [B, T, 2*h_size]

        # h0 shape: [num_layers*2, B, size]
        h_last = self.lh(h0) # [B, 2*size]

        # apply BN + FC
        x = self.ln1(h_last)         # [B, 2*h_size] → batch norm
        for layer in self.linear:
            x = layer(x)
        
        # apply BN + Output
        x = self.ln2(x)
        out = self.output(x)

        return out
    

class DPU_GRU(nn.Module):
    def __init__(self, i_size, h_size, n_layers, dropout, device):
        super().__init__()
        """
        Constructor for GRU model
        Creates gru layers and linear layers
        Args:
            i_size (int): input size to first layer
            h_size (int): size in hidden layers
            dropout (float): dropout rate in rnn layers and linear layers
            device (string): device location of model
        """

        self.h_size = h_size
        self.n_layers = n_layers
        self.device = device

        rnn_dropout = dropout if n_layers > 1 else 0.0 # can not have dropout with only one rnn layer

        # define gru layers
        self.rnn = nn.GRU(
            input_size=i_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=True
        )

        if n_layers == 4:
            self.bn1 = nn.BatchNorm1d(2 * h_size) # add batch norm
        else:
            self.bn1 = None

    def forward(self, x):
        h0 = torch.zeros(2*self.n_layers, x.shape[0], self.h_size).to(self.device)
          
        output, h0 = self.rnn(x, h0)  # sequence: [B, T, 2*h_size]

        if self.n_layers == 4:
            # h0 shape: [num_layers*2, B, size]
            h_last = h0.view(self.n_layers, 2, x.shape[0], self.h_size)[-1] # [2, B, h_size]
            h_last = torch.cat((h_last[0], h_last[1]), dim=1)               # [B, 2*h_size]
            output = self.bn1(h_last)                                       # [B, 2*h_size] → batch norm

        return output


class GRU_QP(GRU):
    def __init__(self, i_size, h_size, n_layers, linear_sizes, dropout, device):
        super().__init__(i_size, h_size, n_layers, linear_sizes, dropout, device)
        self.bn1 = nn.LayerNorm(2 * h_size)
        self.bn2 = nn.LayerNorm(linear_sizes[-1])