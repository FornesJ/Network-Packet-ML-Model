import torch
import torch.nn as nn
import torch.ao.quantization as quant

class LSTM(nn.Module):
    """
    class for LSTM model
    """
    def __init__(self, i_size, h_size, n_layers, linear_sizes, dropout, device):
        super().__init__()
        """
        Constructor for LSTM model
        Creates lstm layers and linear layers
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

        rnn_dropout = dropout if n_layers > 1 else 0.0 # can not have dropout with only one rnn layer

        # define lstm layers
        self.rnn = nn.LSTM(
            input_size=i_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=True
        )

        self.bn1 = nn.BatchNorm1d(2 * h_size) # add batch norm

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
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.output = nn.Linear(in_dim, 24)

    def forward(self, x):
        """
        Forward method to model
        Args:
            x (torch.tensor): input tensor
        Returns:
            out (torch.tensor): prediction output from linear layers
        """
        h0 = torch.zeros(2*self.n_layers, x.shape[0], self.h_size).to(self.device)
        c0 = torch.zeros(2*self.n_layers, x.shape[0], self.h_size).to(self.device)
        
        out = None         
        _, (h0, c0) = self.rnn(x, (h0, c0))    # output: [B, T, 2*h_size]

        # h0 shape: [num_layers*2, B, size]
        h_last = h0.view(self.n_layers, 2, x.shape[0], self.h_size)[-1] # [2, B, h_size]
        h_last = torch.cat((h_last[0], h_last[1]), dim=1)  # [B, 2*size]

        # apply BN + FC
        x = self.bn1(h_last)         # [B, 2*h_size] → batch norm
        for layer in self.linear:
            x = layer(x)
        
        # apply BN + Output
        x = self.bn2(x)
        out = self.output(x)

        return out
    

class DPU_LSTM(nn.Module):
    def __init__(self, i_size, h_size, n_layers, dropout, device):
        super().__init__()
        """
        Constructor for LSTM model
        Creates lstm layers and linear layers
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

        # define lstm layers
        self.rnn = nn.LSTM(
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
        """
        Forward method to model
        Args:
            x (torch.tensor): input tensor
        Returns:
            out (torch.tensor): prediction output from linear layers
        """
        h0 = torch.zeros(2*self.n_layers, x.shape[0], self.h_size).to(self.device)
        c0 = torch.zeros(2*self.n_layers, x.shape[0], self.h_size).to(self.device)
                
        output, (h0, c0) = self.rnn(x, (h0, c0))    # output: [B, T, 2*h_size]

        if self.n_layers == 4:
            # h0 shape: [num_layers*2, B, size]
            h_last = h0.view(self.n_layers, 2, x.shape[0], self.h_size)[-1] # [2, B, h_size]
            h_last = torch.cat((h_last[0], h_last[1]), dim=1)               # [B, 2*h_size]
            output = self.bn1(h_last)                                       # [B, 2*h_size] → batch norm

        return output



class LSTM_QP(LSTM):
    def __init__(self, i_size, h_size, n_layers, linear_sizes, dropout, device):
        super().__init__(i_size, h_size, n_layers, linear_sizes, dropout, device)
        self.bn1 = nn.LayerNorm(2 * h_size)
        self.bn2 = nn.LayerNorm(linear_sizes[-1])