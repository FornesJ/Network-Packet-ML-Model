import torch
import torch.nn as nn

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

    def forward(self, x, h0=None, c0=None, feat=False, classify=True):
        """
        Forward method to model
        Args:
            x (torch.tensor): input tensor
            h0 (torch.tensor): initial hidden state should be zero or None
            c0 (torch.tensor): initial cell state should be zero or None
            feat (boolean): if true return all states, else return last state as embedding
            classify (boolean): if true return logits for classified output, else return None
        Returns:
            embeddings | sequence (torch.tensor): last state | all states from rnn layers
            out (torch.tensor): prediction output from linear layers
        """
        if h0 is None or c0 is None:
            h0 = torch.zeros(2*self.n_layers, x.shape[0], self.h_size).to(self.device)
            c0 = torch.zeros(2*self.n_layers, x.shape[0], self.h_size).to(self.device)
        
        out = None
        sequence, (h0, c0) = self.rnn(x, (h0, c0))  # sequence: [B, T, 2*h_size]

        # take last layer's hidden state (both directions)
        # h0 shape: [num_layers*2, B, size]
        h_last = h0.view(self.n_layers, 2, x.shape[0], self.h_size)[-1]  # [2, B, h_size]
        x = torch.cat((h_last[0], h_last[1]), dim=1)  # [B, 2*h_size]

        embeddings = x.unsqueeze(1) # embedding of last hidden state

        # apply BN + FC
        if classify:
            x = self.bn1(x)         # [B, 2*h_size] → batch norm

            for layer in self.linear:
                x = layer(x)
            
            x = self.bn2(x)
            out = self.output(x)

        if feat:
            return [sequence], out
        else:
            return embeddings, out

        

    