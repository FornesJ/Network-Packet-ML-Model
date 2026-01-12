import torch.nn as nn

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
            feat (boolean): if true return all features, else return last feature
            classify (boolean): if true return logits for classified output, else return None
        Returns:
            features (torch.tensor): features from rnn layers
            out (torch.tensor): prediction output from linear layers
        """
        for layer in self.linear:
            x = layer(x)
        features = x

        # BatchNorm + Output layer
        x = self.bn(x)
        out = self.output(x)

        return features, out
    
    def feature_map(self, x):
        features = []
        out = None

        for layer in self.linear:
            x = layer(x)
            features.append(x)

        # BatchNorm + Output layer
        x = self.bn(x)
        out = self.output(x)

        return features, out


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
        
        return x, None
       