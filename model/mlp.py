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

    def forward(self, x, feat=False, classify=True):
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
        features = []
        out = None

        for layer in self.linear:
            x = layer(x)
            features.append(x)
        
        if not feat:
            features = features[-1]     # final hidden representation

        if classify:
            x = self.bn(x)              # batchnorm before output layer
            out = self.output(x)        # model prediction logits

        return features, out



       