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

        for h in hidden_sizes:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_dim = h
        
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(in_dim, 24)

    def forward(self, x):
        """
        Forward method to model
        Args:
            x (torch.tensor): input tensor
        Returns:
            features (torch.tensor): features from rnn layers
            out (torch.tensor): prediction output from linear layers
        """
        for layer in self.layers:
            x = layer(x)
        
        features = x              # final hidden representation
        out = self.output(x)      # model prediction logits

        return features, out



       