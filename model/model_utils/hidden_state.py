import torch
import torch.nn as nn

class LastHidden(nn.Module):
    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # h0 shape: [num_layers*2, B, h_size]
        h_last = x.view(self.num_layers, 2, x.shape[1], self.hidden_dim)[-1] # [2, B, h_size]
        return torch.cat((h_last[0], h_last[1]), dim=1)  # [B, 2*h_size]
    
class AllHidden(LastHidden):
    def __init__(self, num_layers, hidden_dim):
        super().__init__(num_layers, hidden_dim)

    def forward(self, x):
        # h0 shape: [num_layers*2, B, h_size]
        h_last = x.view(self.num_layers, 2, x.shape[1], self.hidden_dim) # [N, 2, B, h_size]
        return torch.cat((h_last[:,0], h_last[:,1]), dim=2) # [N, B, 2*h_size]
