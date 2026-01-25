import torch
import torch.nn as nn
import torch.nn.functional as F

class L2ByteNorm(nn.Module):
    def __init__(self,  idx: int, gamma: float=255.0, dim: int=1):
        super().__init__()
        self.idx = idx
        self.gamma = gamma
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # split data into [B,idx] and [B, L-idx]
        x_head, x_data = x.split(
            [self.idx, x.shape[self.dim] - self.idx], 
            dim=self.dim)

        # Normalize and Scale head and payload 
        x_head = F.normalize(x_head, dim=self.dim)
        x_data = F.normalize(x_data, dim=self.dim)
        x = torch.cat((x_head, x_data), dim=self.dim)
        x = x * self.gamma
        return x



    