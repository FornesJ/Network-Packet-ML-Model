import torch
import torch.nn as nn

class SplitModel(nn.Module):
    """
    class for SplitModel
    """
    def __init__(self, dpu_model, host_model):
        super().__init__()
        """
        Constructer for split model
        Parameters:
            dpu_model (nn.Module): model on dpu
            host_model (nn.Module): model on host
        """
        self.dpu_model = dpu_model
        self.host_model = host_model

    def forward(self, x, split="none"):
        """
        Forward method returns features and output from split model
        dpu model | host model | both dpu and host models
        Parameters:
            x (torch.Tensor): input tensor
            split (string): run inference on "dpu" -> dpu model | "host" -> host model | "none" -> both dpu and host models
        Returns:
            features (torch.tensor): features from hidden layers
            out (torch.tensor): prediction output from output layer
        """
        if split == "dpu":
            features, out = self.dpu_model(x, classify=False)
        elif split == "host":
            features, out = self.host_model(x)
        else:
            x, _ = self.dpu_model(x, classify=False)
            x.detach()
            features, out = self.host_model(x)
        return features, out

