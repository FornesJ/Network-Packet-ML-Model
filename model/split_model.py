import torch
import torch.nn as nn

class SplitModel(nn.Module):
    """
    class for SplitModel
    """
    def __init__(self, dpu_model, host_model, split="none"):
        super().__init__()
        """
        Constructer for split model
        Parameters:
            dpu_model (nn.Module): model on dpu
            host_model (nn.Module): model on host
            split (string): run inference on "dpu" -> dpu model | "host" -> host model | "none" -> both dpu and host models
        """
        if split == "dpu":
            self.dpu_model = dpu_model
            self.host_model = None
            del host_model
        elif split == "host":
            self.dpu_model = None
            self.host_model = host_model
            self.host_model.embedding = nn.Identity()
            del dpu_model
        else:
            self.dpu_model = dpu_model
            self.host_model = host_model
            self.host_model.embedding = nn.Identity()
        self.split = split

    def forward(self, x):
        """
        Forward method returns features and output from split model
        dpu model | host model | both dpu and host models
        Parameters:
            x (torch.Tensor): input tensor
        Returns:
            features (torch.tensor): features from hidden layers
            out (torch.tensor): prediction output from output layer
        """
        if self.split == "dpu":
            return self.dpu_model(x)
        elif self.split == "host":
            return self.host_model(x)
        else:
            x = self.dpu_model(x)
            return self.host_model(x)

