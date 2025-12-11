import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class Prune_Model(nn.Module):
    def __init__(self, rnn=False, amount=0.2):
        super().__init__()
        self.rnn = rnn
        self.amount = amount

    def forward(self, model):
        parameters = []

        if self.rnn:
            parameters.append((model.rnn, "weight"))

        for layer in model.linear:
            parameters.append((layer, "weight"))
        parameters.append((model.output, "weight"))

        prune.global_unstructured(
            parameters=parameters,
            pruning_method=prune.L1Unstructured,
            amount=self.amount
        )




