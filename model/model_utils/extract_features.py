import torch
import torch.nn as nn
from config import Config
conf = Config()


class FeatureExtractor:
    def __init__(self, model, model_type):
        self.features = {}
        self.handles = []
        self.model_type = model_type

        if model_type == "mlp":
            self.module = nn.ReLU # extract features after activation
        elif model_type == "rnn":
            self.module = type(model.rnn)
        elif model_type == "cnn":
            self.module = nn.LeakyReLU # extract features after activation
        else:
            raise ValueError(f"model_type must be 'mlp', 'rnn' or 'cnn' but got '{model_type}'!")

        for name, layer in model.named_modules():
            if isinstance(layer, self.module) and name != "output":
                handle = layer.register_forward_hook(self.save_output(name))
                self.handles.append(handle)

    def save_output(self, name):
        def hook(module, input, output):
            self.features[name] = output

        return hook

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
    
    def clear(self):
        self.features = {}
