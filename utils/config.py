import os
import torch

class Config:
    def __init__(self):
        # file paths
        self.root_folder = os.path.join(os.getcwd())

        # system settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model architecture
        self.mlp_input_size = 513
        self.rnn_input_size = 1
        self.hidden_size = 64
        self.light_hidden_size = 32

        # Optimization
        self.learning_rate = 1e-4
        self.weight_decay = 1e-2
        self.gamma = 0.9
        self.epochs = 10
        self.batch_size = 512
        self.dropout = 0.15