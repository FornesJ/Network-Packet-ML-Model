import os
import torch

class Config:
    def __init__(self):
        # file paths
        self.root_folder = os.path.join(os.path.abspath(__file__).replace("utils/config.py", ""))
        self.large_models_path = os.path.join(self.root_folder, "large_models")
        self.distilled_models_path = os.path.join(self.root_folder, "distilled_models")
        self.split_models_path = os.path.join(self.root_folder, "split_models")
        self.socket_transfer = os.path.join(self.root_folder, "socket_transfer")
        self.loss_functions = os.path.join(self.root_folder, "loss_functions")
        self.models = os.path.join(self.root_folder, "models")
        self.datasets = os.path.join(self.root_folder, "datasets")
        self.utils = os.path.join(self.root_folder, "utils")

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
        self.dropout = 0.20
        self.epochs = 10
        self.batch_size = 512