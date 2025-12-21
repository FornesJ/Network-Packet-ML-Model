import os
import torch

class Config:
    def __init__(self):
        # file paths
        self.root_folder = os.path.join(os.path.abspath(__file__).replace("config.py", ""))
        self.checkpoint = os.path.join(self.root_folder, "checkpoint")
        self.benchmark_host = os.path.join(self.root_folder, "benchmark", "host")
        self.benchmark_dpu = os.path.join(self.root_folder, "benchmark", "dpu")
        self.transfer = os.path.join(self.root_folder, "transfer")
        self.compact = os.path.join(self.root_folder, "compact")
        self.loss_functions = os.path.join(self.root_folder, "loss_functions")
        self.model = os.path.join(self.root_folder, "model")
        self.datasets = os.path.join(self.root_folder, "data")
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