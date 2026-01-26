import torch
import torch.nn as nn
import os
from config import Config
from loss_functions.loss import FocalLoss
from model.split_model import SplitModel
from train import Model
from model.mlp import MLP, DPU_MLP
from model.lstm import LSTM, DPU_LSTM
from model.gru import GRU, DPU_GRU
from model.cnn import CNN, DPU_CNN
conf = Config()

class MLP_Models:
    def __init__(self, ):
        self.type = "mlp"

        # full mlp model
        self.mlp_4 = {
            "name": "mlp_4",
            "hidden_sizes": [512, 256, 128, 64],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "large_model", "mlp_4.pth"),
            "split": False,
            "split_idx": 0
        }


        # light mlp model
        self.light_mlp_1 = {
            "name": "light_mlp_1",
            "distill_type": "relation",
            "hidden_sizes": [64],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "compressed_model", "light_mlp_1.pth"),
            "split": False,
            "split_idx": 0
        }

        self.result_light_mlp_1 = {
            "name": "result_light_mlp_1",
            "distill_type": "response",
            "hidden_sizes": [64],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "compressed_model", "result_light_mlp_1.pth"),
            "split": False,
            "split_idx": 0
        }

        self.light_mlp_4 = {
            "name": "light_mlp_4",
            "distill_type": "feature",
            "hidden_sizes": [256, 128, 64, 32],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "compressed_model", "light_mlp_4.pth"),
            "split": False,
            "split_idx": 0
        }



        # split mlp models
        self.split_mlp_3 = {
            "name": "split_mlp_3",
            "hidden_sizes": [512, 256, 128, 64],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_mlp_3.pth"),
            "split": True,
            "split_idx": 1
        }

        self.split_mlp_2 = {
            "name": "split_mlp_2",
            "hidden_sizes": [512, 256, 128, 64],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_mlp_2.pth"),
            "split": True,
            "split_idx": 2
        }

        self.split_mlp_1 = {
            "name": "split_mlp_1",
            "hidden_sizes": [512, 256, 128, 64],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_mlp_1.pth"),
            "split": True,
            "split_idx": 3
        }



    def get_model(self, model_conf: dict):
        """
        Method returns model based on selected config
        Params:
            model_conf (dict): dict with model hyper parameters
        Returns:
            (Model): Model object containing model based on parameters from config
        """
        if model_conf["split"]:
            idx = model_conf["split_idx"]
            dpu_sizes = model_conf["hidden_sizes"][:idx]
            host_sizes = model_conf["hidden_sizes"][idx:]
            model = SplitModel(
                dpu_model=DPU_MLP(
                    i_size=conf.mlp_input_size, 
                    hidden_sizes=dpu_sizes, 
                    dropout=model_conf["dropout"]),
                host_model=MLP(
                    i_size=dpu_sizes[-1], 
                    hidden_sizes=host_sizes, 
                    dropout=model_conf["dropout"]),
                split=conf.location
            ).to(conf.device)
        else:
            model = MLP(i_size=conf.mlp_input_size, 
                hidden_sizes=model_conf["hidden_sizes"], 
                dropout=model_conf["dropout"]).to(conf.device)
            
        return Model(model=model,
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=model_conf["checkpoint_path"],
            split_model=model_conf["split"])




class LSTM_Models:
    def __init__(self):
        self.type = "rnn"

        # full lstm model
        self.lstm_4 = {
            "name": "lstm_4",
            "i_size": conf.rnn_input_size,
            "h_size": conf.hidden_size,
            "n_layers": 4,
            "linear_sizes": [2*conf.hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "large_model", "lstm_4.pth"),
            "split": False,
            "split_idx": 0
        }


        # light lstm model
        self.light_lstm_1 = {
            "name": "light_lstm_1",
            "distill_type": "relation",
            "i_size": conf.rnn_input_size,
            "h_size": conf.light_hidden_size,
            "n_layers": 1,
            "linear_sizes": [2*conf.light_hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "compressed_model", "light_lstm_1.pth"),
            "split": False,
            "split_idx": 0
        }

        self.result_light_lstm_1 = {
            "name": "result_light_lstm_1",
            "distill_type": "response",
            "i_size": conf.rnn_input_size,
            "h_size": conf.light_hidden_size,
            "n_layers": 1,
            "linear_sizes": [2*conf.light_hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "compressed_model", "result_light_lstm_1.pth"),
            "split": False,
            "split_idx": 0
        }

        self.light_lstm_4 = {
            "name": "light_lstm_4",
            "distill_type": "feature",
            "i_size": conf.rnn_input_size,
            "h_size": conf.light_hidden_size,
            "n_layers": 4,
            "linear_sizes": [2*conf.light_hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "compressed_model", "light_lstm_4.pth"),
            "split": False,
            "split_idx": 0
        }



        # split lstm model
        self.split_lstm_3 = {
            "name": "split_lstm_3",
            "i_size": conf.rnn_input_size,
            "h_size": conf.hidden_size,
            "n_layers": 4,
            "linear_sizes": [2*conf.hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_lstm_3.pth"),
            "split": True,
            "split_idx": 1
        }

        self.split_lstm_2 = {
            "name": "split_lstm_2",
            "i_size": conf.rnn_input_size,
            "h_size": conf.hidden_size,
            "n_layers": 4,
            "linear_sizes": [2*conf.hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_lstm_2.pth"),
            "split": True,
            "split_idx": 2
        }

        self.split_lstm_1 = {
            "name": "split_lstm_1",
            "i_size": conf.rnn_input_size,
            "h_size": conf.hidden_size,
            "n_layers": 4,
            "linear_sizes": [2*conf.hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_lstm_1.pth"),
            "split": True,
            "split_idx": 3
        }

        self.split_lstm_0 = {
            "name": "split_lstm_0",
            "i_size": conf.rnn_input_size,
            "h_size": conf.hidden_size,
            "n_layers": 4,
            "linear_sizes": [2*conf.hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_lstm_0.pth"),
            "split": True,
            "split_idx": 4
        }



    def get_model(self, model_conf: dict):
        """
        Method returns model based on selected config
        Params:
            model_conf (dict): dict with model hyper parameters
        Returns:
            (Model): Model object containing model based on parameters from config
        """
        if model_conf["split"]:
            idx = model_conf["split_idx"]
            dpu_layers = idx
            host_layers = model_conf["n_layers"] - idx
            model = SplitModel(
                dpu_model=DPU_LSTM(
                    i_size=model_conf["i_size"],
                    h_size=model_conf["h_size"], 
                    n_layers=dpu_layers, 
                    dropout=model_conf["dropout"], 
                    device=conf.device),
                host_model=LSTM(
                    i_size=2*model_conf["h_size"], 
                    h_size=model_conf["h_size"], 
                    n_layers=host_layers, 
                    linear_sizes=model_conf["linear_sizes"], 
                    dropout=model_conf["dropout"], 
                    device=conf.device
                ) if host_layers > 0 else MLP(
                    i_size=2*model_conf["h_size"], 
                    hidden_sizes=model_conf["linear_sizes"], 
                    dropout=model_conf["dropout"]),
                split=conf.location
            ).to(conf.device)

        else:
            model = LSTM(i_size=model_conf["i_size"], 
                        h_size=model_conf["h_size"], 
                        n_layers=model_conf["n_layers"], 
                        linear_sizes=model_conf["linear_sizes"], 
                        dropout=model_conf["dropout"], 
                        device=conf.device).to(conf.device)
            
        return Model(model=model,
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=model_conf["checkpoint_path"],
            split_model=model_conf["split"])




class GRU_Models:
    def __init__(self):
        self.type = "rnn"

        # full gru model
        self.gru_4 = {
            "name": "gru_4",
            "i_size": conf.rnn_input_size,
            "h_size": conf.hidden_size,
            "n_layers": 4,
            "linear_sizes": [2*conf.hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "large_model", "gru_4.pth"),
            "split": False,
            "split_idx": 0
        }

        # light gru model
        self.light_gru_1 = {
            "name": "light_gru_1",
            "distill_type": "relation",
            "i_size": conf.rnn_input_size,
            "h_size": conf.light_hidden_size,
            "n_layers": 1,
            "linear_sizes": [2*conf.light_hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "compressed_model", "light_gru_1.pth"),
            "split": False,
            "split_idx": 0
        }

        self.result_light_gru_1 = {
            "name": "result_light_gru_1",
            "distill_type": "response",
            "i_size": conf.rnn_input_size,
            "h_size": conf.light_hidden_size,
            "n_layers": 1,
            "linear_sizes": [2*conf.light_hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "compressed_model", "result_light_gru_1.pth"),
            "split": False,
            "split_idx": 0
        }

        self.light_gru_4 = {
            "name": "light_gru_4",
            "distill_type": "feature",
            "i_size": conf.rnn_input_size,
            "h_size": conf.light_hidden_size,
            "n_layers": 4,
            "linear_sizes": [2*conf.light_hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "compressed_model", "light_gru_4.pth"),
            "split": False,
            "split_idx": 0
        }



        # split gru model
        self.split_gru_3 = {
            "name": "split_gru_3",
            "i_size": conf.rnn_input_size,
            "h_size": conf.hidden_size,
            "n_layers": 4,
            "linear_sizes": [2*conf.hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_gru_3.pth"),
            "split": True,
            "split_idx": 1
        }

        self.split_gru_2 = {
            "name": "split_gru_2",
            "i_size": conf.rnn_input_size,
            "h_size": conf.hidden_size,
            "n_layers": 4,
            "linear_sizes": [2*conf.hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_gru_2.pth"),
            "split": True,
            "split_idx": 2
        }

        self.split_gru_1 = {
            "name": "split_gru_1",
            "i_size": conf.rnn_input_size,
            "h_size": conf.hidden_size,
            "n_layers": 4,
            "linear_sizes": [2*conf.hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_gru_1.pth"),
            "split": True,
            "split_idx": 3
        }

        self.split_gru_0 = {
            "name": "split_gru_0",
            "i_size": conf.rnn_input_size,
            "h_size": conf.hidden_size,
            "n_layers": 4,
            "linear_sizes": [2*conf.hidden_size],
            "dropout": conf.dropout,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_gru_0.pth"),
            "split": True,
            "split_idx": 4
        }


    
    def get_model(self, model_conf: dict):
        """
        Method returns model based on selected config
        Params:
            model_conf (dict): dict with model hyper parameters
        Returns:
            (Model): Model object containing model based on parameters from config
        """
        if model_conf["split"]:
            idx = model_conf["split_idx"]
            dpu_layers = idx
            host_layers = model_conf["n_layers"] - idx
            model = SplitModel(
                dpu_model=DPU_GRU(
                    i_size=model_conf["i_size"],
                    h_size=model_conf["h_size"], 
                    n_layers=dpu_layers, 
                    dropout=model_conf["dropout"], 
                    device=conf.device),
                host_model=GRU(
                    i_size=2*model_conf["h_size"], 
                    h_size=model_conf["h_size"], 
                    n_layers=host_layers, 
                    linear_sizes=model_conf["linear_sizes"], 
                    dropout=model_conf["dropout"], 
                    device=conf.device
                ) if host_layers > 0 else MLP(
                    i_size=2*model_conf["h_size"], 
                    hidden_sizes=model_conf["linear_sizes"], 
                    dropout=model_conf["dropout"]),
                split=conf.location
            ).to(conf.device)

        else:
            model = GRU(i_size=model_conf["i_size"], 
                        h_size=model_conf["h_size"], 
                        n_layers=model_conf["n_layers"], 
                        linear_sizes=model_conf["linear_sizes"], 
                        dropout=model_conf["dropout"], 
                        device=conf.device).to(conf.device)
            
        return Model(model=model,
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=model_conf["checkpoint_path"],
            split_model=model_conf["split"])





class CNN_models:
    def __init__(self):
        self.type = "cnn"

        # full cnn model
        self.cnn_4 = {
            "name": "cnn_4",
            "i_size": 1, 
            "filters": [(32, 23, True), (64, 19, True), (128, 15, True), (192, 11, True), (256, 7, False)], 
            "linear_sizes": [512, 256], 
            "dropout": conf.dropout, 
            "flatten_size": 3328,
            "checkpoint_path": os.path.join(conf.checkpoint, "large_model", "cnn_4.pth"),
            "split": False,
            "split_idx": 0
        }



        # light cnn model
        self.light_cnn_1 = {
            "name": "light_cnn_1",
            "distill_type": "relation",
            "i_size": 1, 
            "filters": [(8, 7, True), (16, 5, False)], 
            "linear_sizes": [128, 64], 
            "dropout": conf.dropout, 
            "flatten_size": 3984,
            "checkpoint_path": os.path.join(conf.checkpoint, "compressed_model", "light_cnn_1.pth"),
            "split": False,
            "split_idx": 0
        }

        self.result_light_cnn_1 = {
            "name": "result_light_cnn_1",
            "distill_type": "response",
            "i_size": 1, 
            "filters": [(8, 7, True), (16, 5, False)], 
            "linear_sizes": [128, 64], 
            "dropout": conf.dropout, 
            "flatten_size": 3984,
            "checkpoint_path": os.path.join(conf.checkpoint, "compressed_model", "result_light_cnn_1.pth"),
            "split": False,
            "split_idx": 0
        }

        self.light_cnn_4 = {
            "name": "light_cnn_4",
            "distill_type": "feature",
            "i_size": 1, 
            "filters": [(16, 11, True), (32, 9, True), (64, 7, True), (128, 5, True), (192, 3, False)], 
            "linear_sizes": [512, 256], 
            "dropout": conf.dropout, 
            "flatten_size": 4608,
            "checkpoint_path": os.path.join(conf.checkpoint, "compressed_model", "light_cnn_4.pth"),
            "split": False,
            "split_idx": 0
        }



        # split cnn model
        self.split_cnn_3 = {
            "name": "split_cnn_3",
            "i_size": 1,
            "i_size_host": 64,
            "filters": [(32, 23, True), (64, 19, True), (128, 15, True), (192, 11, True), (256, 7, False)], 
            "linear_sizes": [512, 256], 
            "dropout": conf.dropout, 
            "flatten_size": 3328,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_cnn_3.pth"),
            "split": True,
            "split_idx": 2,
            "max_pool_last": True
        }

        self.split_cnn_2 = {
            "name": "split_cnn_2",
            "i_size": 1,
            "i_size_host": 128,
            "filters": [(32, 23, True), (64, 19, True), (128, 15, True), (192, 11, True), (256, 7, False)], 
            "linear_sizes": [512, 256], 
            "dropout": conf.dropout, 
            "flatten_size": 3328,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_cnn_2.pth"),
            "split": True,
            "split_idx": 3,
            "max_pool_last": True
        }

        self.split_cnn_1 = {
            "name": "split_cnn_1",
            "i_size": 1,
            "i_size_host": 192,
            "filters": [(32, 23, True), (64, 19, True), (128, 15, True), (192, 11, True), (256, 7, False)], 
            "linear_sizes": [512, 256], 
            "dropout": conf.dropout, 
            "flatten_size": 3328,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_cnn_1.pth"),
            "split": True,
            "split_idx": 4,
            "max_pool_last": True
        }

        self.split_cnn_0 = {
            "name": "split_cnn_0",
            "i_size": 1,
            "i_size_host": 64,
            "filters": [(32, 23, True), (64, 19, True), (128, 15, True), (192, 11, True), (256, 7, False)], 
            "linear_sizes": [512, 256], 
            "dropout": conf.dropout, 
            "flatten_size": 3328,
            "checkpoint_path": os.path.join(conf.checkpoint, "split_model", "split_cnn_0.pth"),
            "split": True,
            "split_idx": 5,
            "max_pool_last": False
        }


    
    def get_model(self, model_conf: dict):
        """
        Method returns model based on selected config
        Params:
            model_conf (dict): dict with model hyper parameters
        Returns:
            (Model): Model object containing model based on parameters from config
        """
        if model_conf["split"]:
            idx = model_conf["split_idx"]
            model = SplitModel(
                dpu_model=DPU_CNN(
                    input_dim=model_conf["i_size"],
                    filters=model_conf["filters"][:idx],
                    flatten_dim=model_conf["flatten_size"],
                    dropout=model_conf["dropout"]
                ),
                host_model=CNN(
                    input_dim=model_conf["i_size_host"],
                    filters=model_conf["filters"][idx:],
                    linear_sizes=model_conf["linear_sizes"],
                    flatten_dim=model_conf["flatten_size"],
                    dropout=model_conf["dropout"]
                ) if idx < len(model_conf["filters"]) else MLP(
                    i_size=model_conf["flatten_size"], 
                    hidden_sizes=model_conf["linear_sizes"], 
                    dropout=model_conf["dropout"]
                ),
                split=conf.location
            ).to(conf.device)
        else:
            model = CNN(input_dim=model_conf["i_size"],
                        filters=model_conf["filters"],
                        linear_sizes=model_conf["linear_sizes"],
                        flatten_dim=model_conf["flatten_size"],
                        dropout=model_conf["dropout"]).to(conf.device)
            
        return Model(model=model,
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=model_conf["checkpoint_path"],
            split_model=model_conf["split"])