import torch
import torch.nn as nn
import os
from utils.config import Config
from loss_functions.loss import FocalLoss
from models.model import Model
from models.mlp import MLP
from models.lstm import LSTM
from models.gru import GRU
conf = Config()

class MLP_Models:
    def __init__(self, ):

        # full mlp model
        self.mlp_4 = Model(
            model=MLP(i_size=conf.mlp_input_size, 
                      hidden_sizes=[512, 256, 128, 64], 
                      dropout=conf.dropout).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.large_models_path, "checkpoint", "mlp_4.pth")
        )

        # light mlp models
        self.light_mlp_3 = Model(
            model=MLP(i_size=conf.mlp_input_size, hidden_sizes=[256, 128, 64], dropout=conf.dropout).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.distilled_models_path, "checkpoint", "mlp_3.pth")
        )
        self.light_mlp_2 = Model(
            model=MLP(i_size=conf.mlp_input_size, hidden_sizes=[128, 64], dropout=conf.dropout).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.distilled_models_path, "checkpoint", "mlp_2.pth")
        )
        self.light_mlp_1 = Model(
            model=MLP(i_size=conf.mlp_input_size, hidden_sizes=[64], dropout=conf.dropout).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.distilled_models_path, "checkpoint", "mlp_1.pth")
        )


class LSTM_Models:
    def __init__(self):
        # full lstm model
        self.lstm_4 = Model(
            model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=4, 
                       linear_sizes=[2*conf.hidden_size, conf.hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.large_models_path, "checkpoint", "lstm_4.pth")
        )

        # smaller lstm models 
        self.light_lstm_3 = Model(
            model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=3, 
                       linear_sizes=[2*conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.distilled_models_path, "checkpoint", "lstm_3.pth")
        )
        self.light_lstm_2 = Model(
            model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=2, 
                       linear_sizes=[2*conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.distilled_models_path, "checkpoint", "lstm_2.pth")
        )
        self.light_lstm_1 = Model(
            model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=1, 
                       linear_sizes=[2*conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.distilled_models_path, "checkpoint", "lstm_1.pth")
        )


class GRU_Models:
    def __init__(self):
        self.gru_4 = Model(
            model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=4, 
                       linear_sizes=[2*conf.hidden_size, conf.hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.large_models_path, "checkpoint", "gru_4.pth")
        )

        # smaller lstm models 
        self.light_gru_3 = Model(
            model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=3, 
                       linear_sizes=[2*conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.distilled_models_path, "checkpoint", "gru_3.pth")
        )
        self.light_gru_2 = Model(
            model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=2, 
                       linear_sizes=[2*conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.distilled_models_path, "checkpoint", "gru_2.pth")
        )
        self.light_gru_1 = Model(
            model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=1, 
                       linear_sizes=[2*conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.distilled_models_path, "checkpoint", "gru_1.pth")
        )