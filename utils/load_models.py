import torch
import torch.nn as nn
from config import Config
from ..loss_functions.loss import FocalLoss
from ..models.model import Model
from ..models.mlp import MLP
from ..models.lstm import LSTM
from ..models.gru import GRU
conf = Config()

class MLP_Models:
    def __init__(self):

        # full mlp model
        self.mlp_4 = Model(
            model=MLP(i_size=conf.mlp_input_size, hidden_sizes=[384, 256, 128, 64], dropout=conf.dropout),
            loss_function=FocalLoss(),
            conf=conf
        )

        # light mlp models
        self.light_mlp_3 = Model(
            model=MLP(i_size=conf.mlp_input_size, hidden_sizes=[384, 256, 128], dropout=conf.dropout),
            loss_function=FocalLoss(),
            conf=conf
        )
        self.light_mlp_2 = Model(
            model=MLP(i_size=conf.mlp_input_size, hidden_sizes=[384, 256], dropout=conf.dropout),
            loss_function=FocalLoss(),
            conf=conf
        )
        self.light_mlp_1 = Model(
            model=MLP(i_size=conf.mlp_input_size, hidden_sizes=[384], dropout=conf.dropout),
            loss_function=FocalLoss(),
            conf=conf
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
                       device=conf.device),
            loss_function=FocalLoss(),
            conf=conf
        )

        # smaller lstm models 
        self.light_lstm_3 = Model(
            model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=3, 
                       linear_sizes=[2*conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device),
            oss_function=FocalLoss(),
            conf=conf
        )
        self.light_lstm_2 = Model(
            model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=2, 
                       linear_sizes=[2*conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device),
            oss_function=FocalLoss(),
            conf=conf
        )
        self.light_lstm_1 = Model(
            model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=1, 
                       linear_sizes=[2*conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device),
            oss_function=FocalLoss(),
            conf=conf
        )


class GRU_Models:
    def __init__(self):
        self.gru_4 = Model(
            model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=4, 
                       linear_sizes=[2*conf.hidden_size, conf.hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device),
            loss_function=FocalLoss(),
            conf=conf
        )

        # smaller lstm models 
        self.light_gru_3 = Model(
            model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=3, 
                       linear_sizes=[2*conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device),
            oss_function=FocalLoss(),
            conf=conf
        )
        self.light_gru_2 = Model(
            model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=2, 
                       linear_sizes=[2*conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device),
            oss_function=FocalLoss(),
            conf=conf
        )
        self.light_gru_1 = Model(
            model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=1, 
                       linear_sizes=[2*conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device),
            oss_function=FocalLoss(),
            conf=conf
        )