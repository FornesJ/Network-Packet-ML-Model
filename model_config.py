import torch
import torch.nn as nn
import os
from config import Config
from loss_functions.loss import FocalLoss
from model.split_model import SplitModel
from train import Model
from model.mlp import MLP
from model.lstm import LSTM
from model.gru import GRU
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
            checkpoint_path=os.path.join(conf.checkpoint, "large_model", "mlp_4.pth")
        )

        # light mlp model
        self.light_mlp_1 = Model(
            model=MLP(i_size=conf.mlp_input_size, hidden_sizes=[64], dropout=conf.dropout).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "compressed_model", "light_mlp_1.pth")
        )

        self.light_mlp_4 = Model(
            model=MLP(i_size=conf.mlp_input_size, hidden_sizes=[256, 128, 64, 32], dropout=conf.dropout).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "compressed_model", "light_mlp_4.pth")
        )

        # split mlp models
        self.split_mlp_3 = Model(
            model=SplitModel(
                dpu_model=MLP(i_size=conf.mlp_input_size, hidden_sizes=[512], dropout=conf.dropout),
                host_model=MLP(i_size=512, hidden_sizes=[256, 128, 64], dropout=conf.dropout)
            ).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "split_model", "split_mlp_3.pth"),
            split_model=True
        )

        self.split_mlp_2 = Model(
            model=SplitModel(
                dpu_model=MLP(i_size=conf.mlp_input_size, hidden_sizes=[512, 256], dropout=conf.dropout),
                host_model=MLP(i_size=256, hidden_sizes=[128, 64], dropout=conf.dropout)
            ).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "split_model", "split_mlp_2.pth"),
            split_model=True
        )

        self.split_mlp_1 = Model(
            model=SplitModel(
                dpu_model=MLP(i_size=conf.mlp_input_size, hidden_sizes=[512, 256, 128], dropout=conf.dropout),
                host_model=MLP(i_size=128, hidden_sizes=[64], dropout=conf.dropout)
            ).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "split_model", "split_mlp_1.pth"),
            split_model=True
        )


class LSTM_Models:
    def __init__(self):
        # full lstm model
        self.lstm_4 = Model(
            model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=4, 
                       linear_sizes=[conf.hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "large_model", "lstm_4.pth")
        )

        # light lstm model
        self.light_lstm_1 = Model(
            model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=1, 
                       linear_sizes=[conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "compressed_model", "light_lstm_1.pth")
        )

        self.light_lstm_4 = Model(
            model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=4, 
                       linear_sizes=[conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "compressed_model", "light_lstm_4.pth")
        )

        # split lstm model
        self.split_lstm_3 = Model(
            model=SplitModel(
                dpu_model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=1, 
                       linear_sizes=[], 
                       dropout=conf.dropout, 
                       device=conf.device),
                host_model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=3, 
                       linear_sizes=[conf.hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device)
            ).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "split_model", "split_lstm_3.pth"),
            split_model=True
        )

        self.split_lstm_2 = Model(
            model=SplitModel(
                dpu_model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=2, 
                       linear_sizes=[], 
                       dropout=conf.dropout, 
                       device=conf.device),
                host_model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=2, 
                       linear_sizes=[conf.hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device)
            ).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "split_model", "split_lstm_2.pth"),
            split_model=True
        )

        self.split_lstm_1 = Model(
            model=SplitModel(
                dpu_model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=3, 
                       linear_sizes=[], 
                       dropout=conf.dropout, 
                       device=conf.device),
                host_model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=1, 
                       linear_sizes=[conf.hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device)
            ).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "split_model", "split_lstm_1.pth"),
            split_model=True
        )

        self.split_lstm_0 = Model(
            model=SplitModel(
                dpu_model=LSTM(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=4, 
                       linear_sizes=[], 
                       dropout=conf.dropout, 
                       device=conf.device),
                host_model=MLP(i_size=2*conf.hidden_size, hidden_sizes=[conf.hidden_size], dropout=conf.dropout)
            ).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "split_model", "split_lstm_0.pth"),
            split_model=True
        )




class GRU_Models:
    def __init__(self):
        self.gru_4 = Model(
            model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=4, 
                       linear_sizes=[conf.hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "large_model", "gru_4.pth")
        )

        # light gru model
        self.light_gru_1 = Model(
            model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=1, 
                       linear_sizes=[conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "compressed_model", "light_gru_1.pth")
        )

        self.light_gru_4 = Model(
            model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.light_hidden_size, 
                       n_layers=4, 
                       linear_sizes=[conf.light_hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "compressed_model", "light_gru_4.pth")
        )

        # split lstm model
        self.split_gru_3 = Model(
            model=SplitModel(
                dpu_model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=1, 
                       linear_sizes=[], 
                       dropout=conf.dropout, 
                       device=conf.device),
                host_model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=3, 
                       linear_sizes=[conf.hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device)
            ).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "split_model", "split_gru_3.pth"),
            split_model=True
        )

        self.split_gru_2 = Model(
            model=SplitModel(
                dpu_model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=2, 
                       linear_sizes=[], 
                       dropout=conf.dropout, 
                       device=conf.device),
                host_model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=2, 
                       linear_sizes=[conf.hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device)
            ).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "split_model", "split_gru_2.pth"),
            split_model=True
        )

        self.split_gru_1 = Model(
            model=SplitModel(
                dpu_model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=3, 
                       linear_sizes=[], 
                       dropout=conf.dropout, 
                       device=conf.device),
                host_model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=1, 
                       linear_sizes=[conf.hidden_size], 
                       dropout=conf.dropout, 
                       device=conf.device)
            ).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "split_model", "split_gru_1.pth"),
            split_model=True
        )

        self.split_gru_0 = Model(
            model=SplitModel(
                dpu_model=GRU(i_size=conf.rnn_input_size, 
                       h_size=conf.hidden_size, 
                       n_layers=4, 
                       linear_sizes=[], 
                       dropout=conf.dropout, 
                       device=conf.device),
                host_model=MLP(i_size=2*conf.hidden_size, hidden_sizes=[conf.hidden_size], dropout=conf.dropout)
            ).to(conf.device),
            loss_function=FocalLoss(),
            conf=conf,
            checkpoint_path=os.path.join(conf.checkpoint, "split_model", "split_gru_0.pth"),
            split_model=True
        )