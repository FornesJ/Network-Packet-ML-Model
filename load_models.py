import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Model
from split_model import SplitModelDPU, SplitModelHost

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.1, alpha=0.9, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        


######################################################################
#                                                                    #
#   Pytorch Models: MLP and light MLP                                #
#                                                                    #
######################################################################

# create MLP model:
class MLP(nn.Module):
    def __init__(self, size):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(size, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Dropout(p=0.10),

            nn.Linear(320, 24)
        )
    
    def forward(self, x):
        out = self.mlp(x)
        return out

# create light MLP model
class LightMLP(nn.Module):
    def __init__(self, size):
        super(LightMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.10),

            nn.Linear(32, 24)
        )
    
    def forward(self, x):
        out = self.mlp(x)
        return out


# Load mlp model
torch.manual_seed(42)
mlp_model = MLP(513).to(device)
mlp_criterion = FocalLoss()
mlp_optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=0.0001, weight_decay=0.01)
mlp_scheduler = torch.optim.lr_scheduler.ExponentialLR(mlp_optimizer, 0.9)
mlp = Model(mlp_model, mlp_criterion, mlp_optimizer, mlp_scheduler, device)

# Load light mlp model
torch.manual_seed(42)
light_mlp_model = LightMLP(513).to(device)
light_mlp_criterion = FocalLoss()
light_mlp_optimizer = torch.optim.AdamW(light_mlp_model.parameters(), lr=0.0001, weight_decay=0.01)
light_mlp_scheduler = torch.optim.lr_scheduler.ExponentialLR(light_mlp_optimizer, 0.9)
light_mlp = Model(light_mlp_model, light_mlp_criterion, light_mlp_optimizer, light_mlp_scheduler, device)




######################################################################
#                                                                    #
#   Pytorch Models: LSTM and light LSTM                              #
#                                                                    #
######################################################################


# create LSTM model:
class INML(nn.Module):
    def __init__(self, i_size, h_size):
        super(INML, self).__init__()
        self.i_size = i_size
        self.h_size = h_size
        self.lstm1 = nn.LSTM(input_size=i_size, hidden_size=h_size, num_layers=4, batch_first=True, dropout=0.15, bidirectional=True, device=device)
        self.bn1 = nn.BatchNorm1d(2 * h_size)

        self.fc = nn.Sequential(
            nn.Linear(2*h_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(128, 24)
        )
        
    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(8, x.shape[0], self.h_size).to(device)
            c0 = torch.zeros(8, x.shape[0], self.h_size).to(device)

        output, (h0, c0) = self.lstm1(x, (h0, c0))  # output: [B, T, 2*h_size]

        # take last layer's hidden state (both directions)
        # h0 shape: [num_layers*2, B, size]
        h_last = h0.view(4, 2, x.shape[0], self.h_size)[-1]  # [2, B, h_size]
        h_last = torch.cat((h_last[0], h_last[1]), dim=1)  # [B, 2*h_size]

        # apply BN + FC
        h_last = self.bn1(h_last)         # [B, 2*h_size] → batch norm


        out = self.fc(h_last)            # [B, 24]
        return out
    

class LightLSTM(nn.Module):
    def __init__(self, i_size, h_size):
        super(LightLSTM, self).__init__()
        self.i_size = i_size
        self.h_size = h_size
        self.lstm1 = nn.LSTM(input_size=i_size, hidden_size=h_size, num_layers=1, batch_first=True, bidirectional=True, device=device)
        self.bn1 = nn.BatchNorm1d(2 * h_size)

        self.fc = nn.Sequential(
            nn.Linear(2*h_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(32, 24)
        )

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(2, x.shape[0], self.h_size).to(device)
            c0 = torch.zeros(2, x.shape[0], self.h_size).to(device)

        output, (h0, c0) = self.lstm1(x, (h0, c0))  # output: [B, T, 2*h_size]

        # take last layer's hidden state (both directions)
        # h0 shape: [num_layers*2, B, size]
        h_last = h0.view(1, 2, x.shape[0], self.h_size)[-1]  # [2, B, h_size]
        h_last = torch.cat((h_last[0], h_last[1]), dim=1)  # [B, 2*h_size]

        # apply BN + FC
        h_last = self.bn1(h_last)         # [B, 2*h_size] → batch norm


        out = self.fc(h_last)            # [B, 24]
        return out


# Load LSTM model
torch.manual_seed(42)
lstm_model = INML(1, 64).to(device)
lstm_criterion = FocalLoss()
lstm_optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=0.0001, weight_decay=0.01)
lstm_scheduler = torch.optim.lr_scheduler.ExponentialLR(lstm_optimizer, 0.9)
lstm = Model(lstm_model, lstm_criterion, lstm_optimizer, lstm_scheduler, device)

# Load light LSTM model
torch.manual_seed(42)
light_lstm_model = LightLSTM(1, 32).to(device)
light_lstm_criterion = FocalLoss()
light_lstm_optimizer = torch.optim.AdamW(light_lstm_model.parameters(), lr=0.0001, weight_decay=0.01)
light_lstm_scheduler = torch.optim.lr_scheduler.ExponentialLR(light_lstm_optimizer, 0.9)
light_lstm = Model(light_lstm_model, light_lstm_criterion, light_lstm_optimizer, light_lstm_scheduler, device)






######################################################################
#                                                                    #
#   Pytorch Models: GRU and light GRU                                #
#                                                                    #
######################################################################


# create GRU model:
class INMLGRU(nn.Module):
    def __init__(self, i_size, h_size):
        super(INMLGRU, self).__init__()
        self.i_size = i_size
        self.h_size = h_size
        self.gru1 = nn.GRU(input_size=i_size, hidden_size=h_size, num_layers=4, batch_first=True, dropout=0.15, bidirectional=True, device=device)
        self.bn1 = nn.BatchNorm1d(2 * h_size)

        self.fc = nn.Sequential(
            nn.Linear(2*h_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(128, 24)
        )
        
    def forward(self, x, h0=None):
        if h0 is None:
            h0 = torch.zeros(8, x.shape[0], self.h_size).to(device)

        output, h0 = self.gru1(x, h0)  # output: [B, T, 2*h_size]

        # take last layer's hidden state (both directions)
        # h0 shape: [num_layers*2, B, size]
        h_last = h0.view(4, 2, x.shape[0], self.h_size)[-1]  # [2, B, h_size]
        h_last = torch.cat((h_last[0], h_last[1]), dim=1)  # [B, 2*h_size]

        # apply BN + FC
        h_last = self.bn1(h_last)         # [B, 2*h_size] → batch norm


        out = self.fc(h_last)            # [B, 24]
        return out
    



class LightGRU(nn.Module):
    def __init__(self, i_size, h_size):
        super(LightGRU, self).__init__()
        self.i_size = i_size
        self.h_size = h_size
        self.gru1 = nn.GRU(input_size=i_size, hidden_size=h_size, num_layers=1, batch_first=True, bidirectional=True, device=device)
        self.bn1 = nn.BatchNorm1d(2 * h_size)

        self.fc = nn.Sequential(
            nn.Linear(2*h_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(32, 24)
        )

    def forward(self, x, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(2, x.shape[0], self.h_size).to(device)

        output, h0 = self.gru1(x, h0)  # output: [B, T, 2*h_size]

        # take last layer's hidden state (both directions)
        # h0 shape: [num_layers*2, B, size]
        h_last = h0.view(1, 2, x.shape[0], self.h_size)[-1]  # [2, B, h_size]
        h_last = torch.cat((h_last[0], h_last[1]), dim=1)  # [B, 2*h_size]

        # apply BN + FC
        h_last = self.bn1(h_last)         # [B, 2*h_size] → batch norm


        out = self.fc(h_last)            # [B, 24]
        return out


# Load GRU model
torch.manual_seed(42)
gru_model = INMLGRU(1, 64).to(device)
gru_criterion = FocalLoss()
gru_optimizer = torch.optim.AdamW(gru_model.parameters(), lr=0.0001, weight_decay=0.01)
gru_scheduler = torch.optim.lr_scheduler.ExponentialLR(gru_optimizer, 0.9)
gru = Model(gru_model, gru_criterion, gru_optimizer, gru_scheduler, device)

# Load light GRU model
torch.manual_seed(42)
light_gru_model = LightGRU(1, 32).to(device)
light_gru_criterion = FocalLoss()
light_gru_optimizer = torch.optim.AdamW(light_gru_model.parameters(), lr=0.0001, weight_decay=0.01)
light_gru_scheduler = torch.optim.lr_scheduler.ExponentialLR(light_gru_optimizer, 0.9)
light_gru = Model(light_gru_model, light_gru_criterion, light_gru_optimizer, light_gru_scheduler, device)




######################################################################
#                                                                    #
#   Pytorch Models: Split GRU model                                  #
#                                                                    #
######################################################################


# create GRU model:
class SPLITGRU(nn.Module):
    def __init__(self, i_size, h_size, binary=False):
        super(SPLITGRU, self).__init__()
        self.i_size = i_size
        self.h_size = h_size
        self.binary = binary
        self.gru1 = nn.GRU(input_size=i_size, hidden_size=h_size, num_layers=2, batch_first=True, dropout=0.15, bidirectional=True, device=device)
        self.bn1 = nn.BatchNorm1d(2 * h_size)

        self.fc_binary = nn.Sequential(
            nn.Linear(2*self.h_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.fc_class = nn.Sequential(
            nn.Linear(2*self.h_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(128, 24)
        )
        
    def forward(self, x, h0=None):
        if h0 is None:
            h0 = torch.zeros(4, x.shape[0], self.h_size).to(device)

        output, h0 = self.gru1(x, h0)  # output: [B, T, 2*h_size]

        # take last layer's hidden state (both directions)
        # h0 shape: [num_layers*2, B, size]
        h_last = h0.view(2, 2, x.shape[0], self.h_size)[-1]  # [2, B, h_size]
        h_last = torch.cat((h_last[0], h_last[1]), dim=1)  # [B, 2*h_size]

        # apply BN + FC
        logits = self.bn1(h_last)  # [B, 2*h_size] → batch norm
        if self.binary:
            out = self.fc_binary(logits) # [B, 1]
        else:
            out = self.fc_class(logits) # [B, 24]

        return out, logits
    
class LightSplitDPU(nn.Module):
    def __init__(self, i_size, h_size):
        super(LightSplitDPU, self).__init__()
        self.i_size = i_size
        self.h_size = h_size
        self.gru1 = nn.GRU(input_size=i_size, hidden_size=h_size, num_layers=1, batch_first=True, bidirectional=True, device=device)
        self.bn1 = nn.BatchNorm1d(2 * h_size)

        self.fc = nn.Sequential(
            nn.Linear(2*h_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(2, x.shape[0], self.h_size).to(device)

        output, h0 = self.gru1(x, h0)  # output: [B, T, 2*h_size]

        # take last layer's hidden state (both directions)
        # h0 shape: [num_layers*2, B, size]
        h_last = h0.view(1, 2, x.shape[0], self.h_size)[-1]  # [2, B, h_size]
        h_last = torch.cat((h_last[0], h_last[1]), dim=1)  # [B, 2*h_size]

        # apply BN + FC
        logits = self.bn1(h_last)         # [B, 2*h_size] → batch norm

        out = self.fc(logits)            # [B, 1]
        return out, logits

# load dpu model
torch.manual_seed(42)
dpu_model = SPLITGRU(1, 64, binary=True).to(device)
dpu_criterion = nn.BCELoss()
dpu_optimizer = torch.optim.AdamW(dpu_model.parameters(), lr=0.0001, weight_decay=0.01)
dpu_scheduler = torch.optim.lr_scheduler.ExponentialLR(dpu_optimizer, 0.9)
dpu = SplitModelDPU(dpu_model, dpu_criterion, dpu_optimizer, dpu_scheduler, device)


# load host model
torch.manual_seed(42)
host_model = SPLITGRU(128, 64).to(device)
host_criterion = FocalLoss()
host_optimizer = torch.optim.AdamW(host_model.parameters(), lr=0.0001, weight_decay=0.01)
host_scheduler = torch.optim.lr_scheduler.ExponentialLR(host_optimizer, 0.9)
host = SplitModelHost(host_model, host_criterion, host_optimizer, host_scheduler, device)


# load light dpu model
torch.manual_seed(42)
light_dpu_model = LightSplitDPU(1, 64).to(device)
light_dpu_criterion = nn.BCELoss()
light_dpu_optimizer = torch.optim.AdamW(light_dpu_model.parameters(), lr=0.0001, weight_decay=0.01)
light_dpu_scheduler = torch.optim.lr_scheduler.ExponentialLR(light_dpu_optimizer, 0.9)
light_dpu = SplitModelDPU(light_dpu_model, light_dpu_criterion, light_dpu_optimizer, light_dpu_scheduler, device)




######################################################################
#                                                                    #
#   All models available in model dict                               #
#                                                                    #
######################################################################


models = {
    "mlp": mlp,
    "light_mlp": light_mlp,
    "lstm": lstm,
    "light_lstm": light_lstm,
    "gru": gru,
    "light_gru": light_gru,
    "dpu": dpu,
    "host": host,
    "light_dpu": light_dpu
}