import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from config import Config
conf = Config()

class PruneModel():
    def __init__(self, prune_ratio=0.2):
        self.prune_ratio = prune_ratio

    def hidden_unit_importance(self, rnn, layer, gates, reverse=False):
        """
        Method computes importance per hidden unit for a single direction.
        Params:
            rnn (nn.Module): pytorch nn.LSTM or nn.GRU rnn
            layer (int): number of hidden layers
            reverse (boolean): if true: reverse direction
        """
        H = rnn.hidden_size
        suffix = "_reverse" if reverse else ""

        W_hh = getattr(rnn, f'weight_hh_l{layer}{suffix}')
        W_hh = W_hh.view(gates, H, H)

        importance = W_hh.norm(dim=(0, 2))
        return importance
    
    def neuron_importance_linear(self, layer: nn.Linear):
        # importance per output neuron
        return layer.weight.norm(dim=1)  # (out_features,)
    
    def select_units(self, importance):
        keep = int(importance.numel() * (1 - self.prune_ratio))
        idx = torch.topk(importance, keep).indices.sort().values
        return idx
    
    def prune_linear_units(self, linear, out_layer, in_features=None):
        """
        Structured neuron pruning for linear layers.
        Returns a new pruned model.
        """
        layers = []
        prev_keep = in_features

        for (layer, relu, dropout) in linear:
            importance = self.neuron_importance_linear(layer)
            keep = self.select_units(importance, self.prune_ratio)

            # Input and output size of layer
            in_features = layer.in_features if prev_keep is None else len(prev_keep)
            out_features = len(keep)

            # Create new layer
            new_layer = nn.Linear(in_features, out_features, bias=layer.bias is not None)

            # Copy weights from old layer
            W = layer.weight[keep]
            if prev_keep is not None:
                W = W[:, prev_keep]

            new_layer.weight.data.copy_(W)

            # Copy bias from old layer
            if layer.bias is not None:
                new_layer.bias.data.copy_(layer.bias[keep])
            
            full_layer = nn.Sequential(new_layer, relu, dropout)
            layers.append(full_layer)
            prev_keep = keep
        
        return nn.ModuleList(*layers), nn.Linear(len(prev_keep), out_layer.out_features)

    
    def prune_rnn_units(self, rnn):
        assert isinstance(rnn, (nn.LSTM, nn.GRU)), "rnn must be instance of LSTM or GRU"
        assert rnn.bidirectional, "RNN must be bidirectional"

        gates = 4 if isinstance(rnn, nn.LSTM) else 3 # get number of gates
        device = conf.device

        # Compute keep indices for forward & backward
        imp_f = self.hidden_unit_importance(rnn, layer=0, gates=gates, reverse=False)
        imp_b = self.hidden_unit_importance(rnn, layer=0, gates=gates, reverse=True)

        keep_f = self.select_units(imp_f)
        keep_b = self.select_units(imp_b)

        new_hidden = keep_f.numel() # new hidden size

        # new pruned lstm/gru rnn
        pruned = type(rnn)(
            input_size=rnn.input_size,
            hidden_size=new_hidden,
            num_layers=rnn.num_layers,
            bias=rnn.bias,
            batch_first=rnn.batch_first,
            dropout=rnn.dropout,
            bidirectional=True
        ).to(device)

        for layer in range(rnn.num_layers):
            for reverse, keep in [(False, keep_f), (True, keep_b)]:
                suffix = "_reverse" if reverse else ""

                # Input to hidden parameters
                W_ih = getattr(rnn, f'weight_ih_l{layer}{suffix}')
                W_ih = W_ih.view(gates, -1, W_ih.shape[1])

                # Input size doubles after first layer
                if layer > 0:
                    W_ih = W_ih[:, keep, :][:, :, torch.cat([keep_f, keep_b])]
                else:
                    W_ih = W_ih[:, keep, :]
                
                # Set input to hidden parameters in layer
                setattr(
                    pruned,
                    f'weight_ih_l{layer}{suffix}',
                    nn.Parameter(W_ih.reshape(gates * new_hidden, -1))
                )

                # Hidden to hidden parameters
                W_hh = getattr(rnn, f'weight_hh_l{layer}{suffix}')
                W_hh = W_hh.view(gates, -1, -1)
                W_hh = W_hh[:, keep][:, :, keep]

                # Set hidden to hidden parameters in layer
                setattr(
                    pruned,
                    f'weight_hh_l{layer}{suffix}',
                    nn.Parameter(W_hh.reshape(gates * new_hidden, new_hidden))
                )

                # Biases for input to hidden and hidden to hidden
                if rnn.bias:
                    for b in ['bias_ih_l', 'bias_hh_l']:
                        bias = getattr(rnn, f'{b}{layer}{suffix}')
                        bias = bias.view(gates, -1)[:, keep]

                        # Set bias
                        setattr(
                            pruned,
                            f'{b}{layer}{suffix}',
                            nn.Parameter(bias.reshape(gates * new_hidden))
                        )

        return pruned, keep_f, keep_b








