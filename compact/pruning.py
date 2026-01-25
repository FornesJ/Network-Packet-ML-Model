import torch
import torch.nn as nn
from model.model_utils.hidden_state import LastHidden
import math
from config import Config
conf = Config()

def hidden_unit_importance(rnn, layer, gates, reverse=False):
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

def neuron_importance_linear(layer: nn.Linear):
    # importance per output neuron
    return layer.weight.norm(dim=1)  # (out_features,)

def conv1d_channel_importance(conv: nn.Conv1d):
    # importance per output channel
    return conv.weight.norm(dim=(1, 2))  # (out_channels,)

def select_units(importance, prune_ratio, rnn=False):
    H = importance.numel() if rnn else len(importance)
    keep = int(H * (1 - prune_ratio))
    idx = torch.topk(importance, keep).indices.sort().values
    return idx

def prune_linear_units(model: nn.Module, prune_ratio, in_features=None):
    """
    Structured neuron pruning for linear layers.
    Returns a new pruned model.
    """
    layers = []
    #hidden_sizes = []
    prev_keep = in_features
    linear, out_layer = model.linear, model.output

    for (layer, relu, dropout) in linear:
        importance = neuron_importance_linear(layer)
        keep = select_units(importance, prune_ratio)

        # Input and output size of layer
        in_features = layer.in_features if prev_keep is None else len(prev_keep)
        out_features = len(keep)
        #hidden_sizes.append(out_features)

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

    model.linear = nn.ModuleList(layers)
    model.output = nn.Linear(len(prev_keep), out_layer.out_features)

    return model, keep

def prune_rnn_units(model: nn.Module, prune_ratio):
    rnn = model.rnn
    assert isinstance(rnn, (nn.LSTM, nn.GRU)), "rnn must be instance of LSTM or GRU"
    assert rnn.bidirectional, "RNN must be bidirectional"

    gates = 4 if isinstance(rnn, nn.LSTM) else 3 # get number of gates
    device = conf.device

    # Compute keep indices for forward & backward
    imp_f = hidden_unit_importance(rnn, layer=0, gates=gates, reverse=False)
    imp_b = hidden_unit_importance(rnn, layer=0, gates=gates, reverse=True)

    keep_f = select_units(imp_f, prune_ratio, rnn=True)
    keep_b = select_units(imp_b, prune_ratio, rnn=True)

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
            W_hh = W_hh.view(gates, W_hh.shape[-1], W_hh.shape[-1])
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

    model.rnn = pruned
    model.h_size = new_hidden
    model.lh = LastHidden(model.rnn.num_layers, model.rnn.hidden_size)
    final_keep = torch.cat([keep_f, keep_b])
    return model, final_keep

def prune_conv_units(model: nn.Module, prune_ratio, prev_keep_idx=None):
    """
    prev_keep_idx: channels kept from previous Conv1d
    """

    conv_layers = []
    L = conf.input_size

    for layer in model.conv:
        if isinstance(layer, nn.Sequential):
            (conv, relu, dropout) = layer

            # get importance tensor and select units from conv layer
            imp = conv1d_channel_importance(conv)
            keep_idx = select_units(imp, prune_ratio)

            in_ch = conv.in_channels if prev_keep_idx is None else len(prev_keep_idx)
            out_ch = len(keep_idx)

            new_conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=conv.bias is not None,
                padding_mode=conv.padding_mode
            )

            W = conv.weight.data[keep_idx]
            if prev_keep_idx is not None:
                W = W[:, prev_keep_idx, :]

            new_conv.weight.data.copy_(W)

            if conv.bias is not None:
                new_conv.bias.data.copy_(conv.bias.data[keep_idx])
            
            conv_layers.append(nn.Sequential(new_conv, relu, dropout))
            prev_keep_idx = keep_idx

            # Calculate new spatial/temporal length based on Conv layer
            K = new_conv.kernel_size[0]
            P = new_conv.padding[0]
            S = new_conv.stride[0]
            L = math.floor((L - K + 2*P) / S + 1)
        else:
            conv_layers.append(layer)

            # Calculate new spatial/temporal length based on MaxPool layer
            K = layer.kernel_size
            P = layer.padding
            S = layer.stride
            D = layer.dilation
            if S == 0:
                S = K
            L = math.floor((L + 2*P - D*(K - 1) - 1) / S + 1)
    
    model.conv = nn.ModuleList(conv_layers) # replace old conv layers with pruned conv layers

    # Reshape keep indexes to match flattend output shape
    keep_flat = torch.cat([
        c * L + torch.arange(L).to(conf.device) for c in prev_keep_idx
    ])

    return model, keep_flat

def prune_batchnorm(bn, keep):
    new_bn = nn.BatchNorm1d(len(keep))
    new_bn.weight.data = bn.weight.data[keep]
    new_bn.bias.data = bn.bias.data[keep]
    new_bn.running_mean = bn.running_mean[keep]
    new_bn.running_var = bn.running_var[keep]
    return new_bn

def prune_layernorm(ln: nn.LayerNorm, keep_idx):
    new_ln = nn.LayerNorm(
        normalized_shape=len(keep_idx),
        eps=ln.eps,
        elementwise_affine=ln.elementwise_affine
    )

    if ln.elementwise_affine:
        new_ln.weight.data = ln.weight.data[keep_idx].clone()
        new_ln.bias.data = ln.bias.data[keep_idx].clone()

    return new_ln



def prune_mlp_model(model, prune_ratio=0.2):
    # Prune Linear layers
    pruned_model, keep = prune_linear_units(model, prune_ratio)

    # Set new hidden sizes
    pruned_model.hidden_sizes = [layer.out_features for (layer, _, _) in pruned_model.linear]

    # Prune Layer Norm
    pruned_model.ln1 = prune_layernorm(pruned_model.ln1, keep)

    return model

def prune_rnn_model(model, prune_ratio=0.2):

    # Prune LSTM/GRU layers
    pruned_model, keep = prune_rnn_units(model, prune_ratio)

    # Prune Layer Norm 1
    pruned_model.ln1 = prune_layernorm(pruned_model.ln1, keep)

    # Prune Linear ayers
    pruned_model, keep = prune_linear_units(pruned_model, prune_ratio, in_features=keep)

    # Set new hidden sizes
    pruned_model.linear_sizes = [layer.out_features for (layer, _, _) in pruned_model.linear]

    # Prune Layer Norm 2
    pruned_model.ln2 = prune_layernorm(pruned_model.ln2, keep)

    return model

def prune_cnn_model(model, prune_ratio=0.2):

    # Prune Conv1d layers
    pruned_model, keep = prune_conv_units(model, prune_ratio)

    # Set new out channels
    new_filters = []
    idx = 0
    for layer in pruned_model.conv:
        if isinstance(layer, nn.Sequential):
            (conv, _, _) = layer
            new_filters.append((conv.out_channels, conv.kernel_size[0], pruned_model.filters[idx][-1]))
            idx += 1
    pruned_model.filters = new_filters

    # Prune Layer Norm 1
    pruned_model.ln1 = prune_layernorm(pruned_model.ln1, keep)

    # Prune Linear ayers
    pruned_model, keep = prune_linear_units(pruned_model, prune_ratio, in_features=keep)

    # Set new hidden sizes
    pruned_model.linear_sizes = [layer.out_features for (layer, _, _) in pruned_model.linear]

    # Prune Layer Norm 2
    pruned_model.ln2 = prune_layernorm(pruned_model.ln2, keep)

    return model







