import torch
import torch.nn as nn
from model.mlp import MLP

def copy_lstm_layers(src, dst, src_start_layer):
    with torch.no_grad():
        for i in range(dst.num_layers):
            src_layer = src_start_layer + i

            for suffix in ["", "_reverse"]:
                getattr(dst, f"weight_ih_l{i}{suffix}").copy_(
                    getattr(src, f"weight_ih_l{src_layer}{suffix}")
                )
                getattr(dst, f"weight_hh_l{i}{suffix}").copy_(
                    getattr(src, f"weight_hh_l{src_layer}{suffix}")
                )
                getattr(dst, f"bias_ih_l{i}{suffix}").copy_(
                    getattr(src, f"bias_ih_l{src_layer}{suffix}")
                )
                getattr(dst, f"bias_hh_l{i}{suffix}").copy_(
                    getattr(src, f"bias_hh_l{src_layer}{suffix}")
                )

def copy_linear_layers(src, dst):
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)

def copy_batch_norm(src, dst):
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)
        
        # buffers
        dst.running_mean.copy_(src.running_mean)
        dst.running_var.copy_(src.running_var)
        dst.num_batches_tracked.copy_(src.num_batches_tracked)

def copy_layer_norm(src, dst):
    assert src.normalized_shape == dst.normalized_shape
    with torch.no_grad():
        if src.elementwise_affine:
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)

def copy_conv_layers(src, dst):
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)


def dpu_copy_model(model, dpu_model):
    for name, mod in dpu_model.named_modules():
        if isinstance(mod, nn.Linear):
            src_mod = model.get_submodule(name)
            copy_linear_layers(src_mod, mod)
        
        if isinstance(mod, nn.BatchNorm1d):
            src_mod = getattr(model, name)
            copy_batch_norm(src_mod, mod)

        if isinstance(mod, nn.LayerNorm):
            src_mod = getattr(model, name)
            copy_layer_norm(src_mod, mod)

        if isinstance(mod, (nn.LSTM, nn.GRU)):
            src_mod = model.get_submodule(name)
            copy_lstm_layers(src_mod, mod, 0)

        if isinstance(mod, nn.Conv1d):
            src_mod = model.get_submodule(name)
            copy_conv_layers(src_mod, mod)

    return dpu_model

def host_copy_model(model, host_model, split_idx, type):
    linear_index = 0
    conv_indx = 0
    for name, mod in host_model.named_modules():
        if isinstance(mod, nn.Linear):
            if type == "mlp" and name != "output":
                src_name = f"linear.{split_idx + linear_index}.0"
                src_mod = model.get_submodule(src_name)
            else:
                src_mod = model.get_submodule(name)
            copy_linear_layers(src_mod, mod)
            linear_index += 1

        if isinstance(mod, nn.BatchNorm1d):
            src_mod = getattr(model, name, None)
            if src_mod == None and name == "bn":
                src_mod = getattr(model, "bn2")
            copy_batch_norm(src_mod, mod)

        if isinstance(mod, nn.LayerNorm):
            if type != "mlp" and isinstance(host_model, (MLP)):
                name = "ln2"
            src_mod = getattr(model, name)
            copy_layer_norm(src_mod, mod)

        if isinstance(mod, (nn.LSTM, nn.GRU)):
            src_mod = model.get_submodule(name)
            copy_lstm_layers(src_mod, mod, split_idx)

        if isinstance(mod, nn.Conv1d):
            src_name = f"conv.{2*split_idx + conv_indx}.0"
            src_mod = model.get_submodule(src_name)
            copy_conv_layers(src_mod, mod)
            conv_indx += 2
            
    return host_model