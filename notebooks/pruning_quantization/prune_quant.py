import os
import sys
sys.path.append(os.path.join(os.getcwd().replace("notebooks/pruning_quantization", "")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import Config
from data.dataset import NetworkDataset, load_datasets, get_subset
from model_config import MLP_Models, LSTM_Models, GRU_Models, CNN_models
from utils.benchmark import Benchmark
from compact.pruning import prune_mlp_model, prune_rnn_model, prune_cnn_model
from compact.quantization import dynamic_quantize, static_quantization
import copy
import warnings
warnings.filterwarnings("ignore")

params = {
    "quant": True,
    "prune": True,
    "model": "gru"
}

# setup model
conf = Config()

if params["model"] == "mlp":
    load_model = MLP_Models()
    model_conf = load_model.mlp_4
elif params["model"] == "lstm":
    load_model = LSTM_Models()
    model_conf = load_model.lstm_4
elif params["model"] == "gru":
    load_model = GRU_Models()
    model_conf = load_model.gru_4
elif params["model"] == "cnn":
    load_model = CNN_models()
    model_conf = load_model.cnn_4
else:
    raise ValueError("params['model'] not recognized!")

model = load_model.get_model(model_conf)
model.load()
print(conf.device)


# data loader
X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(conf.datasets, model_type=load_model.type)

# create train dataloader
train_dataset = NetworkDataset(X_train, y_train)

# create test dataloader
dataset = NetworkDataset(X_test, y_test)
subset, length = get_subset(dataset, y_test)
loader = DataLoader(subset, conf.batch_size, shuffle=True)
assert conf.batch_size < length
train_dataset_no_aug = copy.deepcopy(train_dataset)
calibration_loader = DataLoader(train_dataset_no_aug, batch_size=conf.batch_size, shuffle=False)


def prune_model(model):
    # new hidden sizes
    if load_model.type == "mlp":
        pruned_model = prune_mlp_model(model.model, prune_ratio=0.4)
    elif load_model.type == "rnn":
        pruned_model = prune_rnn_model(model.model, prune_ratio=0.4)
    elif load_model.type == "cnn":
        pruned_model = prune_cnn_model(model.model, prune_ratio=0.4)
    else:
        raise ValueError("model type must be 'mlp', 'rnn' or 'cnn'!")
        

    model.model = pruned_model.to(conf.device)
    model.optimizer = torch.optim.AdamW(
        model.model.parameters(), 
        lr=conf.learning_rate, 
        weight_decay=conf.weight_decay
    )
    model.scheduler = torch.optim.lr_scheduler.ExponentialLR(
        model.optimizer, 
        gamma=conf.gamma
    )
    model.load()


def quantize_model(model):
    fp32_model = model.model
    fp32_model.cpu()
    fp32_model.eval()

    if load_model.type == "rnn":
        int8_model = dynamic_quantize(fp32_model, arch="arm")

    else:
        if load_model.type == "mlp":
            fp32_modules = ["embedding", "ln1", "output"] # mlp modules
            example_input = torch.randn(1, 513) # mlp
        elif load_model.type == "cnn":
            fp32_modules = ["embedding", "ln1", "ln2", "output"] # cnn modules
            example_input = torch.randn(1, 1, 513) # cnn
        else:
            raise ValueError("Model type not recognized!")
        
        int8_model = static_quantization(fp32_model, calibration_loader, fp32_modules, example_input, arch="arm")

    model.model = int8_model


named_comp = ""


# prune model
if params["prune"]:
    checkpoint_path = os.path.join(conf.checkpoint, "pruned_quantized", "pruned_" + model_conf["name"] + ".pth")
    model.checkpoint_path = checkpoint_path
    prune_model(model)
    named_comp += "pruned_"


# quantize model
if params["quant"]:
    quantize_model(model)
    named_comp += "quant_"


# benchmark model
name = named_comp + model_conf["name"]
result_path = os.path.join(conf.benchmark_dpu, "pruned_quantized_model", name + ".csv")
plot_path = os.path.join(conf.plot, conf.location, "pruned_quantized_model")

benchmark = Benchmark(model, loader, conf.batch_size, name, result_path, runs=length)
benchmark(plot=True, plot_path=plot_path)

# print and save result
benchmark.print_result()
benchmark.save()


