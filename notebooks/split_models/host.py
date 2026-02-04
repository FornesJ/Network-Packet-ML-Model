import os
import sys
sys.path.append(os.path.join(os.getcwd().replace("notebooks/split_models", "")))

import torch
from torch.utils.data import DataLoader
from config import Config
from data.dataset import NetworkDataset, load_datasets, get_subset
from model_config import CNN_models, MLP_Models, LSTM_Models, GRU_Models
from model.model_utils.copy_param import host_copy_model
from transfer.transfer_tensors import HostSocket
from utils.benchmark import SplitBenchmark


params = {
    "model": "gru",
    "split_index": 0
}

# setup model
conf = Config()

if params["model"] == "mlp":
    load_model = MLP_Models()
    model_conf = load_model.mlp_4

    if params["split_index"] == 3:
        split_conf = load_model.split_mlp_3
    elif params["split_index"] == 2:
        split_conf = load_model.split_mlp_2
    elif params["split_index"] == 1:
        split_conf = load_model.split_mlp_1
    else:
        raise ValueError(f"index: {params['split_index']} does not exist!")
    
elif params["model"] == "lstm":
    load_model = LSTM_Models()
    model_conf = load_model.lstm_4

    if params["split_index"] == 3:
        split_conf = load_model.split_lstm_3
    elif params["split_index"] == 2:
        split_conf = load_model.split_lstm_2
    elif params["split_index"] == 1:
        split_conf = load_model.split_lstm_1
    elif params["split_index"] == 0:
        split_conf = load_model.split_lstm_0
    else:
        raise ValueError(f"index: {params['split_index']} does not exist!")
    
elif params["model"] == "gru":
    load_model = GRU_Models()
    model_conf = load_model.gru_4

    if params["split_index"] == 3:
        split_conf = load_model.split_gru_3
    elif params["split_index"] == 2:
        split_conf = load_model.split_gru_2
    elif params["split_index"] == 1:
        split_conf = load_model.split_gru_1
    elif params["split_index"] == 0:
        split_conf = load_model.split_gru_0
    else:
        raise ValueError(f"index: {params['split_index']} does not exist!")
    
elif params["model"] == "cnn":
    load_model = CNN_models()
    model_conf = load_model.cnn_4

    if params["split_index"] == 3:
        split_conf = load_model.split_cnn_3
    elif params["split_index"] == 2:
        split_conf = load_model.split_cnn_2
    elif params["split_index"] == 1:
        split_conf = load_model.split_cnn_1
    elif params["split_index"] == 0:
        split_conf = load_model.split_cnn_0
    else:
        raise ValueError(f"index: {params['split_index']} does not exist!")
    
else:
    raise ValueError("params['model'] not recognized!")

model = load_model.get_model(model_conf)
split_model = load_model.get_model(split_conf)
split_model.model.split = "host"
model.load()
host_sock = HostSocket(so_file=conf.sock_so)
name = split_conf["name"]
result_path = os.path.join(conf.benchmark_host, "split_model", name + ".csv")
plot_path = os.path.join(conf.plot, conf.location, "split_model")


# data loader
X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(conf.datasets, model_type=load_model.type)

# create train dataloader
train_dataset = NetworkDataset(X_train, y_train)

# create test dataloader
dataset = NetworkDataset(X_test, y_test)
subset, length = get_subset(dataset, y_test)
loader = DataLoader(subset, conf.batch_size, shuffle=True)


# copy parameters from model to split model
split_idx = split_conf["split_idx"]
host_model = split_model.model.host_model
split_model.model.host_model = host_copy_model(model.model, host_model, split_idx, type=load_model.type)


# run benchmark
benchmark = SplitBenchmark(split_model, loader, conf.batch_size, name, result_path, runs=length, socket=host_sock, split=conf.location)
benchmark.open()
benchmark(plot=True, plot_path=plot_path)
benchmark.transfer_time()
benchmark.close()

# print and save result
benchmark.print_result()
benchmark.save()