import os
import sys
sys.path.append(os.path.join(os.getcwd().replace("notebooks/split_models", "")))

import torch
from torch.utils.data import DataLoader
from config import Config
from data.dataset import NetworkDataset, load_datasets
from model_config import CNN_models, MLP_Models, LSTM_Models, GRU_Models
from model.copy_param import dpu_copy_model
from transfer.transfer_tensors import DPUSocket
from utils.benchmark import SplitBenchmark

# setup
conf = Config()
load_models = CNN_models()
split_conf = load_models.split_cnn_3
model_conf = load_models.cnn_4
split_model = load_models.get_model(split_conf)
model = load_models.get_model(model_conf)
model.load()
dpu_sock = DPUSocket(so_file=conf.sock_so, localhost=False)
location = "dpu"
name = "split_" + split_conf["name"]
result_path = os.path.join(conf.benchmark_dpu, "split_model", name + ".txt")
split_model.model.split = location

# dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(conf.datasets, load_models.type)

# create train, val and test dataloaders
# train_dataset = NetworkDataset(X_train, y_train)
# train_loader = DataLoader(train_dataset, conf.batch_size, shuffle=True)
# 
# val_dataset = NetworkDataset(X_val, y_val)
# val_loader = DataLoader(val_dataset, conf.batch_size, shuffle=True)

test_dataset = NetworkDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, conf.batch_size, shuffle=True)


# copy parameters from model to split model
split_idx = split_conf["split_idx"]
dpu_model = split_model.model.dpu_model
split_model.model.dpu_model = dpu_copy_model(model.model, dpu_model)


# run benchmark
benchmark = SplitBenchmark(split_model, test_loader, conf.batch_size, name, result_path, socket=dpu_sock, split=location)
benchmark.open()
benchmark()
benchmark.transfer_time()
benchmark.close()

# print and save result
benchmark.print_result()