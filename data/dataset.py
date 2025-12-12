import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
import json

class NetworkDataset(Dataset):
    """
    Instance of class creates torch dataset from features and corresponding labels
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if len(self.data.shape) > 2:
            return self.data[index, :, :], self.labels[index]
        else:
            return self.data[index, :], self.labels[index]



def parse_dataset(file):
    """
    Function for parsing csv file
        Parameter:
            file: path and filename to .csv file
        Returns: 
            packet_data: parsed features from dataset as a tensor of shape [N, 513]
            labels: parsed labels from dataset as a tensor of shape [N]
            label_dict: dict describing what each label integer represents
    """
    data_file = open(file, 'r')
    packet_data = []
    labels = []
    label_dict = {}
    try:
        firstline = data_file.readline()
        firstline.replace("\n", "")
        print(firstline)
        n_feature = 0
        for line in data_file.readlines():
            line = line.replace("\n", "")
            line = line.split(',')

            if line[1] not in label_dict:
                label_dict[line[1]] = n_feature
                n_feature += 1
            
            data = line[0].split(' ')
            data = [float(d) for d in data]
            packet_data.append(data)
            labels.append(label_dict[line[1]])

        packet_data = torch.tensor(packet_data, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        return packet_data, labels, label_dict
    finally:
        data_file.close()

def split_datasets(X, y, val_size=0.1, test_size=0.2):
    """
    Splits the dataset into training, validation, and test sets using stratified sampling.
    Args:
        X (list): featuires
        labels (list): Corresponding labels
        val_size: validation size ratio of dataset
        test_size: test size ratio of dataset
    Returns:
        tuple: (X_train, y_train, X_test, y_test, X_val, y_val)
    """

    # first split: train and (val + test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=42)
    train_ids, remaining_ids = next(sss1.split(X, y))

    X_train, y_train = X[train_ids], y[train_ids]
    X_remaining, y_remaining = X[remaining_ids], y[remaining_ids]

    # second split: val and test
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size / (val_size + test_size), random_state=42)
    val_ids, test_ids = next(sss2.split(X_remaining, y_remaining))

    X_val, y_val = X_remaining[val_ids], y_remaining[val_ids]
    X_test, y_test = X_remaining[test_ids], y_remaining[test_ids]

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_datasets(dataset_path, model_type):
    """
    Function loads datasets from dataset files
    Args:
        dataset_path (string): path to datasets
        model_type (string): type of model
    Returns:
        tuple: (X_train, y_train, X_test, y_test, X_val, y_val)
    """
    # load train, val and test datasets
    train_dataset_file = os.path.join(dataset_path, "train_dataset.pt")
    X_train, y_train = torch.load(train_dataset_file)

    val_dataset_file = os.path.join(dataset_path, "val_dataset.pt")
    X_val, y_val = torch.load(val_dataset_file)

    test_dataset_file = os.path.join(dataset_path, "test_dataset.pt")
    X_test, y_test = torch.load(test_dataset_file)

    # unsqueeze tensor to fit rnn models
    if model_type != "mlp":
        X_train, X_val, X_test = X_train.unsqueeze(-1), X_val.unsqueeze(-1), X_test.unsqueeze(-1)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_label_dict(dataset_path):
    """
    Function loads label dict from dataset folder
    Args:
        dataset_path (string): path to datasets
    Returns:
        label_dict (dict): dict containing labels and respective indexes
    """
    json_file = os.path.join(dataset_path, "label_index.json")
    with open(json_file, 'r') as file:
        label_dict = json.load(file)

    return label_dict


if __name__ == "__main__":
    # extract dataset from csv file
    dataset_path = os.path.join(os.getcwd(), "datasets")
    network_data_file = os.path.join(dataset_path, "network_packet_data_test.csv")

    data, labels, label_dict = parse_dataset(network_data_file)
    json_data = json.dumps(label_dict, indent=4)
    json_file = os.path.join(dataset_path, "label_index.json")
    with open(json_file, 'w') as file:
        file.write(json_data)

    X_train, y_train, X_val, y_val, X_test, y_test = split_datasets(data, labels)

    # save train, val and test datasets
    train_dataset_file = os.path.join(dataset_path, "train_dataset.pt")
    torch.save((X_train, y_train), train_dataset_file)

    val_dataset_file = os.path.join(dataset_path, "val_dataset.pt")
    torch.save((X_val, y_val), val_dataset_file)

    test_dataset_file = os.path.join(dataset_path, "test_dataset.pt")
    torch.save((X_test, y_test), test_dataset_file)