import torch

def binary_dataset(labels, label_dict):
    binary_labels = []
    label_names = list(label_dict.keys())
    for label in labels:
        if label_names[label] == "Normal":
            binary_labels.append(0)
        else:
            binary_labels.append(1)
    y_binary = torch.tensor(binary_labels, dtype=torch.float)
    return y_binary

class SplitModel:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device=self.device))

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}!")

    def save(self, checkpoint_path):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

class SplitModelHost(SplitModel):
    def __init__(self, model, criterion, optimizer, scheduler, device):
        super().__init__(model, criterion, optimizer, scheduler, device)

    def inference(self, data):

        with torch.no_grad():
            pred, _ = self.model(data)

        return pred

    def evaluate(self, logits, logits_targets):
        y_true, y_pred = [], []

        self.model.eval()
        for (data, labels) in zip(logits, logits_targets):
            if not data.is_cuda or not labels.is_cuda:
                data, labels = data.to(self.device), labels.to(self.device)

            pred = self.inference(data)
            y_true.append(labels)
            y_pred.append(pred)

        y_true, y_pred = torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)
        loss = self.criterion(y_pred, y_true)
        acc = (y_pred.argmax(dim=1) == y_true).float().mean()

        return acc, loss


class SplitModelDPU(SplitModel):
    def __init__(self, model, criterion, optimizer, scheduler, device):
        super().__init__(model, criterion, optimizer, scheduler, device)
    
    def inference(self, data, labels=None):
        logits_list = []
        labels_list = []
        with torch.no_grad():
            pred, logits = self.model(data)
            pred = torch.squeeze(pred)
            pred = pred.round()
            logits = logits.detach()  # break the graph here
            logits = logits.unsqueeze(1)
        
        for idx in range(data.shape[0]):
            if pred[idx] > 0:
                logits_list.append(logits[idx])
                if labels != None:
                    labels_list.append(labels[idx])

        logits_list = torch.cat(logits_list, dim=0).unsqueeze(1)

        if labels != None:
            labels_list = torch.tensor(labels_list).to(self.device)
            return pred, logits_list, labels_list
        else:
            return pred, logits_list
    
    def evaluate(self, loader, label_dict):
        y_true, y_pred = [], []
        logits_list, logits_labels = [], []
        self.model.eval()
        for (data, labels) in loader:
            if not data.is_cuda or not labels.is_cuda:
                data, labels = data.to(self.device), labels.to(self.device)

            bin_labels = binary_dataset(labels, label_dict)
            bin_labels = bin_labels.to(self.device)

            pred, logits, logits_targets = self.inference(data, labels=labels)
            y_true.append(bin_labels)
            y_pred.append(pred)
            logits_list.append(logits)
            logits_labels.append(logits_targets)

        y_true, y_pred = torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)
        loss = self.criterion(y_pred, y_true)
        acc = (y_pred == y_true).float().mean()

        return acc, loss, logits_list, logits_labels

        





            





        





