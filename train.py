import torch
import torch.nn as nn

class Model:
    """
    Class for storing model weights, training and evaluating model
    """
    def __init__(self, 
                model, 
                loss_function, 
                conf,
                checkpoint_path,
                split_model=False):
        """
        Constructor for class
        Args:
            model (torch.nn.Module model): Pytorch model (mlp, gru, lstm)
            loss_function (torch.nn.Module): criterion for model
            conf (Config): hyperparameters
            dpu_model (torch.nn.Module model): if split model (self.model -> host model, self.dpu_model -> dpu model)
        """
        self.model = model
        self.criterion = loss_function
        self.split_model = split_model

        if self.split_model:
            parameters = list(self.model.dpu_model.parameters()) + list(self.model.host_model.parameters())
        else:
            parameters = self.model.parameters()

        self.optimizer = torch.optim.AdamW(
            parameters, 
            lr=conf.learning_rate, 
            weight_decay=conf.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, 
            gamma=conf.gamma
        )
        
        self.device = conf.device
        self.checkpoint_path = checkpoint_path


    def train(self, train_loader, val_loader, epochs):
        """
        Method for training model on training dataset and validating each epoch with validation dataset
        Args:
            train_loader (DataLoader): training data_loader
            val_loader (DataLoader): validation data_loader
            epochs (int): number of epochs
        Returns:
            acc_list (list): list with accuracies pr. epoch
            train_loss_list (list): list conatining training loss per epoch
            val_loss_list (list): list conatining validation loss per epoch
        """
        acc_list = []
        train_loss_list = []
        val_loss_list = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            # train model
            for (data, labels) in train_loader:
                if not data.is_cuda or not labels.is_cuda:
                    data, labels = data.to(self.device), labels.to(self.device)
                
                _, pred = self.model(data)
                loss = self.criterion(pred, labels)
                running_loss += loss.item() * data.size(0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # evaluate model
            val_loss, acc = self.evaluate(val_loader)
            train_loss = running_loss / (train_loader.__len__() * train_loader.batch_size)
            self.scheduler.step()

            acc_list.append(acc)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
        
            print(f"Epoch: {epoch}/{epochs}, Accuracy: {100*acc:.2f}%, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        return acc_list, train_loss_list, val_loss_list
    
    def evaluate(self, val_loader):
        """
        Method for evaluating model accuracy and validation loss
        Args:
            val_loader: Dataloader containing batches and labels
        Retruns:
            acc (float): accuracy of the model
        """
        y_true, y_pred = [], []

        self.model.eval()

        for (data, labels) in val_loader:
            if not data.is_cuda or not labels.is_cuda:
                data, labels = data.to(self.device), labels.to(self.device)

            with torch.no_grad():
                _, pred = self.model(data)
            
            y_true.append(labels)
            y_pred.append(pred)

        y_true, y_pred = torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)

        loss = self.criterion(y_pred, y_true)

        # evaluate accuracy
        acc = (y_pred.argmax(dim=1) == y_true).float().mean()
        # bal_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred.argmax(dim=1))

        return loss, acc
    
    def load(self):
        """
        Method for loading model wights from checkpoint
        Args:
            checkpoint_path (string): path to checkpoint file
        """
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device(device=self.device))

        if self.split_model:
            self.model.dpu_model.load_state_dict(checkpoint["dpu_model_state_dict"])
            self.model.host_model.load_state_dict(checkpoint["host_model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {self.checkpoint_path}!")

    def save(self):
        """
        Method for saving model weights
        Args:
            checkpoint_path (string): path to checkpoint file
        """
        checkpoint = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }

        if self.split_model:
            checkpoint["dpu_model_state_dict"] = self.model.dpu_model.state_dict()
            checkpoint["host_model_state_dict"] = self.model.host_model.state_dict()
        else:
            checkpoint["model_state_dict"] = self.model.state_dict()

        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at {self.checkpoint_path}")




