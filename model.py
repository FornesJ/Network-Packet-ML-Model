import torch

class Model:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, train_loader, val_loader, epochs):
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
                
                pred = self.model(data)
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
        y_true, y_pred = [], []

        self.model.eval()
        for (data, labels) in val_loader:
            if not data.is_cuda or not labels.is_cuda:
                data, labels = data.to(self.device), labels.to(self.device)
            
            with torch.no_grad():
                pred = self.model(data)
            
            y_true.append(labels)
            y_pred.append(pred)

        y_true, y_pred = torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)

        loss = self.criterion(y_pred, y_true)

        # evaluate accuracy

        acc = (y_pred.argmax(dim=1) == y_true).float().mean()

        return loss, acc
    
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

