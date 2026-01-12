import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_functions.kd_loss import KD_Loss, RKD_Loss, Feature_Loss
from config import Config
conf = Config()

class KnowledgeDistillation:
    """
    class for Knowledge Distillation
    """
    def __init__(self, 
                teacher, 
                student, 
                device, 
                model_type,
                distillation="response",  
                T=4.0,
                soft_target_loss_weight=0.25,
                rkd_loss_weight=0.1,
                feature_loss_weight=1.5, 
                loss_weight=0.75):
        """
        Constructor for KnowledgeDistillation
        Params:
            teacher (Model): large teacher model (pretrained)
            student (Model): light student model
            device (string): device
            distillation (string): define distillation mode: response | feature | relation
            T (float): temperature for kd_distillation
            soft_target_loss_weight (float): weight
            rkd_loss_weight (float): wight
            feature_loss_weight (float): weight
            loss_weight (float): weight
            rnn (boolean): difined as true to handle rnn models
        """
        self.teacher = teacher
        self.student = student
        self.device = device
        self.type = model_type
        self.distillation = distillation
        self.T=T
        self.soft_target_loss_weight = soft_target_loss_weight
        self.rkd_loss_weight = rkd_loss_weight
        self.feature_loss_weight= feature_loss_weight
        self.loss_weight = loss_weight
        self.kd_loss = KD_Loss(T=self.T)
        self.rkd_loss = RKD_Loss()
        self.feature_loss = Feature_Loss()

        # if feature distillation: create attention adapter to reshape features from student and teacher
        if self.distillation == "feature":
            # Use adapter to match teacher and student hidden dimensions
            if self.type == "rnn":
                assert self.student.model.n_layers == self.teacher.model.n_layers, \
                "Teacher and student must have eaqual number of hidden layers!"
                s_sizes = [2 * self.student.model.h_size for _ in range(self.student.model.n_layers)]
                t_sizes = [2 * self.teacher.model.h_size for _ in range(self.teacher.model.n_layers)]
            elif self.type == "cnn":
                assert self.student.model.conv_layers == self.teacher.model.conv_layers, \
                "Teacher and student must have eaqual number of hidden layers!"
                s_sizes, t_sizes = self.student.model.conv_layers, self.teacher.model.conv_layers
            else:
                assert len(self.student.model.hidden_sizes) == len(self.teacher.model.hidden_sizes), \
                "Teacher and student must have eaqual number of hidden layers!"
                s_sizes, t_sizes = self.student.model.hidden_sizes, self.teacher.model.hidden_sizes

            self.adapter = Adapter(s_sizes, t_sizes).to(conf.device) # create adapter

            # reset student optimizer with parameters from both student and adapter
            self.student.optimizer = torch.optim.AdamW(
                list(self.student.model.parameters()) + list(self.adapter.parameters()), 
                lr=conf.learning_rate, 
                weight_decay=conf.weight_decay
            )

            self.student.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.student.optimizer, 
                gamma=conf.gamma
            )

    def train_kd(self, train_loader, val_loader, epochs):
        """
        Use knowledge distillation to train student model
        Params:
            train_loader (DataLoader): data loader for training set
            val_loader (DataLoader): data loader for validation set
            epochs (int): number of training epochs
        Returns:
            accuracy_list (list(float)): model accuracy per epoch
            train_loss (list(float)): trainging loss per epoch
            val_loss (list(float)): validation loss per epoch
        """
        self.teacher.model.eval()
        metrics_list, train_loss_list, val_loss_list = [], [], []

        for epoch in range(1, epochs + 1):
            self.student.model.train()
            running_loss = 0.0

            for (data, labels) in train_loader:
                if not data.is_cuda or not labels.is_cuda:
                    data, labels = data.to(self.device), labels.to(self.device)

                self.student.optimizer.zero_grad()
                
                # Forward pass with student model and teacher model
                if self.distillation == "feature":
                    with torch.no_grad():
                        t_emb, t_logits = self.teacher.model.feature_map(data)
                        t_logits.detach()
                    s_emb, s_logits = self.student.model.feature_map(data)
                else:        
                    with torch.no_grad():
                        t_emb, t_logits = self.teacher.model(data)
                        t_logits.detach()
                    s_emb, s_logits = self.student.model(data)

                # Calculate the soft targets loss
                soft_targets_loss = self.kd_loss(s_logits, t_logits)

                # Calculate the true label loss
                label_loss = self.student.criterion(s_logits, labels)



                if self.distillation == "relation":
                    # detach embeddings from model gradients
                    t_emb = t_emb.detach()

                    # if rnn model: convert embedding to shape [B, 2*h_size]
                    if self.type == "rnn":
                        t_emb, s_emb = torch.squeeze(t_emb), torch.squeeze(s_emb)
                    if self.type == "cnn":
                        t_emb, s_emb = torch.flatten(t_emb), torch.flatten(s_emb)

                    rkd_loss = self.rkd_loss(s_emb, t_emb) # calculate rkd loss

                    # Weighted sum of relation loss, soft target loss and true label loss
                    loss = self.rkd_loss_weight * rkd_loss + self.soft_target_loss_weight * soft_targets_loss + self.loss_weight * label_loss
                
                elif self.distillation == "feature":
                    # detach embeddings from model gradients
                    t_emb = [emb.detach() for emb in t_emb]

                    # adapt student feature shapes to teacher feature shapes
                    s_emb = self.adapter(s_emb)

                    feat_loss = self.feature_loss(s_emb, t_emb) # calculate feature loss

                    # Weighted sum of feature loss, soft target loss and true label loss
                    loss = self.feature_loss_weight * feat_loss + self.soft_target_loss_weight * soft_targets_loss + self.loss_weight * label_loss
                
                elif self.distillation == "response":
                    # Weighted sum of soft target loss and true label loss
                    loss = self.soft_target_loss_weight * soft_targets_loss + self.loss_weight * label_loss
                
                else:
                    raise ValueError("distillation must be 'response', 'relation' or 'feature'!")
                

                running_loss += loss.item()
                loss.backward()

                # Perform gradient clipping by value
                nn.utils.clip_grad_value_(self.student.model.parameters(), clip_value=0.2)

                self.student.optimizer.step()
            


            # validate student model
            val_epoch_loss, metrics = self.student.evaluate(val_loader)
            epoch_loss = running_loss / len(train_loader.dataset)
            self.student.scheduler.step()

            metrics_list.append(metrics)
            train_loss_list.append(epoch_loss)
            val_loss_list.append(val_epoch_loss)
        
            print(f"Epoch: {epoch}/{epochs}, Macro-F1 score: {metrics['f1_macro']:.2f}, Micro-F1 score: {metrics['f1_micro']:.2f}, Macro ROC AUC score: {metrics['roc_auc_macro']:.2f}, Train loss: {epoch_loss:.3f}, Val loss: {val_epoch_loss:.3f}")


        if self.distillation == "feature":
            # remove adapter parameters from student optimizer after kd
            self.student.optimizer = torch.optim.AdamW(
                list(self.student.model.parameters()), 
                lr=conf.learning_rate, 
                weight_decay=conf.weight_decay
            )

            self.student.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.student.optimizer, 
                gamma=conf.gamma
            )
        
        return metrics_list, train_loss_list, val_loss_list


class Adapter(nn.Module):
    def __init__(self, s_sizes, t_sizes):
        super().__init__()
        self.adapters = nn.ModuleList(
            nn.Linear(s, t) if s != t else nn.Identity()
            for s, t in zip(s_sizes, t_sizes)
        )

    def forward(self, features):
        return [a(f) for a, f in zip(self.adapters, features)]

