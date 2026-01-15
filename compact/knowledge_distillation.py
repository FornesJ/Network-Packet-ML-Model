import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_functions.kd_loss import KD_Loss, RKD_Loss, Feature_Loss
from model.model_utils.extract_features import FeatureExtractor
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

        if self.distillation == "relation":
            # add feature extractors to teacher and student model
            self.teacher_extractor = FeatureExtractor(teacher.model, self.type)
            self.student_extractor = FeatureExtractor(student.model, self.type)

        # if feature distillation: create adapter to reshape features from student and teacher
        if self.distillation == "feature":
            # add feature extractors to teacher and student model
            self.teacher_extractor = FeatureExtractor(teacher.model, self.type)
            self.student_extractor = FeatureExtractor(student.model, self.type)

            # Use adapter to match teacher and student hidden dimensions
            if self.type == "rnn":
                assert self.student.model.n_layers == self.teacher.model.n_layers, \
                "Teacher and student must have eaqual number of hidden layers!"
                s_sizes = [2 * self.student.model.h_size for _ in range(self.student.model.n_layers)]
                t_sizes = [2 * self.teacher.model.h_size for _ in range(self.teacher.model.n_layers)]
            elif self.type == "cnn":
                assert self.student.model.conv_layers == self.teacher.model.conv_layers, \
                "Teacher and student must have eaqual number of hidden layers!"
                s_sizes = [ch for ch, _ in self.student.model.filters]
                t_sizes = [ch for ch, _ in self.teacher.model.filters]
            else:
                assert len(self.student.model.hidden_sizes) == len(self.teacher.model.hidden_sizes), \
                "Teacher and student must have eaqual number of hidden layers!"
                s_sizes, t_sizes = self.student.model.hidden_sizes, self.teacher.model.hidden_sizes

            if self.type == "cnn":
                self.adapter = ConvAdapter(s_sizes, t_sizes).to(conf.device) # create adapter
            else:
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

    def relation_kd(self):
        # detach embeddings from model gradients
        t_feat = list(self.teacher_extractor.features.values())
        s_feat = list(self.student_extractor.features.values())
        
        if self.type == "rnn":
            # get hidden states from output
            if isinstance(self.teacher.model.rnn, nn.LSTM):
                t_feat = t_feat[0][1][0]
                s_feat = s_feat[0][1][0]
            else:
                t_feat = t_feat[0][1]
                s_feat = s_feat[0][1]

            # get last hidden state h_n
            t_feat = t_feat.view(self.teacher.model.n_layers, 2, t_feat.shape[1], self.teacher.model.h_size)[-1] # last hidden [2, B, t_hidden]
            s_feat = s_feat.view(self.student.model.n_layers, 2, s_feat.shape[1], self.student.model.h_size)[-1] # last hidden [2, B, s_hidden]

            # reshape hidden state
            t_feat = torch.cat((t_feat[0], t_feat[1]), dim=1) # [B, 2*t_hidden]
            s_feat = torch.cat((s_feat[0], s_feat[1]), dim=1) # [B, 2*s_hidden]

        else:
            # feature from last hidden layer
            t_feat = t_feat[-1]
            s_feat = s_feat[-1]

        t_feat = t_feat.detach() # detach features from model gradients

        if self.type == "cnn":
            # Global Average Pooling teacher and student features
            t_feat, s_feat = t_feat.mean(dim=2), s_feat.mean(dim=2)

        return self.rkd_loss(s_feat, t_feat) # calculate rkd loss
    
    def feature_kd(self):
        # extract student and teacher features
        t_feat = list(self.teacher_extractor.features.values())
        s_feat = list(self.student_extractor.features.values())

        if self.type == "rnn":
            # get hidden states from output
            if isinstance(self.teacher.model.rnn, nn.LSTM):
                t_feat = t_feat[0][1][0]
                s_feat = s_feat[0][1][0]
            else:
                t_feat = t_feat[0][1]
                s_feat = s_feat[0][1]

            # get last hidden state h_n
            t_feat = t_feat.view(self.teacher.model.n_layers, 2, t_feat.shape[1], self.teacher.model.h_size) # last hidden [N, 2, B, t_hidden]
            s_feat = s_feat.view(self.student.model.n_layers, 2, s_feat.shape[1], self.student.model.h_size) # last hidden [N, 2, B, s_hidden]

            # reshape hidden state
            t_feat = torch.cat((t_feat[:,0], t_feat[:,1]), dim=2) # [N, B, 2*t_hidden]
            s_feat = torch.cat((s_feat[:,0], s_feat[:,1]), dim=2) # [N, B, 2*s_hidden]
            
            # detach rnn features/hidden states
            new_t_feat = [feat.detach() for feat in t_feat]
            new_s_feat = [feat for feat in s_feat]

            t_feat = new_t_feat
            s_feat = new_s_feat
        else:
            # detach features from model gradients
            t_feat = [feat.detach() for feat in t_feat]
        
        # adapt student feature to teacher feature shapes
        s_feat = self.adapter(s_feat)

        if self.type == "cnn":
            # Adaptive Avg. Pooling teacher features down to student length
            t_feat = [F.adaptive_avg_pool1d(t, s.size(-1)) for (t, s) in zip(t_feat, s_feat)]
        
        return self.feature_loss(s_feat, t_feat) # calculate feature loss

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
                if self.distillation == "relation" or self.distillation == "feature":
                    self.teacher_extractor.clear()
                    self.student_extractor.clear()

                # Forward pass with student model and teacher model   
                with torch.no_grad():
                    t_logits = self.teacher.model(data)
                    t_logits.detach()
                s_logits = self.student.model(data)

                # Calculate the soft targets loss
                soft_targets_loss = self.kd_loss(s_logits, t_logits)

                # Calculate the true label loss
                label_loss = self.student.criterion(s_logits, labels)


                if self.distillation == "relation":
                    rkd_loss = self.relation_kd() # relation distillation loss

                    # Weighted sum of relation loss, soft target loss and true label loss
                    loss = self.rkd_loss_weight * rkd_loss + self.soft_target_loss_weight * soft_targets_loss + self.loss_weight * label_loss
                
                elif self.distillation == "feature":
                    feat_loss = self.feature_kd() # feature distillation loss

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

        if self.distillation == "relation" or self.distillation == "feature":
            self.teacher_extractor.remove_hooks()
            self.student_extractor.remove_hooks()
        
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
    
class ConvAdapter(nn.Module):
    def __init__(self, s_ch, t_ch):
        super().__init__()
        self.adapters = nn.ModuleList(
            nn.Conv1d(s, t, kernel_size=1) if s != t else nn.Identity()
            for s, t in zip(s_ch, t_ch)
        )

    def forward(self, features):
        return [a(f) for a, f in zip(self.adapters, features)]