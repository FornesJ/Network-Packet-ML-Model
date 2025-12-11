import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from loss_functions.kd_loss import KD_Loss, RKD_Loss, Feature_Loss
from utils.config import Config
conf = Config()

class KnowledgeDistillation:
    """
    class for Knowledge Distillation
    """
    def __init__(self, 
                teacher, 
                student, 
                device, 
                distillation="result",  
                T=4.0,
                soft_target_loss_weight=0.25,
                rkd_loss_weight=0.1,
                feature_loss_weight=50.0, 
                loss_weight=0.75,
                rnn=False):
        """
        Constructor for KnowledgeDistillation
        Params:
            teacher (Model): large teacher model (pretrained)
            student (Model): light student model
            device (string): device
            distillation (string): define distillation mode: result | feature | relation
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
        self.distillation = distillation
        self.T=T
        self.soft_target_loss_weight = soft_target_loss_weight
        self.rkd_loss_weight = rkd_loss_weight
        self.feature_loss_weight= feature_loss_weight
        self.loss_weight = loss_weight
        self.kd_loss = KD_Loss(T=self.T)
        self.rkd_loss = RKD_Loss()
        self.feature_loss = Feature_Loss()
        self.rnn = rnn

        # if feature distillation: create attention adapter to reshape features from student and teacher
        if self.distillation == "feature":
            # Use adapter to match teacher and student hidden dimensions
            if self.rnn:
                s_sizes, t_sizes = [2 * self.student.model.h_size], [2 * self.teacher.model.h_size]
            else:
                s_sizes, t_sizes = self.student.model.hidden_sizes, self.teacher.model.hidden_sizes

            self.adapter = AttentionAdapter(s_sizes, t_sizes, hidden_dim=t_sizes[-1]).to(conf.device) # create adapter

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
        train_loss = []
        val_loss = []
        accuracy_list = []

        for epoch in range(1, epochs + 1):
            self.student.model.train()
            running_loss = 0.0

            for (data, labels) in train_loader:
                if not data.is_cuda or not labels.is_cuda:
                    data, labels = data.to(self.device), labels.to(self.device)

                self.student.optimizer.zero_grad()
                
                # Forward pass with student model and teacher model
                get_feat = True if self.distillation == "feature" else False
                with torch.no_grad():
                    t_emb, t_logits = self.teacher.model(data, feat=get_feat)
                    t_logits.detach()
                s_emb, s_logits = self.student.model(data, feat=get_feat)

                # Calculate the soft targets loss
                soft_targets_loss = self.kd_loss(s_logits, t_logits)

                # Calculate the true label loss
                label_loss = self.student.criterion(s_logits, labels)

                if self.distillation == "relation":
                    # detach embeddings from model gradients
                    t_emb = t_emb.detach()

                    # if rnn model: convert embedding to shape [B, 2*h_size]
                    if self.rnn:
                        t_emb, s_emb = torch.squeeze(t_emb), torch.squeeze(s_emb)

                    rkd_loss = self.rkd_loss(s_emb, t_emb) # calculate rkd loss

                    # Weighted sum of relation loss, soft target loss and true label loss
                    loss = self.rkd_loss_weight * rkd_loss + self.soft_target_loss_weight * soft_targets_loss + self.loss_weight * label_loss
                
                elif self.distillation == "feature":
                    # detach embeddings from model gradients
                    t_emb = [emb.detach() for emb in t_emb]

                    # adapt student feature shapes to teacher feature shapes
                    pairs = self.adapter(s_emb, t_emb)

                    feat_loss = self.feature_loss(pairs) # calculate feature loss

                    # Weighted sum of feature loss, soft target loss and true label loss
                    loss = self.feature_loss_weight * feat_loss + self.soft_target_loss_weight * soft_targets_loss + self.loss_weight * label_loss
                
                else:
                    # Weighted sum of soft target loss and true label loss
                    loss = self.soft_target_loss_weight * soft_targets_loss + self.loss_weight * label_loss
                
                running_loss += loss.item()
                loss.backward()

                # Perform gradient clipping by value
                nn.utils.clip_grad_value_(self.student.model.parameters(), clip_value=0.2)

                self.student.optimizer.step()
            
            # validate student model
            epoch_loss = running_loss / len(train_loader.dataset)
            val_epoch_loss, acc = self.student.evaluate(val_loader)
            self.student.scheduler.step()

            train_loss.append(epoch_loss)
            val_loss.append(val_epoch_loss)
            accuracy_list.append(acc)

            print(f"Epoch: {epoch}/{epochs}, Accuracy: {100*acc:.2f}%, Train loss: {epoch_loss:.4f}, Val loss: {val_epoch_loss:.4f}")
        
        return accuracy_list, train_loss, val_loss
    


class AttentionAdapter(nn.Module):
    """
    Class AttentionAdapter creates adapter for reshaping student and teacher features
    """
    def __init__(self, student_dims, teacher_dims, hidden_dim=128):
        """
        Constructor for AttentionAdapter
        Parameters:
            student_dims (list(int)): list of shapes for each student feature
            student_dims (list(int)): list of shapes for each teacher feature
            hidden_dim (int): shape of hidden dim
        """
        super().__init__()
        self.student_to_hidden = nn.ModuleList(
            nn.Linear(s, hidden_dim) for s in student_dims
        )
        self.teacher_to_hidden = nn.ModuleList(
            nn.Linear(t, hidden_dim) for t in teacher_dims
        )
        self.hidden_to_output = nn.ModuleList(
            nn.Linear(hidden_dim, teacher_dims[-1])
            for _ in student_dims
        )

    def forward(self, student_feats, teacher_feats):
        """
        Forward method creates attention adapter from student and teacher features
        Parameters:
            student_feats (list(torch.Tensor)): list of student features
            teacher_feats (list(torch.Tensor)): list of teacher features
        Returns:
            adapted_pairs (tuple(torch.Tensor, torch.Tensor)): pairs with outup from student and teahcer projections
        """
        adapted_pairs = []
  
        # Project teacher features into a shared hidden space
        t_embeds = [
            t_proj(t)
            for t, t_proj in zip(teacher_feats, self.teacher_to_hidden)
        ]

        # Process each student layer
        for i, s in enumerate(student_feats):
            B = s.size(0)

            # Project student hidden state
            s_embed = self.student_to_hidden[i](s)

            # Compute attention weights over ALL teacher layers
            # Cosine similarity between (B,H) and each teacher layer (B,H)
            sims = torch.stack([
                torch.cosine_similarity(s_embed, t, dim=1)
                for t in t_embeds
            ], dim=1)                        # (B, num_teacher)

            weights = F.softmax(sims, dim=1) # (B, num_teacher)

            # Expand weights for broadcast in mixing
            weights = weights.unsqueeze(-1)  # (B, num_teacher, 1)
            
            # Stack teacher features (raw, not projected)
            # project teacher feats to same dimension BEFORE stacking
            t_proj_stack = torch.stack(t_embeds, dim=1)  # (B, num_teacher, H)

            # Weighted sum: (B, num_teacher, H) * (B, num_teacher, 1)
            t_mix = (weights * t_proj_stack).sum(dim=1)   # (B, H)

            # Final projection to match teacher layer dimension
            s_out = self.hidden_to_output[i](s_embed)  # (B, T_last)

            adapted_pairs.append((s_out, t_mix))

        return adapted_pairs

