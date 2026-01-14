import torch
import torch.nn as nn
import torch.nn.functional as F

class KD_Loss(nn.Module):
    def __init__(self, T=4.0, alpha=0.5):
        super().__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits):
        #Soften the student logits by applying softmax first and log() second
        soft_targets = nn.functional.softmax(teacher_logits / self.T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / self.T, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = torch.sum(
            soft_targets * ((soft_targets + 1e-8).log() - soft_prob)
        ) / (self.alpha * (self.T**2))

        return soft_targets_loss
    

class RKD_Loss(nn.Module):
    def __init__(self, distance_weight=25.0, angle_weight=50.0):
        super().__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight

    def pairwise_distances(self, x):
        # x: [B, D]
        diff = x.unsqueeze(0) - x.unsqueeze(1)   # [B, B, D]
        dist = torch.norm(diff, dim=2)           # [B, B]
        return dist
    
    def compute_angle(self, x):
        # Vector difference between pairs
        diff = x.unsqueeze(0) - x.unsqueeze(1)        # [B,B,D]
        normed = F.normalize(diff, p=2, dim=2)        # unit vectors
        # angle(i,j,k) ~ dot product of ij and ik
        angles = torch.bmm(normed, normed.transpose(1, 2))
        return angles

    def forward(self, student, teacher):
        """
        Forward method to RKD loss
        Args:
            student (torch.Tensor): [B, D] student embeddings
            teacher (torch.Tensor): [B, D] teacher embeddings
        Returns:
            rkd loss (float): weighted sum of distance loss and angle loss
        """

        # RKD Distance Loss
        with torch.no_grad():
            t_dist = self.pairwise_distances(teacher)
            mean_t = t_dist[t_dist > 0].mean()
            t_dist = t_dist / mean_t

        s_dist = self.pairwise_distances(student)
        mean_s = s_dist[s_dist > 0].mean()
        s_dist = s_dist / (mean_s + 1e-8)

        dist_loss = F.smooth_l1_loss(s_dist, t_dist)
        

        # RKD Angle Loss
        with torch.no_grad():
            t_angle = self.compute_angle(teacher)

        s_angle = self.compute_angle(student)
        angle_loss = F.smooth_l1_loss(s_angle, t_angle)

        return self.distance_weight * dist_loss + self.angle_weight * angle_loss
    
    
class Feature_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_features, teacher_features):
        loss = 0
        for s, t in zip(student_features, teacher_features):
            loss += F.mse_loss(s, t)
        return loss