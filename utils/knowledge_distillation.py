import torch
import torch.nn as nn

class KnowledgeDistillation:
    def __init__(self, teacher, student, device, T=2, soft_target_loss_weight=0.25, loss_weight=0.75):
        self.teacher = teacher
        self.student = student
        self.device = device
        self.T = T
        self.soft_target_loss_weight = soft_target_loss_weight
        self.loss_weight = loss_weight

    def train_kd(self, train_loader, val_loader, epochs):
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

                # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                with torch.no_grad():
                    _, teacher_logits = self.teacher.model(data)

                # Forward pass with the student model
                _, student_logits = self.student.model(data)

                #Soften the student logits by applying softmax first and log() second
                soft_targets = nn.functional.softmax(teacher_logits / self.T, dim=-1)
                soft_prob = nn.functional.log_softmax(student_logits / self.T, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                soft_targets_loss = torch.sum(
                    soft_targets * ((soft_targets + 1e-8).log() - soft_prob)
                ) / (soft_prob.size(0) * (self.T**2))

                # Calculate the true label loss
                label_loss = self.student.criterion(student_logits, labels)

                # Weighted sum of the two losses
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
