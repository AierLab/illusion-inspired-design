from .model_102_v2 import Model as BaseModel
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn

class DependencyModel:
    def __init__(self):
        self.fitted = False

    def fit(self, loss_dim1_values, loss_dim2_values):
        # Convert accumulated loss values to PyTorch tensors
        loss_dim1_values = torch.tensor(loss_dim1_values)
        loss_dim2_values = torch.tensor(loss_dim2_values)
        epsilon = 1e-8
        log_loss_dim1_values = torch.log(loss_dim1_values + epsilon)
        log_loss_dim2_values = torch.log(loss_dim2_values + epsilon)
        data = torch.stack((log_loss_dim1_values, log_loss_dim2_values), dim=1)
        self.mu = torch.mean(data, dim=0)
        self.cov = torch.cov(data.T)
        self.fitted = True

    def predict_dependency_loss(self, loss_dim1, loss_dim2):
        if not self.fitted:
            return torch.tensor(0.0, device=loss_dim1.device)
        epsilon = 1e-8
        log_loss_dim1 = torch.log(loss_dim1 + epsilon)
        log_loss_dim2 = torch.log(loss_dim2 + epsilon)
        mu_x, mu_y = self.mu
        sigma_xx = self.cov[0, 0]
        sigma_xy = self.cov[0, 1]
        sigma_yy = self.cov[1, 1]
        # Ensure positive variance
        if sigma_xx <= 0 or sigma_yy <= 0:
            return torch.tensor(0.0, device=loss_dim1.device)
        sigma_cond = sigma_yy - sigma_xy**2 / sigma_xx
        if sigma_cond <= 0:
            return torch.tensor(0.0, device=loss_dim1.device)
        # Compute conditional mean
        mu_cond = mu_y + sigma_xy / sigma_xx * (log_loss_dim1 - mu_x)
        # Compute negative log-likelihood
        nll = 0.5 * torch.log(2 * torch.pi * sigma_cond) + 0.5 * ((log_loss_dim2 - mu_cond)**2 / sigma_cond)
        return nll



class Model(BaseModel):
    def __init__(self, model_name, steps_per_epoch, num_classes, lr, update_interval=50):
        super().__init__(model_name, steps_per_epoch, num_classes, lr)

        # Dependency model to capture dependency between loss_dim1 and loss_dim2
        self.dependency_model = DependencyModel()
        self.update_interval = update_interval  # How often to update the model
        self.loss_dim1_values = []
        self.loss_dim2_values = []

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)

        # Create masks for each dimension based on label conditions
        mask_dim1 = labels < self.num_classes - 2
        mask_dim2 = labels >= self.num_classes - 2

        # Compute losses for each dimension
        loss_dim1 = self.criterion(outputs[mask_dim1], labels[mask_dim1]) if mask_dim1.any() else torch.tensor(0.0, dtype=outputs.dtype, device=outputs.device)
        loss_dim2 = self.criterion(outputs[mask_dim2], labels[mask_dim2]) if mask_dim2.any() else torch.tensor(0.0, dtype=outputs.dtype, device=outputs.device)

        # Accumulate losses for periodic model update
        if mask_dim1.any() and mask_dim2.any():
            self.loss_dim1_values.append(loss_dim1.item())
            self.loss_dim2_values.append(loss_dim2.item())

        # Periodically fit the dependency model
        if (batch_idx + 1) % self.update_interval == 0 and len(self.loss_dim1_values) > 10:
            self.dependency_model.fit(self.loss_dim1_values, self.loss_dim2_values)
            # Clear accumulated data
            self.loss_dim1_values = []
            self.loss_dim2_values = []

        # Compute third loss without detaching from the computational graph
        third_loss = self.dependency_model.predict_dependency_loss(loss_dim1, loss_dim2)

        # Combine losses
        total_loss = loss_dim1 + loss_dim2 + third_loss

        # Calculate Top-1 and Top-5 accuracy for the first num_classes - 2 classes
        if mask_dim1.any():
            labels_acc = labels[mask_dim1]
            outputs_acc = outputs[mask_dim1]

            # Top-1 accuracy
            top1_preds = outputs_acc.argmax(dim=1)
            top1_acc = accuracy_score(labels_acc.cpu(), top1_preds.cpu())

            # Top-5 accuracy
            top5_preds = torch.topk(outputs_acc, k=5, dim=1).indices
            top5_acc = torch.tensor([(label in top5) for label, top5 in zip(labels_acc, top5_preds)]).float().mean().item()
        else:
            top1_acc = 0
            top5_acc = 0

        # Log metrics
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_top1_acc', top1_acc, prog_bar=True)
        self.log('train_top5_acc', top5_acc, prog_bar=True)

        return total_loss
