from .model_102 import *
import numpy as np
from sklearn.metrics import accuracy_score

class Model(Model):
    def __init__(self, model_name, steps_per_epoch, num_classes, lr):
        super(Model, self).__init__(model_name, steps_per_epoch, num_classes, lr)

        self.save_hyperparameters()

        # Load pre-trained ResNet50 model from timm with the correct number of classes
        self.model = create_model(model_name, pretrained=True, num_classes=num_classes)

        # Loss function and learning rate
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch 
        
        self.log_var_dim1 = nn.Parameter(torch.tensor(0.0))
        self.log_var_dim2 = nn.Parameter(torch.tensor(0.0))
    
    def training_step(self, batch, batch_idx):
        '''
        | **Dimension 1 (CIFAR-100 + 1)** | **Dimension 2 (Illusion Switch)** | **Description**                                                    |
        |---------------------------------|-----------------------------------|--------------------------------------------------------------------|
        | 0 - 99                          | n/a                               | CIFAR-100 classes within the "no illusion" category                |
        | n/a                             | 0                                 | Extra class with "no illusion"                                     |
        | n/a                             | 1                                 | Extra class with "illusion present"                                |

        ### Explanation:
        - **Dimension 1**: Covers CIFAR-100 classes `0-99`, with an added `100` for an extra class.
        - **Dimension 2**: The "Illusion Switch," where `0` represents "no illusion" and `1` represents "illusion present," applied specifically to the extra class (`100`).
        '''
        images, labels = batch
        outputs = self(images)

        # Create masks for each dimension based on label conditions
        mask_dim1 = labels < 1000          # Mask for CIFAR-100 classes (Dimension 1)
        mask_dim2 = labels >= 1000         # Mask for extra classes with illusion states (Dimension 2)

        # Compute loss for Dimension 1, applying mask
        loss_dim1 = self.criterion(outputs[mask_dim1], labels[mask_dim1]) if mask_dim1.any() else 0

        # Adjust labels for Dimension 2 and compute loss
        loss_dim2 = self.criterion(outputs[mask_dim2], labels[mask_dim2]) if mask_dim2.any() else 0

        alpha_dim1 = 1/mask_dim1.sum() if mask_dim1.any() else 1
        alpha_dim2 = 1/mask_dim2.sum() if mask_dim2.any() else 1

        # Define weighting factors considering log-variance and data length
        weight_dim1 = alpha_dim1 / (2 * torch.exp(self.log_var_dim1)**2)
        weight_dim2 = alpha_dim2 / (2 * torch.exp(self.log_var_dim2)**2)

        # Combine the two losses with adjusted weighting factors
        loss = (weight_dim1 * loss_dim1 + weight_dim2 * loss_dim2) / (self.log_var_dim1 + self.log_var_dim2)

        # Calculate Top-1 and Top-5 accuracy only for the first 100 classes
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
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_top1_acc', top1_acc, prog_bar=True)
        self.log('train_top5_acc', top5_acc, prog_bar=True)

        return loss
