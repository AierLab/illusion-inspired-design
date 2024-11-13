from .model import *
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
        
        self.num_classes = num_classes
        
        
    
    def training_step(self, batch, batch_idx):
        '''
        | **Dimension 1 (CIFAR-100 + 1)** | **Dimension 2 (Illusion Switch)** | **Description**                                                    |
        |---------------------------------|-----------------------------------|--------------------------------------------------------------------|
        | 0 - (num_classes-1)             | 0 or 1                            | CIFAR-100 classes within the "no illusion" category                |
        | num_classes                     | 0                                 | Extra class with "no illusion"                                     |
        | num_classes                     | 1                                 | Extra class with "illusion present"                                |

        ### Explanation:
        - **Dimension 1**: Covers the CIFAR-100 classes from `0-99` and adds `100` as an extra class.
        - **Dimension 2**: The "Illusion Switch," where `0` represents "no illusion" and `1` represents "illusion present," applied specifically to the extra class (`100`).

        This structure allows the extra class (`100`) to have two variations based on the "illusion" state in Dimension 2.
        '''
        images, labels = batch
        outputs = self(images)

        # Separate output predictions for the two dimensions
        # outputs_dim1 = outputs[:, :101]  # First 101 classes (Dimension 1)
        outputs_dim1 = outputs[:, :-2]  # First 101 classes (Dimension 1)
        # outputs_dim2 = outputs[:, 101:]  # Last 2 classes (Dimension 2)
        outputs_dim2 = outputs[:, -2:]  # Last 2 classes (Dimension 2)

        # Modify labels for the two dimensions without using masks
        labels_dim1 = labels.clone()
        labels_dim2 = labels.clone()

        # For Dimension 1, replace labels >= 101 with 100
        # labels_dim1[labels >= 100] = 100
        labels_dim1[labels >= self.num_classes - 3] = self.num_classes - 3

        # For Dimension 2, replace labels < 100 with 100
        # labels_dim2[labels < 100] = 101 # TODO 100 or 101 need experiments
        labels_dim2[labels < self.num_classes - 3] = self.num_classes - 3 # TODO 100 or 101 need experiments, -1 or -2
        # labels_dim2 = labels_dim2 - 100
        labels_dim2 = labels_dim2 - (self.num_classes - 3)
        
        # print("Label min/max:", labels.min().item(), labels.max().item())
        # print("Label_dim1 min/max:", labels_dim1.min().item(), labels_dim1.max().item())
        # print("Label_dim2 min/max:", labels_dim2.min().item(), labels_dim2.max().item())

        # Apply cross-entropy loss for Dimension 1 and Dimension 2 directly
        loss_dim1 = self.criterion(outputs_dim1, labels_dim1)
        loss_dim2 = self.criterion(outputs_dim2, labels_dim2)

        # Combine the two losses with adjusted weighting factors
        loss = loss_dim1 + loss_dim2

        # Calculate accuracy only for the first 100 classes
        labels_acc = labels[labels < self.num_classes - 3]
        outputs_acc = outputs[labels < self.num_classes - 3][:, :self.num_classes - 3]  # Select only the first 100 classes

        # Top-1 accuracy
        top1_preds = outputs_acc.argmax(dim=1)
        top1_acc = accuracy_score(labels_acc.cpu(), top1_preds.cpu())

        # Top-5 accuracy
        top5_preds = torch.topk(outputs_acc, k=5, dim=1).indices
        top5_acc = torch.tensor([(label in top5) for label, top5 in zip(labels_acc, top5_preds)]).float().mean().item()

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_top1_acc', top1_acc, prog_bar=True)
        self.log('train_top5_acc', top5_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass for all images
        outputs = self(images)

        # Compute loss for all labels (val_all)
        loss_all = self.criterion(outputs, labels)

        # Top-1 and Top-5 accuracies for all labels
        preds_all_top1 = torch.argmax(outputs, dim=1)
        acc_all_top1 = (preds_all_top1 == labels).float().mean()

        top5_preds_all = torch.topk(outputs, k=5, dim=1).indices
        acc_all_top5 = torch.tensor([(label in top5) for label, top5 in zip(labels, top5_preds_all)]).float().mean().item()

        # Filter labels within the range of 0 to 99 (val_B)
        valid_indices = (labels >= 0) & (labels < self.num_classes - 3)

        if valid_indices.any():
            valid_images = images[valid_indices]
            valid_labels = labels[valid_indices]
            logits_valid = self(valid_images)[:, :self.num_classes - 3]  # Forward pass for filtered images

            loss_valid = self.criterion(logits_valid, valid_labels)

            # Top-1 and Top-5 accuracies for filtered labels 0-99
            preds_valid_top1 = torch.argmax(logits_valid, dim=1)
            acc_valid_top1 = (preds_valid_top1 == valid_labels).float().mean()

            top5_preds_valid = torch.topk(logits_valid, k=5, dim=1).indices
            acc_valid_top5 = torch.tensor([(label in top5) for label, top5 in zip(valid_labels, top5_preds_valid)]).float().mean().item()

            # Log validation metrics for filtered labels 0-99
            self.log('val_loss', loss_valid, prog_bar=True)
            self.log('val_top1_acc', acc_valid_top1, prog_bar=True)
            self.log('val_top5_acc', acc_valid_top5, prog_bar=True)
        else:
            loss_valid = None
            acc_valid_top1 = None
            acc_valid_top5 = None

        # Filter labels outside the range of 0 to 99 (val_exc for exclusive labels 100-101)
        exc_indices = (labels >= self.num_classes - 3) & (labels < self.num_classes - 1)

        if exc_indices.any():
            exc_images = images[exc_indices]
            exc_labels = labels[exc_indices] - (self.num_classes - 3)  # Adjust labels to 0 and 1 for the last two classes
            logits_exc = self(exc_images)[:, -2:]  # Forward pass for excluded images

            loss_exc = self.criterion(logits_exc, exc_labels)

            # Top-1 accuracy for exclusive labels 100-101
            preds_exc_top1 = torch.argmax(logits_exc, dim=1)
            acc_exc_top1 = (preds_exc_top1 == exc_labels).float().mean()

            # Log validation metrics for labels 100-101
            self.log('val_exc_loss', loss_exc, prog_bar=True)
            self.log('val_exc_top1_acc', acc_exc_top1, prog_bar=True)
        else:
            loss_exc = None
            acc_exc_top1 = None

        # Log overall validation loss and Top-1 and Top-5 accuracies (for all labels)
        self.log('val_all_loss', loss_all, prog_bar=True)
        self.log('val_all_top1_acc', acc_all_top1, prog_bar=True)
        self.log('val_all_top5_acc', acc_all_top5, prog_bar=True)

        # Return all the metrics
        return {
            'val_loss': loss_valid,
            'val_top1_acc': acc_valid_top1,
            'val_top5_acc': acc_valid_top5,
            'val_exc_loss': loss_exc,
            'val_exc_top1_acc': acc_exc_top1,
            'val_all_loss': loss_all,
            'val_all_top1_acc': acc_all_top1,
            'val_all_top5_acc': acc_all_top5,
        }

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.lr / 10,  # Set the minimum learning rate (can be adjusted)
            max_lr=self.lr,        # Set the maximum learning rate
            step_size_up=self.steps_per_epoch // 2,  # Half of steps per epoch for increasing phase
            mode='exp_range'      # Choose the mode; 'triangular', 'triangular2', or 'exp_range'
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # return {"optimizer": optimizer}
