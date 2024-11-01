from .model import *
import numpy as np

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
        
        self.num_classes = num_classes
        
        
    
    def training_step(self, batch, batch_idx):
        '''
        | **Dimension 1 (CIFAR-100 + 1)** | **Dimension 2 (Illusion Switch)** | **Description**                                                    |
        |---------------------------------|-----------------------------------|--------------------------------------------------------------------|
        | 0 - (num_classes-1)                          | 0 or 1                            | CIFAR-100 classes within the "no illusion" category                |
        | num_classes                             | 0                                 | Extra class with "no illusion"                                     |
        | num_classes                             | 1                                 | Extra class with "illusion present"                                |

        ### Explanation:
        - **Dimension 1**: Covers the CIFAR-100 classes from `0-99` and adds `100` as an extra class.
        - **Dimension 2**: The "Illusion Switch," where `0` represents "no illusion" and `1` represents "illusion present," applied specifically to the extra class (`100`).

        This structure allows the extra class (`100`) to have two variations based on the "illusion" state in Dimension 2.
        '''
        images, labels = batch
        outputs = self(images)

        # Separate output predictions for the two dimensions
        # outputs_dim1 = outputs[:, :101]  # First 101 classes (Dimension 1)
        outputs_dim1 = outputs[:, :self.num_classes+1]  # First 101 classes (Dimension 1)
        # outputs_dim2 = outputs[:, 101:]  # Last 2 classes (Dimension 2)
        outputs_dim2 = outputs[:, self.num_classes+1:]  # Last 2 classes (Dimension 2)

        # Modify labels for the two dimensions without using masks
        labels_dim1 = labels.clone()
        labels_dim2 = labels.clone()

        # For Dimension 1, replace labels >= 101 with 100
        # labels_dim1[labels >= 100] = 100
        labels_dim1[labels >= self.num_classes] = self.num_classes

        # For Dimension 2, replace labels < 100 with 100
        # labels_dim2[labels < 100] = 101 # TODO 100 or 101 need experiments
        labels_dim2[labels < self.num_classes] = self.num_classes+1 # TODO 100 or 101 need experiments
        # labels_dim2 = labels_dim2 - 100
        labels_dim2 = labels_dim2 - self.num_classes
        
        # print("Label min/max:", labels.min().item(), labels.max().item())
        # print("Label_dim1 min/max:", labels_dim1.min().item(), labels_dim1.max().item())
        # print("Label_dim2 min/max:", labels_dim2.min().item(), labels_dim2.max().item())

        # Apply cross-entropy loss for Dimension 1 and Dimension 2 directly
        loss_dim1 = self.criterion(outputs_dim1, labels_dim1)
        loss_dim2 = self.criterion(outputs_dim2, labels_dim2)
        
        alpha_dim1 = 1/len(outputs_dim1)
        alpha_dim2 = 1/len(outputs_dim2)

        # Define weighting factors considering log-variance and data length
        weight_dim1 = alpha_dim1 / (2 * torch.exp(self.log_var_dim1)**2)
        weight_dim2 = alpha_dim2 / (2 * torch.exp(self.log_var_dim2)**2)

        # Combine the two losses with adjusted weighting factors
        loss = weight_dim1 * loss_dim1 + weight_dim2 * loss_dim2 + (self.log_var_dim1 + self.log_var_dim2)

        # Calculate accuracy only for the first 100 classes
        labels_acc = labels[labels < self.num_classes]
        outputs_acc = outputs[labels < self.num_classes][:, :self.num_classes]  # Select only the first 100 classes
        preds = outputs_acc.argmax(dim=1)
        acc = accuracy_score(labels_acc.cpu(), preds.cpu())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        
        # Forward pass for all images
        outputs = self(images)

        # Compute loss and accuracy for all labels (val_all)
        loss_all = self.criterion(outputs, labels)
        preds_all = outputs.argmax(dim=1)
        acc_all = (preds_all == labels).float().mean()

        # Filter labels within the range of 0 to 99 (val_B)
        valid_indices = (labels >= 0) & (labels <= self.num_classes-1)
        
        if valid_indices.any():
            valid_labels = labels[valid_indices]
            logits_valid = outputs[valid_indices][:, :self.num_classes]  # Select first 100 classes
            
            loss_valid = self.criterion(logits_valid, valid_labels)
            preds_valid = logits_valid.argmax(dim=1)
            acc_valid = (preds_valid == valid_labels).float().mean()
            
            # Log validation metrics for filtered labels 0-9
            self.log('val_loss', loss_valid, prog_bar=True)
            self.log('val_acc', acc_valid, prog_bar=True)
        else:
            loss_valid = None
            acc_valid = None

        # Filter labels for the extra classes (val_exc)
        exc_indices = (labels >= self.num_classes) & (labels <= self.num_classes+1)
        if exc_indices.any():
            exc_labels = labels[exc_indices] - self.num_classes
            logits_exc = outputs[exc_indices][:, self.num_classes+1:]  # Select the last 2 classes
            
            loss_exc = self.criterion(logits_exc, exc_labels)
            preds_exc = logits_exc.argmax(dim=1)
            acc_exc = (preds_exc == exc_labels).float().mean()
            
            # Log validation metrics for labels 100-101
            self.log('val_exc_loss', loss_exc, prog_bar=True)
            self.log('val_exc_acc', acc_exc, prog_bar=True)
        else:
            loss_exc = None
            acc_exc = None

        # Log overall validation loss and accuracy
        self.log('val_all_loss', loss_all, prog_bar=True)
        self.log('val_all_acc', acc_all, prog_bar=True)


        # Return all the metrics
        return {
            'val_loss': loss_valid,
            'val_acc': acc_valid,
            'val_exc_loss': loss_exc,
            'val_exc_acc': acc_exc,
            'val_all_loss': loss_all,
            'val_all_acc': acc_all,
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=self.steps_per_epoch, epochs=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}