from .model import *
import numpy as np
from sklearn.metrics import accuracy_score

class Model(Model):
    def __init__(self, model_name, steps_per_epoch, num_classes, lr):
        super(Model, self).__init__(model_name, steps_per_epoch, num_classes, lr)

        self.save_hyperparameters()

        # Load pre-trained ResNet50 model from timm with the correct number of classes
        self.model = create_model(model_name, pretrained=False, num_classes=num_classes)

        # Loss function and learning rate
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        
        if self.num_classes > 500:
            self.split = 1000
        else:
            self.split = 100
            
    def training_step(self, batch, batch_idx):
        """
        Training step to calculate weighted loss and target task accuracy.

        Steps:
        1. Compute weighted Cross-Entropy Loss for the entire (n+1)*2 label space.
        2. Calculate Top-1 and Top-5 accuracy for the target task (100 classes only).
        3. Log metrics for loss and accuracy.

        Args:
            batch (tuple): A batch containing input images and labels.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss for the batch.
        """
        # Forward pass
        images, labels = batch
        outputs = self(images)  # Full logits (202 classes)

        # Determine predicted target and illusion classes
        preds = outputs.argmax(dim=1)  # Predicted labels from 202 classes
        pred_target_class = preds % self.split  # Target class (0-99)
        pred_illusion_class = preds // self.split  # Illusion class (0 or 1)

        # Determine true target and illusion classes
        true_target_class = labels % self.split  # Target class (0-99)
        true_illusion_class = labels // self.split  # Illusion class (0 or 1)

        # Correctness checks
        correct_target = pred_target_class == true_target_class
        correct_illusion = pred_illusion_class == true_illusion_class

        # Assign weights based on correctness
        weights = torch.ones_like(outputs[:, 0])  # Default weight = 1.0
        weights[correct_target & correct_illusion] = 0.25    # Both target and illusion correct
        weights[correct_target & ~correct_illusion] = 0.5  # Correct target only
        weights[~correct_target & correct_illusion] = 0.75  # Correct illusion only
        weights[~correct_target & ~correct_illusion] = 1  # Both incorrect

        # Compute weighted loss
        log_probs = torch.log_softmax(outputs, dim=1)  # Log probabilities
        loss = -torch.sum(weights * log_probs[range(labels.size(0)), labels]) / labels.size(0)


        # Filter out the target task labels (0-99)
        valid_indices = labels < self.split  # Identify target task labels (0-99)
        if valid_indices.any():
            # Filter outputs and labels for the target task
            valid_labels = labels[valid_indices]
            valid_outputs = outputs[valid_indices]  # Full logits for the filtered target task

            # Top-1 Accuracy
            valid_preds_top1 = valid_outputs.argmax(dim=1) % self.split  # Map predictions to 0-99
            top1_acc = accuracy_score(valid_labels.cpu().numpy(), valid_preds_top1.cpu().numpy())

            # Top-5 Accuracy
            valid_top5_preds = torch.topk(valid_outputs, k=5, dim=1).indices % self.split  # Map predictions to 0-99
            top5_acc = torch.tensor(
                [(label.item() in top5.cpu().numpy()) for label, top5 in zip(valid_labels, valid_top5_preds)]
            ).float().mean().item()
            self.log('train_top1_acc', top1_acc, prog_bar=True)
            self.log('train_top5_acc', top5_acc, prog_bar=True)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        """
        Validation step for (n+1)*2 structure:
        - `val_loss` and `val_acc`: For 100 target classes.
        - `val_all_loss` and `val_all_acc`: For full 202 classes.
        - `val_exc_loss` and `val_exc_acc`: For 2 illusion-specific classes.

        Args:
            batch (tuple): A batch containing input images and labels.
            batch_idx (int): Batch index.

        Returns:
            dict: Validation metrics.
        """
        images, labels = batch
        outputs = self(images)

        # Compute overall loss
        loss_all = self.criterion(outputs, labels)

        # Accuracy for full 202 classes
        preds_all = outputs.argmax(dim=1)
        acc_all_top1 = (preds_all == labels).float().mean().item()

        # Compute Top-5 Accuracy for all 202 classes
        top5_preds_all = torch.topk(outputs, k=5, dim=1).indices
        acc_all_top5 = torch.tensor(
            [(label.item() in top5.cpu().numpy()) for label, top5 in zip(labels, top5_preds_all)]
        ).float().mean().item()

        # Log overall validation loss and Top-1 and Top-5 accuracies (for all labels)
        self.log('val_all_loss', loss_all, prog_bar=True)
        self.log('val_all_top1_acc', acc_all_top1, prog_bar=True)
        self.log('val_all_top5_acc', acc_all_top5, prog_bar=True)

        # Target task metrics
        target_indices = (labels < self.split)  # Identify target task labels (0-99)
        if target_indices.any():
            # Extract outputs for target-specific indices
            target_outputs = outputs[target_indices]  # Full 202-class outputs
            target_labels = labels[target_indices] % self.split  # Map labels to target classes (0-99)

            # Top-1 Accuracy for target task
            preds_target_top1 = target_outputs.argmax(dim=1) % self.split  # Map predictions to 0-99
            acc_target_top1 = accuracy_score(target_labels.cpu().numpy(), preds_target_top1.cpu().numpy())

            # Top-5 Accuracy for target task
            top5_preds_target = torch.topk(target_outputs, k=5, dim=1).indices % self.split  # Map predictions to 0-99
            acc_target_top5 = torch.tensor(
                [(label.item() in top5.cpu().numpy()) for label, top5 in zip(target_labels, top5_preds_target)]
            ).float().mean().item()

            # Compute loss for the target task
            loss_target = self.criterion(target_outputs[:, :self.split], target_labels)  # Only use logits for target classes

            # Log validation metrics for filtered labels 0-99
            self.log('val_loss', loss_target, prog_bar=True)
            self.log('val_top1_acc', acc_target_top1, prog_bar=True)
            self.log('val_top5_acc', acc_target_top5, prog_bar=True)
        else:
            loss_target = None
            acc_target_top1 = None
            acc_target_top5 = None

        # Illusion-specific metrics
        illusion_indices = (labels >= self.split)  # Identify illusion-specific labels (100 and 101)
        if illusion_indices.any():
            # Extract outputs and labels for illusion-specific samples
            illusion_outputs = outputs[illusion_indices]
            illusion_labels = labels[illusion_indices]

            # Determine the predicted class in the full 202-class space
            illusion_preds = illusion_outputs.argmax(dim=1)  # Predicted class indices (0-201)

            # Determine if the predictions belong to the first or second 101 classes
            illusion_class_preds = torch.where(illusion_preds <= self.split, 0, 1)  # 0 for 0-100, 1 for 101-201

            # Map labels to illusion classes (0 for 100, 1 for 101)
            illusion_class_labels = (illusion_labels - self.split)  # 0 for 100, 1 for 101

            # Compute loss using logits for all 202 classes
            loss_illusion = self.criterion(
                illusion_outputs[:, self.split:],  # Use logits for the last 2 classes (illusion classes)
                illusion_class_labels
            )

            # Compute Top-1 accuracy for illusion-specific labels
            acc_illusion = accuracy_score(illusion_class_labels.cpu().numpy(), illusion_class_preds.cpu().numpy())

            # Log metrics
            self.log('val_exc_loss', loss_illusion, prog_bar=True)
            self.log('val_exc_top1_acc', acc_illusion, prog_bar=True)
        else:
            loss_illusion = None
            acc_illusion = None

        # Return all the metrics
        return {
            'val_loss': loss_target,
            'val_top1_acc': acc_target_top1,
            'val_top5_acc': acc_target_top5,
            'val_exc_loss': loss_illusion,
            'val_exc_top1_acc': acc_illusion,
            'val_all_loss': loss_all,
            'val_all_top1_acc': acc_all_top1,
            'val_all_top5_acc': acc_all_top5,
        }

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            dict: Optimizer and scheduler configuration.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.lr / 10,
            max_lr=self.lr,
            step_size_up=self.steps_per_epoch // 2,
            mode='exp_range'
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}