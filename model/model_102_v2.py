from .model import *

class ModifiedModel(Model):
    def training_step(self, batch, batch_idx):
        '''
        | **Dimension 1 (CIFAR-100 + 1)** | **Dimension 2 (Illusion Switch)** | **Description**                                                    |
        |---------------------------------|-----------------------------------|--------------------------------------------------------------------|
        | 0 - 99                          | 0 or 1                            | CIFAR-100 classes within the "no illusion" category                |
        | 100                             | 0                                 | Extra class with "no illusion"                                     |
        | 100                             | 1                                 | Extra class with "illusion present"                                |

        ### Explanation:
        - **Dimension 1**: Covers the CIFAR-100 classes from `0-99` and adds `100` as an extra class.
        - **Dimension 2**: The "Illusion Switch," where `0` represents "no illusion" and `1` represents "illusion present," applied specifically to the extra class (`100`).

        This structure allows the extra class (`100`) to have two variations based on the "illusion" state in Dimension 2.
        '''
        images, labels = batch
        outputs = self(images)

        # Separate output predictions for the two dimensions
        outputs_dim1 = outputs[:, :101]  # First 101 classes (Dimension 1)
        outputs_dim2 = outputs[:, 101:]  # Last 2 classes (Dimension 2)

        # Modify labels for the two dimensions without using masks
        labels_dim1 = labels.clone()
        labels_dim2 = labels.clone()

        # For Dimension 1, replace labels >= 100 with 100
        labels_dim1[labels >= 100] = 100

        # For Dimension 2, replace labels < 100 with 100
        labels_dim2[labels < 100] = 100 # TODO or 101 need experiments

        # Apply cross-entropy loss for Dimension 1 and Dimension 2 directly
        loss_dim1 = F.cross_entropy(outputs_dim1, labels_dim1)
        loss_dim2 = F.cross_entropy(outputs_dim2, labels_dim2 - 100)

        # Combine the two losses
        total_loss = loss_dim1 + loss_dim2

        # Calculate accuracy only for the first 100 classes
        acc_labels = labels[labels < 100]
        acc_outputs = outputs[labels < 100, :100].argmax(dim=1)
        acc = accuracy_score(acc_labels.cpu(), acc_outputs.cpu()) if acc_labels.numel() > 0 else 0

        # Log the metrics
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_acc_100_classes', acc, prog_bar=True)

        return total_loss


    def validation_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass for all images
        outputs = self(images)

        # Compute loss and accuracy for all labels (val_all)
        loss_all = self.criterion(outputs, labels)
        preds_all = torch.argmax(outputs, dim=1)
        acc_all = (preds_all == labels).float().mean()

        # Filter labels within the range of 0 to 9 (val_B)
        valid_indices = (labels >= 0) & (labels <= 99)

        if valid_indices.any():
            valid_images = images[valid_indices]
            valid_labels = labels[valid_indices]
            logits_valid = self(valid_images)  # Forward pass for filtered images

            loss_valid = F.cross_entropy(logits_valid, valid_labels)
            preds_valid = torch.argmax(logits_valid, dim=1)
            acc_valid = (preds_valid == valid_labels).float().mean()

            # Log validation metrics for filtered labels 0-9
            self.log('val_loss', loss_valid, prog_bar=True)
            self.log('val_acc', acc_valid, prog_bar=True)

        else:
            loss_valid = None
            acc_valid = None

        # Filter labels outside the range of 0 to 9 (val_exc for exclusive labels 10-11)
        exc_indices = (labels >= 101) & (labels <= 102)

        if exc_indices.any():
            exc_images = images[exc_indices]
            exc_labels = labels[exc_indices]
            logits_exc = self(exc_images)  # Forward pass for excluded images

            loss_exc = F.cross_entropy(logits_exc, exc_labels)
            preds_exc = torch.argmax(logits_exc, dim=1)
            acc_exc = (preds_exc == exc_labels).float().mean()

            # Log validation metrics for labels 10-11
            self.log('val_exc_loss', loss_exc, prog_bar=True)
            self.log('val_exc_acc', acc_exc, prog_bar=True)

        else:
            loss_exc = None
            acc_exc = None

        # Log overall validation loss and accuracy (for all labels)
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