from .model import *

class ModifiedModel(Model):
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
        exc_indices = (labels >= 100) & (labels <= 101)

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