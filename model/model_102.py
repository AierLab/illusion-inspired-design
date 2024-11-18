from .model import *

class Model(Model):
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
        if self.num_classes > 500:
            valid_indices = (labels >= 0) & (labels <= 999)
        else:
            valid_indices = (labels >= 0) & (labels <= 99)
            
        if valid_indices.any():
            valid_images = images[valid_indices]
            valid_labels = labels[valid_indices]
            logits_valid = self(valid_images)  # Forward pass for filtered images

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
        
        if self.num_classes > 500:
            exc_indices = (labels >= 1000) & (labels <= 1001)
        else:
            exc_indices = (labels >= 100) & (labels <= 101)

        if exc_indices.any():
            exc_images = images[exc_indices]
            exc_labels = labels[exc_indices]
            logits_exc = self(exc_images)  # Forward pass for excluded images

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
