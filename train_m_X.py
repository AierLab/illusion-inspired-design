from train_base import *
from data_indl_and_cifar10 import trainloader_combined, testloader_combined

class ResNet50Modified(ResNet50):
    def __init__(self, steps_per_epoch, num_classes=12, lr=1e-3):
        super().__init__(steps_per_epoch, num_classes=num_classes, lr=lr)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        
        # Forward pass for all images
        outputs = self(images)

        # Compute loss and accuracy for all labels (val_all)
        loss_all = self.criterion(outputs, labels)
        preds_all = torch.argmax(outputs, dim=1)
        acc_all = (preds_all == labels).float().mean()

        # Filter labels within the range of 0 to 9 (val_B)
        valid_indices = (labels >= 0) & (labels <= 9)

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
        exc_indices = (labels >= 10) & (labels <= 11)

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


def main():

    # Initialize Wandb logger with a careful naming convention for the model
    wandb_logger = WandbLogger(project="illusion_augmented_models", name="model_m_X", log_model=True)
    # Experiment name and setup
    exp_name = "illusion_augmented_image_classification_model"

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="./models/m_X/",
        filename="m_X_{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Early stopping callback to stop training when the monitored metric has stopped improving
    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # or any other metric you are monitoring
        patience=3,  # Number of epochs to wait after the last improvement
        mode='min',  # 'min' for loss, 'max' for accuracy
        verbose=True
    )

    # Path for latest checkpoint
    checkpoint_dir = "./models/m_X/"
    latest_checkpoint = None

    # Check if a checkpoint exists
    if os.path.exists(checkpoint_dir):
        checkpoints = os.listdir(checkpoint_dir)
        if checkpoints:
            latest_checkpoint = max(
                [os.path.join(checkpoint_dir, ckpt) for ckpt in checkpoints],
                key=os.path.getctime
            )

    # Training model instance
    model = ResNet50Modified(steps_per_epoch=len(trainloader_combined), num_classes=12, lr=1e-3)

    # Trainer configuration
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        accelerator="auto",
        devices=1
    )

    # Model training
    trainer.fit(model, train_dataloaders=trainloader_combined, val_dataloaders=testloader_combined)