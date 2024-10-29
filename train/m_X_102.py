from ._base import *
from data.data102_indl_and_cifar100 import trainloader_combined, testloader_combined
from model.model_102 import ModifiedModel

def main(model_name):
    # Initialize Wandb logger with a careful naming convention for the model
    wandb_logger = WandbLogger(project="illusion_augmented_models", save_dir="tmp", name=f"model_m_X_102_{model_name}", log_model=True)
    # Experiment name and setup
    exp_name = "illusion_augmented_image_classification_model"

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=f"./tmp/models/m_X_102_{model_name}/",
        filename="{epoch:02d}-{val_acc:.2f}",
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
    checkpoint_dir = f"./tmp/models/m_X_102_{model_name}/"
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
    model = ModifiedModel(model_name, steps_per_epoch=len(trainloader_combined), num_classes=102, lr=1e-4)

    # Trainer configuration
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        accelerator="auto",
        devices=1
    )

    # Model training
    trainer.fit(model, train_dataloaders=trainloader_combined, val_dataloaders=testloader_combined)