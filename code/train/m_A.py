from .base import *
from .data_indl import trainloader_indl, testloader_indl

# Initialize Wandb logger with a careful naming convention for the model
wandb_logger = WandbLogger(project="illusion_augmented_models", name="model_m_A", log_model=True)

# Experiment name and setup
exp_name = "illusion_augmented_image_classification_model"

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    dirpath="./models/m_A/",
    filename="m_A_{epoch:02d}-{val_acc:.2f}",
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
checkpoint_dir = "./models/m_A/"
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
model = ResNet50(steps_per_epoch=len(trainloader_indl), num_classes=2, lr=1e-3)

# Trainer configuration
trainer = Trainer(
    logger=wandb_logger,
    callbacks=[checkpoint_callback, lr_monitor],
    accelerator="auto",
    devices=2
)

# Model training
trainer.fit(model, train_dataloaders=trainloader_indl, val_dataloaders=testloader_indl)