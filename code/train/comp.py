from src.train.base import *
from data_cifar10 import *

def main():

    # Define the directory where the checkpoints are saved
    checkpoint_dir = "./models/"  # Directory containing 'm_A' and 'm_B'

    # Function to find the checkpoint with the highest val_acc for a specific model (e.g., m_A, m_B)
    def find_best_checkpoint(model_folder, directory):
        best_val_acc = -float('inf')
        best_checkpoint = None
        model_dir = os.path.join(directory, model_folder)

        # Loop through all files in the model's directory
        for file in os.listdir(model_dir):
            if "val_acc" in file:
                # Extract the validation accuracy from the filename (e.g., 'm_A_epoch=21-val_acc=0.86.ckpt')
                val_acc_str = file.split("val_acc=")[1].split(".ckpt")[0]
                val_acc = float(val_acc_str)

                # Check if this checkpoint has the highest val_acc so far
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_checkpoint = os.path.join(model_dir, file)

        return best_checkpoint

    # Find the best checkpoints for model A and model B
    ckpt_path_A = find_best_checkpoint("m_A", checkpoint_dir)
    ckpt_path_B = find_best_checkpoint("m_B", checkpoint_dir)


    # Initialize Wandb logger with a careful naming convention for the model
    wandb_logger = WandbLogger(project="illusion_augmented_models", name="composition", log_model=True)

    # Experiment name and setup
    exp_name = "illusion_augmented_image_classification_model"

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="./models/comp/",
        filename="comp_{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Early stopping callback to stop training when the monitored metric has stopped improving
    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # or any other metric you are monitoring
        patience=10,  # Number of epochs to wait after the last improvement
        mode='min',  # 'min' for loss, 'max' for accuracy
        verbose=True
    )

    # Path for latest checkpoint
    checkpoint_dir = "./models/comp/"
    latest_checkpoint = None

    # Check if a checkpoint exists
    if os.path.exists(checkpoint_dir):
        checkpoints = os.listdir(checkpoint_dir)
        if checkpoints:
            latest_checkpoint = max(
                [os.path.join(checkpoint_dir, ckpt) for ckpt in checkpoints],
                key=os.path.getctime
            )

    # Training model instance (ensure the correct model class is used)
    model = CompositionModel(len(trainloader_cifar10), ckpt_path_A, ckpt_path_B)

    # Update trainer configuration to include the EarlyStopping callback
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        accelerator="auto",
        devices=1  # Adjust this according to the number of devices (e.g., GPUs)
    )

    # Model training
    trainer.fit(model, train_dataloaders=trainloader_cifar10, val_dataloaders=testloader_cifar10)
