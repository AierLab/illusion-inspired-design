import numpy as np
import torch
import model as m
import data
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from hydra import initialize, compose
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
from pytorch_lightning import Trainer
import argparse

wandb.login(key="ef983325a8df31bd8b4f48223851b7e920c3f8ae")

@hydra.main(version_base=None, config_path="config/train", config_name="default")
def train(cfg: DictConfig):    
    Model = m.get_model(model_name=cfg.model.task)
    train_dataloader, test_dataloader = data.get_dataloader(dataset_name=cfg.data.name)
    
    # Initialize Wandb logger with a careful naming convention for the model
    wandb_logger = WandbLogger(
        project=cfg.trainer.logger_project,
        save_dir=cfg.trainer.logger_save_dir,
        name=cfg.trainer.logger_name,
        log_model=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Early stopping callback
    if cfg.model.task == "m_VAE":
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=cfg.trainer.checkpoint_dir,
            filename=cfg.model.task + "_{epoch:02d}-{val_loss:.3f}",
            save_top_k=cfg.trainer.save_top_k,
            mode="min",
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",  # Metric to monitor for VAE
            patience=cfg.trainer.patience,  # Number of epochs with no improvement
            mode="min"  # "min" because we want to minimize validation loss
        )
    else:
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor=cfg.trainer.monitor_metric,
            dirpath=cfg.trainer.checkpoint_dir,
            filename=cfg.model.task + "_{epoch:02d}-{val_top1_acc:.3f}",
            save_top_k=cfg.trainer.save_top_k,
            mode="max",
        )
        early_stopping = EarlyStopping(
            monitor="val_top1_acc",  # Metric to monitor
            patience=cfg.trainer.patience,  # Number of epochs with no improvement
            mode="max"  # "min" or "max"
        )


    # Path for latest checkpoint
    latest_checkpoint = None
    if os.path.exists(cfg.trainer.checkpoint_dir):
        checkpoints = os.listdir(cfg.trainer.checkpoint_dir)
        if checkpoints:
            latest_checkpoint = max(
                [os.path.join(cfg.trainer.checkpoint_dir, ckpt) for ckpt in checkpoints],
                key=os.path.getctime
            )
    

    # Training model instance
    if cfg.trainer.use_pretrained and latest_checkpoint:
        model = Model.load_from_checkpoint(
            latest_checkpoint
            )
    elif cfg.model.task == "m_VAE":
        model = Model(
            model_name=cfg.model.name,
            latent_dim=cfg.model.latent_dim,
            lr=cfg.model.lr,
            data_name = cfg.data.name
        )
    elif cfg.model.task == "m_VAE_adapter":
        model = Model(
            cfg.model.name,
            latent_dim=cfg.model.latent_dim,
            num_classes=cfg.model.num_classes,
            lr=cfg.model.lr,
            vector_shift_path=cfg.trainer.vector_shift_path,
            ckpt_path=cfg.model.ckpt_path
        ) 
    elif cfg.model.task == "m_X_comp" or cfg.model.task == "m_X_comp_reverse":
        model = Model(
            cfg.model.name,
            steps_per_epoch=len(train_dataloader),
            num_classes=cfg.model.num_classes,
            lr=cfg.model.lr,
            ckpt_path_A=cfg.model.ckpt_path_A,
            ckpt_path_B=cfg.model.ckpt_path_B
        )
    else:
        model = Model(
            cfg.model.name,
            steps_per_epoch=len(train_dataloader),
            num_classes=cfg.model.num_classes,
            lr=cfg.model.lr
        )

    # Trainer configuration
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        accelerator=cfg.trainer.accelerator,
        strategy="ddp",
        devices=cfg.trainer.devices
    )

    # Model training
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
    


def calculate_and_save_vector_shift(cfg, model):
    """
    This function calculates the mean vector shift for style transfer by loading specific datasets.
    """
    # Load specific dataloaders based on task
    if cfg.data.name.endswith("cifar100"):
        indl_loader, _ = data.get_dataloader(dataset_name="indl32")
    else:
        indl_loader, _ = data.get_dataloader(dataset_name="indl224")

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Calculate mean latent vectors for two classes (assuming binary classes 0 and 1 for indl)
    class_latents = {0: [], 1: []}
    with torch.no_grad():
        for images, labels in indl_loader:
            images, labels = images.to(device), labels.to(device)
            _, mu, _ = model(images)
            
            for i, label in enumerate(labels):
                class_latents[label.item()].append(mu[i].cpu().numpy())
                
    # Calculate mean vector for each class and the vector shift
    mean_vector_0 = np.mean(class_latents[0], axis=0)
    mean_vector_1 = np.mean(class_latents[1], axis=0)
    vector_shift = mean_vector_0 - mean_vector_1
    
    # Save the vector shift
    vector_shift_path = os.path.join(cfg.trainer.checkpoint_dir, "vector_shift.pth")
    torch.save(torch.tensor(vector_shift), vector_shift_path)
    cfg.trainer.vector_shift_path = vector_shift_path  # Store for future use
    print(f"Vector shift saved to: {vector_shift_path}")



def main():
    # Seed for reproducibility
    seed_everything(42)
    
    args = parse_args()
    config_name = args.config_name
        
    # Initialize Hydra and compose the config
    with initialize(config_path="config/train", version_base=None):
        cfg = compose(config_name=config_name)
        train(cfg)  # Call `train` with the composed config
        
        # Path for latest checkpoint
        if cfg.model.task == "m_VAE":
            Model = m.get_model(model_name=cfg.model.task)
            latest_checkpoint = None
            if os.path.exists(cfg.trainer.checkpoint_dir):
                checkpoints = os.listdir(cfg.trainer.checkpoint_dir)
                if checkpoints:
                    latest_checkpoint = max(
                        [os.path.join(cfg.trainer.checkpoint_dir, ckpt) for ckpt in checkpoints],
                        key=os.path.getctime
                    )
            model = Model.load_from_checkpoint(
                latest_checkpoint
                )
            calculate_and_save_vector_shift(cfg, model)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config_name', type=str, help='Name of the configuration')
    return parser.parse_args()


if __name__ == "__main__":
    main()
    