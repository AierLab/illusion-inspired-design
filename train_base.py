import os
import torch
import wandb
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything
from sklearn.metrics import accuracy_score
from torch.utils.data import ConcatDataset
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from math import sqrt
from timm import create_model

# Seed for reproducibility
seed_everything(42)
    
    
class ResNet50(LightningModule):
    def __init__(self, steps_per_epoch, num_classes=2, lr=1e-3):
        super(ResNet50, self).__init__()
        
        self.save_hyperparameters()

        # Load pre-trained ResNet50 model from timm with the correct number of classes
        self.model = create_model("resnet50", pretrained=False, num_classes=num_classes)

        # Loss function and learning rate
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
    
    def forward(self, x):
        # Forward pass through the entire model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = accuracy_score(labels.cpu(), outputs.argmax(dim=1).cpu())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = accuracy_score(labels.cpu(), outputs.argmax(dim=1).cpu())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=self.steps_per_epoch, epochs=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}







