import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from pytorch_lightning import Trainer
from .model_vae import ResNetVAE

class AdapterModel(LightningModule):
    def __init__(self, vae_ckpt_path, latent_dim, num_classes, lr=1e-3, shift_path=None):
        super(AdapterModel, self).__init__()
        
        # Load VAE model from checkpoint
        self.vae = ResNetVAE.load_from_checkpoint(vae_ckpt_path)
        self.vae.eval()  # Ensure VAE is in evaluation mode

        # Adapter architecture
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.shift_path = shift_path

        # Load the shift vector if shift_path is provided
        if shift_path:
            self.shift_vector = torch.load(shift_path)  # Assuming shift_vector is saved as a tensor
        else:
            self.shift_vector = torch.zeros(latent_dim)  # Default to zero shift if not provided

    def forward(self, images):
        # Get latent vector z from VAE
        with torch.no_grad():
            _, z, _ = self.vae(images)  # Assuming VAE outputs (reconstruction, mean, log_var)
        
        # Apply shift to latent vector
        z_shifted = z + self.shift_vector.to(z.device)
        
        # Pass shifted vector through the adapter
        return self.adapter(z_shifted)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
