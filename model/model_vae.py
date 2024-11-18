from ._base import *
import torch
from torch import nn
from torch.nn import functional as F
from timm import create_model
from pytorch_lightning import LightningModule
import torchvision.utils as vutils

class ResNetVAE(LightningModule):
    def __init__(self, model_name, latent_dim, lr, data_name):
        super(ResNetVAE, self).__init__()
        
        self.save_hyperparameters()
        
        # Load pre-trained ResNet as the encoder backbone
        self.encoder = create_model(model_name, pretrained=False)
        self.encoder.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Define the output dimensions of the encoder
        encoder_output_dim = self.encoder.num_features  # Typically 2048 for ResNet50
        
        # Latent space mappings
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(encoder_output_dim, latent_dim)
        
        # Decoder network to reconstruct the image
        self.decoder_fc = nn.Linear(latent_dim, encoder_output_dim)
        
        
        # Adjust the decoder layers based on the dataset
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(encoder_output_dim, 256, kernel_size=4, stride=2, padding=1),  # Output: 256x2x2
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x4x4
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # Output: 64x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # Output: 32x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1) if data_name.endswith("cifar100") else
            nn.ConvTranspose2d(224, 3, kernel_size=4, stride=2, padding=1),    # Output: 3x32x32 for CIFAR-100 or ImageNet dimensions
            nn.Sigmoid()  # Map outputs to [0, 1] for image reconstruction
        )
        
        self.lr = lr

    def encode(self, x):
        # Pass through ResNet encoder
        features = self.encoder(x)
        # Generate mean and variance for the latent space
        mu = self.fc_mu(features)
        log_var = self.fc_var(features)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Decode the latent vector back to image dimensions
        z = self.decoder_fc(z)
        z = z.view(z.size(0), -1, 1, 1)  # Reshape for convolutional layers
        x_recon = self.decoder_conv(z)
        return x_recon

    def forward(self, x):
        # Forward pass through the VAE
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, mu, log_var = self(x)
        
        # Losses
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        loss = recon_loss + kl_div
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss, prog_bar=True)
        self.log("train_kl_div", kl_div, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, mu, log_var = self(x)
        
        # Losses
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_div
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_kl_div", kl_div, prog_bar=True)
        
        # Log images
        if batch_idx == 0:
            # Select the first image from the batch
            input_image = x[0].detach().cpu()
            recon_image = x_recon[0].detach().cpu()

            # Create a grid of images
            input_grid = vutils.make_grid(input_image, normalize=True)
            recon_grid = vutils.make_grid(recon_image, normalize=True)

            # Log images to Wandb
            self.logger.experiment.log({
                "Input Image": [wandb.Image(input_grid, caption="Input Image")],
                "Reconstructed Image": [wandb.Image(recon_grid, caption="Reconstructed Image")],
            }, step=self.global_step)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
