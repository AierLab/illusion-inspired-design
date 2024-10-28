from train_base import *
from data_cifar10 import *

class CompositionModel(LightningModule):
    def __init__(self, steps_per_epoch, ckpt_path_A, ckpt_path_B, lr=1e-3):
        super(CompositionModel, self).__init__()
        self.save_hyperparameters()

        # Load two ResNet50 models from checkpoints
        self.model_A = ResNet50.load_from_checkpoint(ckpt_path_A, map_location="cpu").model
        # self.model_A = create_model("resnet50", pretrained=False, num_classes=2)
        # self.model_A.load_state_dict(torch.load(ckpt_path_A)["state_dict"], map_location="cpu")
        self.model_B = ResNet50.load_from_checkpoint(ckpt_path_B, map_location="cpu").model
        # self.model_B = create_model("resnet50", pretrained=False, num_classes=2)
        # self.model_B.load_state_dict(torch.load(ckpt_path_B)["state_dict"], map_location="cpu")


        # Freeze all parameters in models A and B
        self.model_A.eval()
        for param in self.model_A.parameters():
            param.requires_grad = False
        
        self.model_B.eval()
        for param in self.model_B.parameters():
            param.requires_grad = False

        # Extract specific layers from both models
        # Model A layers
        self.conv1_A = self.model_A.conv1
        self.bn1_A = self.model_A.bn1
        self.act1_A = self.model_A.act1
        self.maxpool_A = self.model_A.maxpool
        self.layer1_A = self.model_A.layer1
        self.layer2_A = self.model_A.layer2
        self.layer3_A = self.model_A.layer3
        self.layer4_A = self.model_A.layer4

        # Model B layers
        self.conv1_B = self.model_B.conv1
        self.bn1_B = self.model_B.bn1
        self.act1_B = self.model_B.act1
        self.maxpool_B = self.model_B.maxpool
        self.layer1_B = self.model_B.layer1
        self.layer2_B = self.model_B.layer2
        self.layer3_B = self.model_B.layer3
        self.layer4_B = self.model_B.layer4
        
        # Dynamically get the output channels for each layer
        out_channels_layer1 = self.layer1_A[-1].conv3.out_channels  # Get C for layer1
        out_channels_layer2 = self.layer2_A[-1].conv3.out_channels  # Get C for layer2
        out_channels_layer3 = self.layer3_A[-1].conv3.out_channels  # Get C for layer3

        # Define MLP blocks for feature mapping for each layer
        self.mlp1 = self.define_mlp(out_channels_layer1)
        self.mlp2 = self.define_mlp(out_channels_layer2)
        self.mlp3 = self.define_mlp(out_channels_layer3)

        # Define MultiheadAttention blocks for each layer
        self.attention1 = nn.MultiheadAttention(embed_dim=out_channels_layer1, num_heads=8, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=out_channels_layer2, num_heads=8, batch_first=True)
        self.attention3 = nn.MultiheadAttention(embed_dim=out_channels_layer3, num_heads=8, batch_first=True)

        # Pooling and classifier
        self.global_pool_B = self.model_B.global_pool
        self.fc_B = self.model_B.fc

        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch

    def define_mlp(self, input_dim):
        # Define a 3-layer MLP block
        hidden_dim = input_dim * 4
        output_dim = input_dim
        return nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        )

    def forward(self, x):
        # Initial layers for Model A
        x_A = self.conv1_A(x)
        x_A = self.bn1_A(x_A)
        x_A = self.act1_A(x_A)
        x_A = self.maxpool_A(x_A)

        # Initial layers for Model B
        x_B = self.conv1_B(x)
        x_B = self.bn1_B(x_B)
        x_B = self.act1_B(x_B)
        x_B = self.maxpool_B(x_B)

        # Layer 1
        x_A1 = self.layer1_A(x_A)
        x_B1 = self.layer1_B(x_B)

        # Apply MLP to x_A1
        x_A1_mapped = self.mlp1(x_A1)

        # Attention between x_A1_mapped and x_B1
        x_B2_input = self.apply_attention(x_A1_mapped, x_B1, self.attention1)

        # Layer 2
        x_A2 = self.layer2_A(x_A1)
        x_B2 = self.layer2_B(x_B2_input)

        # Apply MLP to x_A2
        x_A2_mapped = self.mlp2(x_A2)

        # Attention between x_A2_mapped and x_B2
        x_B3_input = self.apply_attention(x_A2_mapped, x_B2, self.attention2)

        # Layer 3
        x_A3 = self.layer3_A(x_A2)
        x_B3 = self.layer3_B(x_B3_input)

        # Apply MLP to x_A3
        x_A3_mapped = self.mlp3(x_A3)

        # Attention between x_A3_mapped and x_B3
        x_B4_input = self.apply_attention(x_A3_mapped, x_B3, self.attention3)

        # Layer 4
        x_B4 = self.layer4_B(x_B4_input)

        # Global pooling and final classification
        x = self.global_pool_B(x_B4)
        x = self.fc_B(x)

        return x

    def apply_attention(self, x_A, x_B, attention_layer):
        # Reshape the inputs to be compatible with nn.MultiheadAttention
        batch_size, channels, height, width = x_A.size()
        
        # Flatten the spatial dimensions (HxW) into a sequence for MultiheadAttention
        x_A_flat = x_A.view(batch_size, channels, -1).permute(0, 2, 1)  # [batch_size, H*W, channels]
        x_B_flat = x_B.view(batch_size, channels, -1).permute(0, 2, 1)  # [batch_size, H*W, channels]

        # Apply attention
        attn_output, _ = attention_layer(x_B_flat, x_A_flat, x_A_flat)

        # Reshape the output back to [batch_size, channels, H, W]
        attn_output = attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)

        # Add the attention output to x_B to get input for the next layer
        x_B_next = x_B + attn_output

        return x_B_next

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
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
