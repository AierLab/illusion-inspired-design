from ._base import *
from .model import Model
from sklearn.metrics import accuracy_score

class ModelComp(LightningModule):
    def __init__(self, model_name, steps_per_epoch, num_classes, lr, ckpt_path_A, ckpt_path_B):
        super(ModelComp, self).__init__()
        self.save_hyperparameters()

        # Load two ResNet50 models from checkpoints
        self.model_A = Model.load_from_checkpoint(ckpt_path_A, map_location="cpu").model
        # self.model_A = create_model("resnet50", pretrained=False, num_classes=2)
        # self.model_A.load_state_dict(torch.load(ckpt_path_A)["state_dict"], map_location="cpu")
        # self.model_B = Model_base.load_from_checkpoint(ckpt_path_B, map_location="cpu").model
        self.model_B = create_model(model_name, pretrained=False, num_classes=num_classes)
        # self.model_B = create_model("resnet50", pretrained=False, num_classes=2)
        # self.model_B.load_state_dict(torch.load(ckpt_path_B)["state_dict"], map_location="cpu")


        # Freeze all parameters in models A and B
        self.model_A.eval()
        for param in self.model_A.parameters():
            param.requires_grad = False
        
        # self.model_B.eval() # TODO in our case, model B is trainable
        # for param in self.model_B.parameters():
        #     param.requires_grad = False

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
        self.attention1 = nn.MultiheadAttention(embed_dim=out_channels_layer1, num_heads=2, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=out_channels_layer2, num_heads=2, batch_first=True)
        self.attention3 = nn.MultiheadAttention(embed_dim=out_channels_layer3, num_heads=2, batch_first=True)

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
        attn_output, _ = attention_layer(x_B_flat, x_A_flat, x_A_flat) # Q, K, V
        
        # Reshape the output back to [batch_size, channels, H, W]
        attn_output = attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)

        # Add the attention output to x_B to get input for the next layer
        x_B_next = x_B + attn_output

        return x_B_next

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate Top-1 accuracy
        top1_preds = outputs.argmax(dim=1)
        top1_acc = accuracy_score(labels.cpu(), top1_preds.cpu())

        # Calculate Top-5 accuracy
        top5_preds = torch.topk(outputs, k=5, dim=1).indices
        top5_acc = torch.tensor([(label in top5) for label, top5 in zip(labels, top5_preds)]).float().mean().item()

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_top1_acc', top1_acc, prog_bar=True)
        self.log('train_top5_acc', top5_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate Top-1 accuracy
        top1_preds = outputs.argmax(dim=1)
        top1_acc = accuracy_score(labels.cpu(), top1_preds.cpu())

        # Calculate Top-5 accuracy
        top5_preds = torch.topk(outputs, k=5, dim=1).indices
        top5_acc = torch.tensor([(label in top5) for label, top5 in zip(labels, top5_preds)]).float().mean().item()

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_top1_acc', top1_acc, prog_bar=True)
        self.log('val_top5_acc', top5_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.lr / 10,  # Set the minimum learning rate (can be adjusted)
            max_lr=self.lr,        # Set the maximum learning rate
            step_size_up=self.steps_per_epoch // 2,  # Half of steps per epoch for increasing phase
            mode='exp_range'      # Choose the mode; 'triangular', 'triangular2', or 'exp_range'
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # return {"optimizer": optimizer}
