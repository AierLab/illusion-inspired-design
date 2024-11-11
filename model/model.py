from ._base import *
from sklearn.metrics import accuracy_score

class Model(LightningModule):
    def __init__(self, model_name, steps_per_epoch, num_classes, lr):
        super(Model, self).__init__()

        self.save_hyperparameters()

        # Load pre-trained ResNet50 model from timm with the correct number of classes
        self.model = create_model(model_name, pretrained=True, num_classes=num_classes)

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
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=self.steps_per_epoch, epochs=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
