from ._base import num_workers
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from PIL import Image
import json
import glob
import os

# Define transformations for training and testing sets
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in [0.507, 0.487, 0.441]],
                         std=[x / 255.0 for x in [0.267, 0.256, 0.276]])])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images
    transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in [0.507, 0.487, 0.441]],
                         std=[x / 255.0 for x in [0.267, 0.256, 0.276]])
])

num_class = 100  # Filter the first 100 classes if necessary

# Define a custom dataset class
class ImageNetDataset(Dataset):
    def __init__(self, root_dirs, labels_file, transform=None, num_classes=None):
        """
        root_dirs: List of directories containing the dataset (e.g., [train.X1, train.X2, ...]).
        labels_file: Path to Labels.json file for synset mapping.
        transform: Optional torchvision transform for preprocessing.
        num_classes: Filter to include only the first `num_classes` classes.
        """
        self.samples = []  # List to store (image_path, label) pairs
        self.transform = transform
        
        # Load synset to human-readable label mapping
        with open(labels_file, 'r') as f:
            self.synset_to_label = json.load(f)
        
        # Map synset (class folder names) to integer indices
        self.synsets = sorted(self.synset_to_label.keys())
        self.synset_to_idx = {synset: idx for idx, synset in enumerate(self.synsets)}
        
        # Gather all image paths and their corresponding labels
        for root_dir in root_dirs:
            for synset in os.listdir(root_dir):
                synset_dir = os.path.join(root_dir, synset)
                if os.path.isdir(synset_dir) and synset in self.synset_to_idx:
                    label = self.synset_to_idx[synset]
                    if num_classes is None or label < num_classes:
                        for img_file in glob.glob(f"{synset_dir}/*.JPEG"):
                            self.samples.append((img_file, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        
        # Ensure all images are in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Dataset paths
dataset_path = "/home/indl/workspace/illusion-inspired-design/datasets/imagenet100"
train_dirs = [f"{dataset_path}/train.X1", f"{dataset_path}/train.X2", 
              f"{dataset_path}/train.X3", f"{dataset_path}/train.X4"]
val_dir = f"{dataset_path}/val.X"
labels_file = f"{dataset_path}/Labels.json"

# Create datasets
train_dataset = ImageNetDataset(train_dirs, labels_file, transform=transform_train, num_classes=num_class)
val_dataset = ImageNetDataset([val_dir], labels_file, transform=transform_test, num_classes=num_class)

# Create DataLoaders
trainloader_imagenet100 = DataLoader(train_dataset, batch_size=256//8, shuffle=True, num_workers=num_workers, pin_memory=True)
testloader_imagenet100 = DataLoader(val_dataset, batch_size=256//8, shuffle=False, num_workers=num_workers, pin_memory=True)

# Output DataLoader summary
print("\nImageNet-100 DataLoader Summary:")
print(f"Total training samples: {len(train_dataset)}")
print(f"Total validation samples: {len(val_dataset)}")
print(f"Training batches: {len(trainloader_imagenet100)}")
print(f"Validation batches: {len(testloader_imagenet100)}")

# Display the first batch to verify data shapes
first_train_batch = next(iter(trainloader_imagenet100))
first_val_batch = next(iter(testloader_imagenet100))

train_images, train_labels = first_train_batch
val_images, val_labels = first_val_batch

print(f"First training batch - Images shape: {train_images.shape}, Labels shape: {train_labels.shape}")
print(f"First validation batch - Images shape: {val_images.shape}, Labels shape: {val_labels.shape}")
