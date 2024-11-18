if __name__ == "__main__":
    import os
    # num_workers = os.cpu_count() // 8
    num_workers = 16
else:
    from ._base import num_workers
from datasets import load_dataset
import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from PIL import Image

datasets.config.IN_MEMORY_MAX_SIZE = 230

# Define transformations for training and testing sets
transform_train = transforms.Compose([
    transforms.Resize((226, 226)),  # Resize all images
    transforms.RandomCrop(224),
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

num_class=100

# Load datasets
dataset_path = f"/home/gus.xia/haobo/ai701-illusion-inspired-design/datasets/imagenet-1k-{num_class}"
train_dataset = datasets.load_from_disk(f"{dataset_path}/train")
val_dataset = datasets.load_from_disk(f"{dataset_path}/validation")

# # Find the max label for the first 200 data points in the training dataset
# max_label_train = max(val_dataset[:200]['label'])
# print(f"Max label in the first 200 training data points: {max_label_train}")

# Filter datasets to include only the first 100 classes
train_dataset = train_dataset.filter(lambda x: x['label'] < num_class)
val_dataset = val_dataset.filter(lambda x: x['label'] < num_class)

# # Find the max label for the first 200 data points in the training dataset
# max_label_train = max(val_dataset[:200]['label'])
# print(f"Max label in the first 200 training data points: {max_label_train}")

# Apply transformations manually
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Check if the image has 3 channels
        if image.mode != 'RGB': 
            image = image.convert('RGB')

        # Apply transformations if the image has 3 channels
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Wrap datasets with transformations
trainset_transformed = TransformedDataset(train_dataset, transform_train)
valset_transformed = TransformedDataset(val_dataset, transform_test)

# Create DataLoaders for transformed datasets
trainloader_imagenet100 = DataLoader(trainset_transformed, batch_size=20, shuffle=True, num_workers=num_workers, pin_memory=True)
testloader_imagenet100 = DataLoader(valset_transformed, batch_size=20, shuffle=False, num_workers=num_workers, pin_memory=True)

# Output DataLoader summary
print("\nImageNet-1k DataLoader Summary:")
print(f"Total training samples: {len(trainset_transformed)}")
print(f"Total validation samples: {len(valset_transformed)}")
print(f"Training batches: {len(trainloader_imagenet100)}")
print(f"Validation batches: {len(testloader_imagenet100)}")

# Display the first batch to verify data shapes
first_train_batch = next(iter(trainloader_imagenet100))
first_val_batch = next(iter(testloader_imagenet100))

train_images, train_labels = first_train_batch
val_images, val_labels = first_val_batch

print(f"First training batch - Images shape: {train_images.shape}, Labels shape: {train_labels.shape}")
print(f"First validation batch - Images shape: {val_images.shape}, Labels shape: {val_labels.shape}")