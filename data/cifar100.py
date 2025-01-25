from ._base import *
import torchvision
import torch
from torch.utils.data import TensorDataset, DataLoader

# Normalize training set together with augmentation
transform_train = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),  # Apply AutoAugment with CIFAR10 policy
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

# Normalize test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.Resize(34),
    transforms.CenterCrop(32),               # Center crop to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

# Download and load datasets
trainset = torchvision.datasets.CIFAR100(root="datasets/", train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root="datasets/", train=False, download=True, transform=transform_test)

# Preload training data to memory
train_data = torch.stack([trainset[i][0] for i in range(len(trainset))])
train_labels = torch.tensor([trainset[i][1] for i in range(len(trainset))])

# Preload test data to memory
test_data = torch.stack([testset[i][0] for i in range(len(testset))])
test_labels = torch.tensor([testset[i][1] for i in range(len(testset))])

# Create TensorDatasets
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

# Dataloaders
trainloader_cifar100 = DataLoader(train_dataset, batch_size=256//8, shuffle=True, num_workers=num_workers, pin_memory=True)
testloader_cifar100 = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)

# DataLoader Summary
print("\nCifar100 DataLoader Summary:")
print(f"Total training samples: {len(train_dataset)}")
print(f"Total testing samples: {len(test_dataset)}")
print(f"Training batches: {len(trainloader_cifar100)}")
print(f"Testing batches: {len(testloader_cifar100)}")

# Display the shape of the first batch for verification
first_train_batch = next(iter(trainloader_cifar100))
first_test_batch = next(iter(testloader_cifar100))

train_images, train_labels = first_train_batch
test_images, test_labels = first_test_batch

print(f"First training batch - Images shape: {train_images.shape}, Labels shape: {train_labels.shape}")
print(f"First testing batch - Images shape: {test_images.shape}, Labels shape: {test_labels.shape}")
