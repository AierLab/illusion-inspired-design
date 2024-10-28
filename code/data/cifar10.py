from base import *
import torchvision
import torch
from torch.utils.data import TensorDataset, DataLoader

# Download and load datasets
trainset = torchvision.datasets.CIFAR10(root='/database', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='/database', train=False, download=True, transform=transform)

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
trainloader_cifar10 = DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader_cifar10 = DataLoader(test_dataset, batch_size=64, shuffle=False)
