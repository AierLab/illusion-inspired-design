import torch
from .imagenet100 import train_dataset, val_dataset
from .indl224 import combined_trainset, combined_testset
from ._base import *
from torch.utils.data import ConcatDataset, DataLoader, Dataset

class StandardizeDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]

        if isinstance(item, tuple):
            data, label = item
        elif isinstance(item, dict):
            data, label = item["image"], item["label"]
        else:
            print(f"Item at index {idx} is not a tuple with 2 elements, with elements: {item}")
            return self.__getitem__(idx + 1)  # Skip this item and move to the next one

        # Convert data and label to tensors if they aren’t already
        data = torch.tensor(data) if not isinstance(data, torch.Tensor) else data
        label = torch.tensor(label) if not isinstance(label, torch.Tensor) else label

        return data, label

class LabelModifier(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Retrieve data and label from the original dataset
        data, label = self.dataset[idx]
        
        # Ensure label is a tensor
        if isinstance(label, tuple):
            label = label[0]  # Handle any tuple by extracting the first element
        
        # Ensure both data and label are tensors
        data = torch.tensor(data) if not isinstance(data, torch.Tensor) else data
        label = torch.tensor(label) if not isinstance(label, torch.Tensor) else label
        
        # Replace 0 with 10 and 1 with 11 in the labels
        label = torch.where(label == 0, torch.tensor(100), label)
        label = torch.where(label == 1, torch.tensor(101), label)
        
        return data, label

# Apply the label modification **only** on the binary classification dataset

# Apply LabelModifier to the binary classification datasets
modified_combined_trainset = LabelModifier(combined_trainset)
modified_combined_testset = LabelModifier(combined_testset)

# Wrap all datasets with StandardizeDataset to ensure uniform format
train_dataset_standardized = StandardizeDataset(train_dataset)
val_dataset_standardized = StandardizeDataset(val_dataset)

modified_combined_trainset_standardized = StandardizeDataset(modified_combined_trainset)
modified_combined_testset_standardized = StandardizeDataset(modified_combined_testset)

# Combine the standardized datasets
combined_trainset_final = ConcatDataset([
    modified_combined_trainset_standardized,
    train_dataset_standardized
])

combined_testset_final = ConcatDataset([
    modified_combined_testset_standardized,
    val_dataset_standardized
])

# Create DataLoader for combined training dataset
trainloader_combined = DataLoader(combined_trainset_final, batch_size=200, shuffle=True, num_workers=num_workers, pin_memory=True)

# Create DataLoader for combined testing dataset (with labels changed to 11 and 12)
testloader_combined = DataLoader(combined_testset_final, batch_size=200, shuffle=False, num_workers=num_workers, pin_memory=True)

# Output DataLoader summary
print("\nCombined DataLoader Summary:")
print(f"Total training samples: {len(combined_trainset_final)}")
print(f"Total validation samples: {len(combined_testset_final)}")
print(f"Training batches: {len(trainloader_combined)}")
print(f"Validation batches: {len(testloader_combined)}")
