from ._base import *
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset

def dataset_filter(strength, return_datasets=False):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 32x32
        transforms.RandomRotation(degrees=180),
        transforms.ToTensor(),   
])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 32x32
        transforms.ToTensor(),
    ])


    # Initialize lists to hold all train and test datasets
    all_trainsets = []
    all_testsets = []

    # Loop through each dataset folder and collect train/test datasets
    for dataset_folder in ["dataset01", "dataset02", "dataset03", "dataset04", "dataset05"]:
        # Paths to data directories
        train_data_dir = f"datasets/indl/train/{dataset_folder}"
        test_data_dir = f"datasets/indl/test/{dataset_folder}"
        
        # Create instances of MyDataset
        trainset = MyDataset(train_data_dir, transforms=transform_train)
        testset = MyDataset(test_data_dir, transforms=transform_test)
        
        # Select 1/10 of the data
        train_indices = np.random.choice(len(trainset), len(trainset), replace=False)
        test_indices = np.random.choice(len(testset), len(testset), replace=False)
        
        # Create subsets
        train_subset = Subset(trainset, train_indices)
        test_subset = Subset(testset, test_indices)
        
        # Add datasets to the list
        all_trainsets.append(train_subset)
        all_testsets.append(test_subset)

    # Combine all training and testing datasets
    combined_trainset = ConcatDataset(all_trainsets)
    combined_testset = ConcatDataset(all_testsets)

    # Create DataLoaders for the combined datasets
    trainloader_indl = DataLoader(combined_trainset, batch_size=256//8, shuffle=True)
    testloader_indl = DataLoader(combined_testset, batch_size=256//8, shuffle=False)

    # DataLoader summary
    print("\nInDL DataLoader Summary:")
    print(f"Total training samples: {len(combined_trainset)}")
    print(f"Total testing samples: {len(combined_testset)}")
    print(f"Training batches: {len(trainloader_indl)}")
    print(f"Testing batches: {len(testloader_indl)}")

    # Display the first batch to verify data shapes
    first_train_batch = next(iter(trainloader_indl))
    first_test_batch = next(iter(testloader_indl))

    train_images, train_labels = first_train_batch
    test_images, test_labels = first_test_batch

    print(f"First training batch - Images shape: {train_images.shape}, Labels shape: {train_labels.shape}")
    print(f"First testing batch - Images shape: {test_images.shape}, Labels shape: {test_labels.shape}")

    if return_datasets:
        return combined_trainset, combined_testset
    
    return trainloader_indl, testloader_indl