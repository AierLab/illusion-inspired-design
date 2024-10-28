from data_base import *
import torch
from torch.utils.data import DataLoader, TensorDataset

# Paths to save or load the combined dataset
train_pth_file = "/database/indl/train/combined_trainset.pth"
test_pth_file = "/database/indl/test/combined_testset.pth"


if os.path.exists(train_pth_file) and os.path.exists(test_pth_file):
    # Load the datasets if .pth files exist
    print(f"Loading train dataset from {train_pth_file} and test dataset from {test_pth_file}...")
    train_data = torch.load(train_pth_file)
    test_data = torch.load(test_pth_file)
    
    # Create datasets from loaded data
    combined_trainset = TensorDataset(train_data['images'], train_data['labels'])
    combined_testset = TensorDataset(test_data['images'], test_data['labels'])
else:
    # If .pth files don't exist, create datasets and save them to .pth
    print("Creating new datasets and saving them to .pth...")

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    # Loop through dataset folders and collect datasets
    for dataset_folder in ["dataset01", "dataset02", "dataset03", "dataset04", "dataset05"]:
        # Paths to data directories
        train_data_dir = f"/database/indl/train/{dataset_folder}"
        test_data_dir = f"/database/indl/test/{dataset_folder}"
        
        # Create instances of MyDataset
        trainset = MyDataset(train_data_dir, transforms=transform)  # No transforms yet
        testset = MyDataset(test_data_dir, transforms=transform)
        
        # Collect images and labels from trainset
        for i in range(len(trainset)):
            image, label = trainset[i]
            image = transform(image)  # Apply transforms here
            train_images.append(image)
            train_labels.append(label)
        
        # Collect images and labels from testset
        for i in range(len(testset)):
            image, label = testset[i]
            image = transform(image)
            test_images.append(image)
            test_labels.append(label)
    
    # Stack images and labels into tensors
    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels)
    test_images = torch.stack(test_images)
    test_labels = torch.tensor(test_labels)
    
    # Save the data to .pth files
    torch.save({'images': train_images, 'labels': train_labels}, train_pth_file)
    torch.save({'images': test_images, 'labels': test_labels}, test_pth_file)
    print(f"Train dataset saved to {train_pth_file}.")
    print(f"Test dataset saved to {test_pth_file}.")

    # Create datasets from the saved data
    combined_trainset = TensorDataset(train_images, train_labels)
    combined_testset = TensorDataset(test_images, test_labels)

# Create DataLoaders for the combined datasets
trainloader_indl = DataLoader(combined_trainset, batch_size=20, shuffle=True)
testloader_indl = DataLoader(combined_testset, batch_size=20, shuffle=False)
