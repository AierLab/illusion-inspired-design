def get_dataloader(dataset_name: str):
    """
    Retrieves the DataLoader for the specified dataset.

    Args:
        data_name (str): The name of the dataset for which to retrieve the DataLoader.

    Returns:
        DataLoader: The DataLoader corresponding to the specified dataset.
    """
    
    if dataset_name == "cifar100":
        from .cifar100 import trainloader_cifar100, testloader_cifar100
        return trainloader_cifar100, testloader_cifar100
    elif dataset_name == "indl":
        from .indl32 import trainloader_indl, testloader_indl
        return trainloader_indl, testloader_indl
    elif dataset_name == "indl_and_cifar100":
        from .indl_and_cifar100 import trainloader_combined, testloader_combined
        return trainloader_combined, testloader_combined
    elif dataset_name == "imagenet1k":
        from .imagenet1k import trainloader_imagenet1k, testloader_imagenet1k
        return trainloader_imagenet1k, testloader_imagenet1k
    elif dataset_name == "imagenet100":
        from .imagenet100 import trainloader_imagenet1k, testloader_imagenet1k
        return trainloader_imagenet1k, testloader_imagenet1k
    elif dataset_name == "indl_and_imagenet1k":
        from .indl_and_imagenet1k import trainloader_combined, testloader_combined
        return trainloader_combined, testloader_combined
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")