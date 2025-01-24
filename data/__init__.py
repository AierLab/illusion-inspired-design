def get_dataloader(dataset_name: str, strength: float = None):
    """
    Retrieves the DataLoader for the specified dataset.

    Args:
        data_name (str): The name of the dataset for which to retrieve the DataLoader.
        strength (float): default None, a float from 0 to 1, indicate the strength of the indl data that we pick from dataset.

    Returns:
        DataLoader: The DataLoader corresponding to the specified dataset.
    """
    
    if dataset_name == "cifar100":
        from .cifar100 import trainloader_cifar100, testloader_cifar100
        return trainloader_cifar100, testloader_cifar100
    elif dataset_name == "indl32":
        from .indl32 import trainloader_indl, testloader_indl
        return trainloader_indl, testloader_indl
    elif dataset_name == "indl_and_cifar100":
        from .indl_and_cifar100 import trainloader_combined, testloader_combined
        return trainloader_combined, testloader_combined
    elif dataset_name == "imagenet100":
        from .imagenet100 import trainloader_imagenet100, testloader_imagenet100
        return trainloader_imagenet100, testloader_imagenet100
    elif dataset_name == "indl224":
        if strength:
            from .indl224_filtered import dataset_filter
            trainloader_indl, testloader_indl = dataset_filter(strength)
            return trainloader_indl, testloader_indl
        from .indl224 import trainloader_indl, testloader_indl
        return trainloader_indl, testloader_indl
    elif dataset_name == "indl_and_imagenet100":
        from .indl_and_imagenet100 import trainloader_combined, testloader_combined
        return trainloader_combined, testloader_combined
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")