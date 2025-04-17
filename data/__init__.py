def get_dataloader(dataset_name: str):
    """
    Retrieves the DataLoader for the specified dataset.

    Args:
        data_name (str): The name of the dataset for which to retrieve the DataLoader.

    Returns:
        DataLoader: The DataLoader corresponding to the specified dataset.
    """
    if dataset_name == "indl224":
        from .indl224 import main
        return main()
    elif dataset_name == "imagenet1k":
        from .imagenet1k import main
        return main()
    elif dataset_name == "indl_and_imagenet1k":
        from .indl_and_imagenet1k import main
        return main()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")