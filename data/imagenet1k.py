from ._base import num_workers
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch

def main(raw=False):
    # 定义 transforms（根据你之前的设定）
    transform_train = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [0.507, 0.487, 0.441]],
                             std=[x / 255.0 for x in [0.267, 0.256, 0.276]])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [0.507, 0.487, 0.441]],
                             std=[x / 255.0 for x in [0.267, 0.256, 0.276]])
    ])

    # 加载本地 ImageNet 1K 数据集（ImageFolder 格式）
    root = os.path.join("datasets", "imagenet-1k")
    train_path = os.path.join(root, "train")
    val_path = os.path.join(root, "val")

    train_dataset = ImageFolder(root=train_path, transform=transform_train)
    val_dataset = ImageFolder(root=val_path, transform=transform_test)

    # 创建 DataLoaders
    trainloader_imagenet1k = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader_imagenet1k = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 输出数据信息
    print("\nImageNet-1k DataLoader Summary:")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(trainloader_imagenet1k)}")
    print(f"Validation batches: {len(testloader_imagenet1k)}")

    # 输出首个 batch 验证数据形状
    first_train_batch = next(iter(trainloader_imagenet1k))
    first_val_batch = next(iter(testloader_imagenet1k))

    train_images, train_labels = first_train_batch
    val_images, val_labels = first_val_batch

    print(f"First training batch - Images shape: {train_images.shape}, Labels shape: {train_labels.shape}")
    print(f"First validation batch - Images shape: {val_images.shape}, Labels shape: {val_labels.shape}")

    if raw:
        return train_dataset, val_dataset

    return trainloader_imagenet1k, testloader_imagenet1k
