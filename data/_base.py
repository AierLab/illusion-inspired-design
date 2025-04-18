# Define transformation

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# num_workers = os.cpu_count()
num_workers = 16


class MyDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.data_info = self.get_img_info(data_dir)
        self.transforms = transforms

    def __getitem__(self, item):
        item = int(item)  # Ensure item is an integer
        path_img, label = self.data_info.iloc[item][0:2]
        label = torch.tensor(int(label))  # Convert label to a tensor
        path_img = os.path.join(self.data_dir, str(path_img))
        image = Image.open(path_img).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.data_info)
    
    def get_img_info(self, data_dir):
        path_dir = os.path.join(data_dir, 'label.csv')
        return pd.read_csv(path_dir)

# Custom Dataset to hold tensors
class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)