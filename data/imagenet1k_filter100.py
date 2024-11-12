num_workers = 32
from datasets import load_dataset
import datasets
import os
import shutil

datasets.config.IN_MEMORY_MAX_SIZE = 32

num_class = 100

new_dataset_path = f"/home/gus.xia/haobo/ai701-illusion-inspired-design/datasets/imagenet-1k-{num_class}"
print(f"New dataset path: {new_dataset_path}")

# Create directory if it does not exist, else remove it then create
if os.path.exists(new_dataset_path):
    shutil.rmtree(new_dataset_path)
    print(f"Removed existing directory: {new_dataset_path}")
os.makedirs(new_dataset_path)

train_dataset = load_dataset('/home/gus.xia/haobo/ai701-illusion-inspired-design/datasets/imagenet-1k', split='train', trust_remote_code=True, cache_dir="tmp/cache")
val_dataset = load_dataset('/home/gus.xia/haobo/ai701-illusion-inspired-design/datasets/imagenet-1k', split='validation', trust_remote_code=True, cache_dir="tmp/cache")

# Find the max label for the first 200 data points in the training dataset
max_label_train = max(val_dataset[:200]['label'])
max_label_val = max(val_dataset[:200]['label'])
print(f"Max label in the first 200 train & val data points: {max_label_train} & {max_label_val}")

# # Filter datasets to include only the first 100 classes
train_dataset = train_dataset.filter(lambda x: x['label'] < num_class)
val_dataset = val_dataset.filter(lambda x: x['label'] < num_class)

# # Reduce the dataset to the first 500 data points
# train_dataset = train_dataset.select(range(500))
# val_dataset = val_dataset.select(range(500))

# Save the reduced datasets to the new path
train_dataset.save_to_disk(f"{new_dataset_path}/train")
val_dataset.save_to_disk(f"{new_dataset_path}/validation")

# Load the new dataset from the saved files [USAGE]
new_dataset_path = f"/home/gus.xia/haobo/ai701-illusion-inspired-design/datasets/imagenet-1k-{num_class}"
new_train_dataset = datasets.load_from_disk(f"{new_dataset_path}/train")
new_val_dataset = datasets.load_from_disk(f"{new_dataset_path}/validation")

# Print the size of the new datasets
print(f"New train dataset size: {len(new_train_dataset)}")
print(f"New validation dataset size: {len(new_val_dataset)}")

# Find the max label for the first 200 data points in the training dataset
max_label_train = max(val_dataset[:200]['label'])
max_label_val = max(val_dataset[:200]['label'])
print(f"Max label in the first 200 train & val data points: {max_label_train} & {max_label_val}")

