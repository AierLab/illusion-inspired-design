import os

# Paths to check
train_path = "datasets/indl/train"
test_path = "datasets/indl/test"

# Only generate data if it doesn't exist
if not os.path.exists(train_path) or not os.path.exists(test_path):
    os.system("cd data/indl_dataset_generate && python main.py --train_dir ../../datasets/indl/train --test_dir ../../datasets/indl/test")
else:
    print("Dataset already exists. Skipping generation.")