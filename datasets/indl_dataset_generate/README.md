# Dataset Generation Script

This repository provides a Python script (`main.py`) to generate datasets using a planner and generator module. The script allows you to specify directories for training and testing data and generates datasets concurrently.

## Requirements

Ensure you have Python 3 installed and the required modules, `generator` and `planner`, available in your Python environment.

## Usage

1. **Change working dir to generator folder:**
   ```bash
   cd indl_dataset_generate
   ```

2. **Run the Script**

   Run `main.py` from the command line, specifying the paths to the training and testing directories. Paths can be either relative or absolute.

   ### Command Syntax

   ```bash
   python main.py --train_dir <path_to_training_directory> --test_dir <path_to_testing_directory>
   ```

   Replace `<path_to_training_directory>` and `<path_to_testing_directory>` with your desired directories.

   ### Example Usage

   Using relative paths:
   ```bash
   python main.py --train_dir ./datasets/indl/train --test_dir ./datasets/indl/test
   ```

   Using absolute paths:
   ```bash
   python main.py --train_dir /home/user/datasets/indl/train --test_dir /home/user/datasets/indl/test
   ```

3. **Output**

   The script generates datasets in parallel, with each dataset containing 2000 entries as specified in the script. 

## Arguments

- `--train_dir` - Path to the training directory (can be a relative or absolute path).
- `--test_dir` - Path to the testing directory (can be a relative or absolute path).

## Code Structure

- **`main.py`** - The main entry point for dataset generation.
- **`generator`** - Module responsible for dataset generation.
- **`planner`** - Planner module to handle dataset planning and configuration.
