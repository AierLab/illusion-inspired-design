#!/bin/bash

# Define the base directory
base_dir="checkpoints"

# Define datasets and models
datasets=("Cifar100" "ImageNet100" "ImageNet1k")
models=("1-baseline" "2-multi-task_concate_label" "3-combined_single-task_label" "+fine_tune" "4-multi-task_loss" "5-batch_bayesian_optimization" "6-comp_model" "7-comp_reverse_model" "8-style_transfer")

# Create folders for each dataset and model under the checkpoints directory
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    # Fine-tune models have subfolders
    if [[ "$model" == "+fine_tune" ]]; then
      mkdir -p "$base_dir/$dataset/3-combined_single-task_label/fine_tune"
      mkdir -p "$base_dir/$dataset/5-batch_bayesian_optimization/fine_tune"
      mkdir -p "$base_dir/$dataset/7-comp_reverse_model/fine_tune"
      mkdir -p "$base_dir/$dataset/8-style_transfer/fine_tune"
    else
      mkdir -p "$base_dir/$dataset/$model"
    fi
  done
done

echo "Folders created successfully in the checkpoints directory!"
