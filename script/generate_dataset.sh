#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH -p long
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=1

source .venv/bin/activate

cd data/indl_dataset_generate

python main.py --train_dir ../../datasets/indl/train --test_dir ../../datasets/indl/test

cd ../..

python data/imagenet1k_filter100.py
