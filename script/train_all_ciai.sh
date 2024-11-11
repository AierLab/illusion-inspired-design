#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -p long
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=1

source .venv/bin/activate

# config_name=m_X_102
# python train.py --config_name $config_name > log/$config_name.log

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
log_dir="log/$timestamp"
mkdir -p $log_dir

config_names=(
    "m_A-none-indl32"
    "m_A-none-indl224"

    "m_B-none-cifar100"
    "m_B-none-imagenet100"
    # "m_B-none-imagenet1k"

    # "m_X-102-cifar100"
    "m_X-102-imagenet100"
    # "m_X-102-imagenet1k"

    # "m_X-102_v2-cifar100"
    "m_X-102_v2-imagenet100"
    # "m_X-102_v2-imagenet1k"

    # "m_X-comp-cifar100"
    "m_X-comp-imagenet100"
    # "m_X-comp-imagenet1k"

    # "m_X-comp_reverse-cifar100"
    "m_X-comp_reverse-imagenet100"
    # "m_X-comp-imagenet1k"

    # "m_X-103-cifar100"
    "m_X-103-imagenet100"
    # "m_X-103-imagenet1k"
)

for config_name in "${config_names[@]}"
do
    python train.py --config_name "$config_name" &> "$log_dir/$config_name.log"    
    echo "Done: $config_name"
    echo "Log: $log_dir/$config_name.log"
    echo "======================================================================"
done
