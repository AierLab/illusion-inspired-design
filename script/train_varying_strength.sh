#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -p long
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16

source .venv/bin/activate

# config_name=m_X_102
# python train.py --config_name $config_name > log/$config_name.log

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
log_dir="log/$timestamp"
mkdir -p $log_dir

config_names=(
    "m_A-none-indl224"
)

model_names=(
    "resnet50"
    "vgg16"
)

strengthes=(
    0
    0.2
    0.4
    0.6
    0.8
    1
)

# ulimit -n 65536
# ulimit -n

for config_name in "${config_names[@]}"
do
    python train.py --config_name "$config_name" &> "$log_dir/$config_name.log"    
    echo "Done: $config_name"
    echo "Log: $log_dir/$config_name.log"
    echo "======================================================================"
done
