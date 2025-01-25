#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -p long
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16

# source .venv/bin/activate

# config_name=m_X_102
# python train.py --config_name $config_name > log/$config_name.log

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
log_dir="log/$timestamp"
mkdir -p $log_dir

config_names=(
    "m_A-none-indl224"
    "m_X-202-imagenet100"
)

model_names=(
    "resnet50"
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
    for model_name in "${model_names[@]}"
    do
        for strength in "${strengthes[@]}"
        do
            python train.py --config_name "$config_name" --model_name $model_name --strength "$strength" &> "$log_dir/$model_name.$config_name.$strength.log"    
            echo "Done: $model_name.$config_name.$strength"
            echo "$log_dir/$model_name.$config_name.$strength.log" 
            echo "======================================================================"
        done
    done
done
