source .venv/bin/activate

# config_name=m_B_100
# config_name=m_B_1k
# config_name=m_B_1k_100
# config_name=m_X_102
# config_name=m_X_103
# config_name=m_X_103_1k3
# config_name=m_X_102_v2

config_name="m_B-none-imagenet100"

python train.py --config_name $config_name
