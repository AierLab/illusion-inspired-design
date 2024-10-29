# from train.m_B_hydra import main
from train.m_X_102_v2 import main

if __name__ == "__main__":
    for model_name in ["resnet50", "convnextv2_huge", "vgg16"]:
        main(model_name)