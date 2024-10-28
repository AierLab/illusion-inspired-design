from train.m_B_hydra import main

if __name__ == "__main__":
    for model_name in ["convnextv2_huge", "convnext_xxlarge", "resnet50"]:
        main(model_name)