# Illusion-Inspired Design

Welcome to the **Illusion-Inspired Design** repository! This project, as presented at the **Conference on Computer Vision and Pattern Recognition (CVPR)**, explores the innovative use of **geometric illusions** for enhancing advanced computer vision applications. Our approach leverages perceptual tricks to inspire new solutions, offering a fresh perspective on feature extraction, depth perception, and object recognition challenges.

---

## Overview

**Illusion-Inspired Design** aims to push the boundaries of computer vision by incorporating principles derived from geometric illusions, providing a unique approach to challenging real-world tasks. The methods and resources provided in this repository demonstrate how perceptual illusions can be harnessed to refine and optimize existing computer vision algorithms, thereby enabling better accuracy and robustness.

A key contribution of this work is the code labeled `m_X_202`, which plays a central role in the implementation and demonstrates our proposed techniques effectively. The project has been evaluated using a combination of public and synthetic datasets, with a primary focus on perceptual consistency and model reliability. We invite researchers, practitioners, and enthusiasts to explore our findings and extend the work further.

Additionally, we present the results of our experiments involving various model configurations, labeled as `202`, `102`, `102v2`, and `103`. These experiments assess the performance of ResNet50 on CIFAR-100 and ImageNet100 datasets using different multi-source data combination methods, providing insights into the robustness and adaptability of the models.

| Experiment | Dataset     | Illusion | Label    | +Loss | Top-1 Acc  | Top-5 Acc  | Illusion Acc |
| ---------- | ----------- | -------- | -------- | ----- | ---------- | ---------- | ------------ |
| Baseline   | CIFAR-100   |          | n        |       | 55.20%     | 83.36%     |              |
| 102        | CIFAR-100   | ✓        | n+2      |       | **59.30%** | 85.50%     | **85.20%**   |
| 102v2      | CIFAR-100   | ✓        | n+2      | ✓     | 56.80%     | 84.30%     | **87.70%**   |
| 103        | CIFAR-100   | ✓        | (n+1)+2  | ✓     | 58.40%     | **85.70%** | 84.00%       |
| 202        | CIFAR-100   | ✓        | (n+1)\*2 |       | 58.40%     | **85.90%** | 84.80%       |
| Baseline   | ImageNet100 |          | n        |       | 85.36%     | 97.52%     |              |
| 102        | ImageNet100 | ✓        | n+2      |       | 85.70%     | **98.00%** | 80.00%       |
| 102v2      | ImageNet100 | ✓        | n+2      | ✓     | 85.70%     | 97.93%     | 91.00%       |
| 103        | ImageNet100 | ✓        | (n+1)+2  | ✓     | **85.78%** | 97.70%     | **81.33%**   |
| 202        | ImageNet100 | ✓        | (n+1)\*2 |       | **85.78%** | 97.93%     | **81.83%**   |

### Implementation Details

The integration of geometric illusions into our image classification model follows several key equations that enhance model robustness and adaptability:

1. **Multi-Source Data Combination**:
   
   We define a loss function that combines primary classification loss with an auxiliary loss for illusion recognition:
   
$$
L_{total} = L_{classification} + \lambda \times L_{illusion}
$$
   
   where $L_{classification}$ is the standard cross-entropy loss for the primary dataset, $L_{illusion}$ is an auxiliary loss term for recognizing geometric illusions, and $\lambda$ is a weighting factor to balance both losses.

2. **Label Augmentation**:
   
   To incorporate additional labels for illusions, we extend the label space as follows:
   
$$
y' = [y, y_{illusion}]
$$
   
   where $y$ represents the original class label, and $y_{illusion}$ represents the presence of a specific geometric illusion. This allows the model to jointly learn object classification and illusion identification.

3. **Feature Extraction Enhancement**:
   
   The feature extraction process is enhanced by including perceptual features influenced by geometric illusions. Specifically, we apply a modified convolutional block that integrates both standard feature maps and illusion-aware feature maps:
   
$$
F_{enhanced} = F_{standard} + \alpha \times F_{illusion}
$$
   
   where $F_{standard}$ are the traditional convolutional features, $F_{illusion}$ are the features derived from illusion-based transformations, and $\alpha$ is a scaling parameter.

These formulas form the basis of our implementation, enabling the model to effectively learn from both primary and illusion-augmented datasets.

---

## Dependencies

This project uses the **[INDL Dataset](https://github.com/AierLab/indl-dataset)** for comprehensive data generation and processing. Please ensure you have cloned and set up the dataset repository before running the training or evaluation scripts.

### Required Tools

- **uv**: a streamlined package manager for managing dependencies effortlessly.
- **Python 3.8+**: this implementation has been tested with Python 3.8 and above.

---

## Getting Started

### [Optional] Download Pretrained Checkpoints 

1. **Download Pretrained Checkpoints**  
   Access the pretrained checkpoints from the provided [Google Drive link](https://drive.google.com/drive/folders/1DEhBiSJGRuxYD4ypEj4z4wRVDwYW-v4X?usp=sharing).
   And put models in `tmp/models` directory.

3. **Organize the Files**  
   After downloading, place the files in the directory:  
   ```
   config/train/_base.yaml
   ```

4. **Update Configuration File**  
   Modify the configuration file to enable pretrained checkpoints:  
   Open `config/train/_base.yaml` and set:  
   ```yaml
   trainer:
       use_pretrain: True
   ```

### Setting Up Your Environment

1. **Install the `uv` package manager** for managing dependencies:
   ```bash
   pip install uv
   ```

2. **Synchronize dependencies** using `uv` for easy installation:
   ```bash
   uv sync
   ```

3. Alternatively, you can manually install dependencies via **pip**:
   ```bash
   pip install -r requirements.txt
   ```

---

### Generate datasets

create imagenet-1k in the datasets folder.

run: `uv run .\data\indl_dataset_generate\main.py --train_dir datasets/indl/train --test_dir datasets/indl/test`

## Running the Training Pipeline

To start training, simply execute the provided training script:

```bash
bash script/train_all.sh
```

The script supports multiple configurations and includes hooks for logging, checkpointing, and monitoring the training progress.

---

## Dataset and Evaluation

This repository is powered by the **INDL Dataset**, which is central to our research. It is used for generating both training and testing data, offering a diverse range of synthetic illusions that challenge traditional perception algorithms.

All experimental results are available in **Weights & Biases (wandb)** for easy visualization and analysis. You may train it and see the result of validation.

---

## Citation

If you find our work helpful, please consider citing our paper:

```latex
@inproceedings{illusion2024,
  title={Geometric Illusion-Augmented Image Classification},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

---

## License

This project is licensed under the **Apache 2.0 License**. Feel free to use the code and resources in compliance with the license terms.

---

## Get Involved

We welcome contributions! Whether you’re interested in extending the methods, adding new datasets, or improving the training pipeline, your input is highly valued. Feel free to submit issues or pull requests to help us make this project even better.

For detailed contribution guidelines, please refer to `CONTRIBUTING.md`.

---

## Connect with Us

If you have any questions or would like to discuss this project further, please feel free to reach out! We’re also looking forward to collaborating with like-minded researchers and enthusiasts in exploring new possibilities inspired by visual perception.

**Let's shape the future of computer vision, one illusion at a time.**

