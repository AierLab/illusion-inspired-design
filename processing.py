import os
import sys
import logging
import shutil
from tqdm import tqdm
from scipy.io import loadmat

# ========== 日志配置 ==========
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.flush = sys.stdout.flush
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

logger.info("🔍 正在执行验证集后处理：分类验证集图片...")

# ========== 数据路径 ==========
dataset_root = os.path.join("datasets")
imagenet_dir = os.path.join(dataset_root, "imagenet-1k")
val_dir = os.path.join(imagenet_dir, "val")

devkit_root = os.path.join(imagenet_dir, "devkit", "ILSVRC2012_devkit_t12")
gt_path = os.path.join(devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt")
meta_path = os.path.join(devkit_root, "data", "meta.mat")


# ========== 加载 ground truth 标签 ==========
with open(gt_path, "r") as f:
    val_labels = [int(line.strip()) for line in f.readlines()]

# ========== 加载 synset 标签映射 ==========
meta_raw = loadmat(meta_path, squeeze_me=True)
meta = meta_raw["synsets"]  # 是 numpy structured array

# 提取前1000个 synset（ID <= 1000），按 ID 排序
synsets = [entry for entry in meta if entry['ILSVRC2012_ID'] <= 1000]
synsets.sort(key=lambda x: x['ILSVRC2012_ID'])

# 提取 synset ID（如 'n01440764'）
idx_to_synset = [s['WNID'][0] for s in synsets]


# ========== 分类图片 ==========
val_images = sorted([f for f in os.listdir(val_dir) if f.endswith(".JPEG")])

for i, img_name in tqdm(enumerate(val_images), total=len(val_images), desc="📂 分类验证集"):
    label = val_labels[i]  # 标签是 1-based
    synset = idx_to_synset[label - 1]
    target_dir = os.path.join(val_dir, synset)
    os.makedirs(target_dir, exist_ok=True)

    src_path = os.path.join(val_dir, img_name)
    dst_path = os.path.join(target_dir, img_name)
    shutil.move(src_path, dst_path)

logger.info("✅ 验证集分类完成，可以直接用于 ImageFolder！")
