import os
import sys
import logging
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import glob

# 日志配置
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.flush = sys.stdout.flush
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# 设置数据集目录
dataset_root = os.path.join("datasets")
imagenet_dir = os.path.join(dataset_root, "imagenet-1k")
train_dir = os.path.join(imagenet_dir, "train")
val_dir = os.path.join(imagenet_dir, "val")

os.makedirs(dataset_root, exist_ok=True)
os.makedirs(imagenet_dir, exist_ok=True)

logger.info(f"📁 当前工作目录：{os.getcwd()}")
logger.info(f"📂 数据集目录：{dataset_root}")

# 下载链接与目标文件
urls = {
    "train": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
    "val": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
    "devkit": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
}
files = {k: os.path.join(dataset_root, os.path.basename(v)) for k, v in urls.items()}


# 下载文件函数
def download_file(key, url, path):
    logger.info(f"⬇️ 开始下载 {key} 数据...")
    os.system(f'aria2c -x 16 -s 16 -c -o "{path}" "{url}"')
    if os.path.exists(path):
        logger.info(f"✅ {key} 下载完成：{os.path.basename(path)}")
    else:
        logger.error(f"❌ {key} 下载失败，未生成文件。")


# 下载文件
for key, url in urls.items():
    path = files[key]
    download_file(key, url, path)

# 解压 devkit（是 .tar.gz，用 pigz 加速）
devkit_path = files["devkit"]
if os.path.exists(devkit_path):
    logger.info("📂 正在解压 devkit...")
    devkit_target = os.path.join(imagenet_dir, "devkit")
    os.makedirs(devkit_target, exist_ok=True)
    os.system(f'tar --use-compress-program=pigz -xf "{devkit_path}" -C "{devkit_target}"')
    logger.info("✅ devkit 解压完成。")

# 解压训练主 tar
if not os.path.exists(train_dir):
    os.makedirs(train_dir, exist_ok=True)
    logger.info("📂 正在解压训练集主 tar 文件...")
    os.system(f'tar -xf "{files["train"]}" -C "{train_dir}"')
else:
    logger.info("📁 训练集主目录已存在，跳过主 tar 解压。")


# 解压训练子 tar（并发 + 容错）
def extract_class_tar(class_tar_path):
    try:
        class_name = os.path.basename(class_tar_path).replace('.tar', '')
        target_dir = os.path.join(train_dir, class_name)

        if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
            return  # 已解压

        os.makedirs(target_dir, exist_ok=True)
        os.system(f'tar -xf "{class_tar_path}" -C "{target_dir}"')
        if len(os.listdir(target_dir)) > 0:
            os.remove(class_tar_path)
        else:
            logger.warning(f"⚠️ 解压失败或为空：{class_tar_path}")
    except Exception as e:
        logger.error(f"❌ 解压失败：{class_tar_path}，错误：{e}")


def retry_extract_until_success():
    attempt = 1
    while True:
        tar_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.tar')]
        if not tar_files:
            logger.info("📦 所有子类 tar 已解压完成。")
            break
        logger.info(f"🔁 尝试第 {attempt} 次解压剩余 {len(tar_files)} 个 tar...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(extract_class_tar, tar_files), total=len(tar_files), desc="🔄 解压中"))
        attempt += 1


retry_extract_until_success()

# 解压验证集
if not os.path.exists(val_dir) or len(os.listdir(val_dir)) < 50000:
    os.makedirs(val_dir, exist_ok=True)
    logger.info("📂 正在解压验证集...")
    os.system(f'tar -xf "{files["val"]}" -C "{val_dir}"')
    logger.warning("⚠️ 验证集图片尚未分类（需要 devkit 标签支持后处理）。")
else:
    logger.info("📁 验证集目录已存在，跳过解压。")

# 统计训练集类别数量 & 图片总数
train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
image_count = len(glob.glob(os.path.join(train_dir, '*', '*.JPEG')))

logger.info(f"📊 当前训练集类别数量：{len(train_classes)}")
logger.info(f"🖼️ 当前训练集图片总数：{image_count}")

if len(train_classes) < 1000:
    logger.warning("⚠️ 警告：训练集类别数量不足 1000，可能存在解压失败。")
else:
    logger.info("✅ 训练集类别完整（1000 类）。")

if image_count < 1280000:
    logger.warning("⚠️ 警告：训练集图片数量低于标准值（应为 1,281,167 张），可能存在缺失。")
else:
    logger.info("✅ 训练集图片数量看起来完整。")

# 完成提示
logger.info("🎉 ImageNet-1K 数据集准备完成，可用于 PyTorch ImageFolder。")
