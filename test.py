import torch
from pytorch_lightning import Trainer, LightningModule

# 打印 CUDA 可用性
print("CUDA available:", torch.cuda.is_available())

# 定义一个最简单的模型（不训练）
class DummyModel(LightningModule):
    def training_step(self, batch, batch_idx):
        return None
    def configure_optimizers(self):
        return None

# 实例化模型
model = DummyModel()

# 初始化 Trainer（自动选择设备）
trainer = Trainer(
    accelerator="auto",   # 可选："cpu", "gpu", "mps"
    devices=1,
    max_epochs=1,
    enable_model_summary=False,  # 避免打印模型结构
)

print("Trainer initialized successfully!")
