# -*- coding: utf-8 -*-
"""M2 基线占比回归——全部配置（项目规则：配置在代码内，不用命令行参数）。

溯源：Modeling Plan.md §2.1 / M2。
- 模型形态：文档 5.3 公式（sigmoid patch 分数 + MeanPool + fraction MSE）
- 超参默认值：D'Amato 2025（Adam lr 2e-4 / wd 1e-5 / 早停 patience 20 / min 50 epoch）
- 集成：每折 3 初始化种子（UBC 种子集成）× 5 折（PANDA 多折）
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据路径
FEATURES_DIR = PROJECT_ROOT / "features"
COORDS_DIR = PROJECT_ROOT / "coords"
WEAK_LABELS_CSV = PROJECT_ROOT / "data" / "weak_labels.csv"
WSI_DIR = PROJECT_ROOT / "TCGA_Download"

# 输出路径（复用既有 outputs/ 目录）
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCORES_DIR = OUTPUTS_DIR / "logs" / "scores"      # 每切片 OOF patch 分数（供热图）
PREDICTIONS_CSV = OUTPUTS_DIR / "predictions.csv"
REGRESSION_PLOT = OUTPUTS_DIR / "regression_plot.png"
TRAIN_LOG = OUTPUTS_DIR / "logs" / "train_baseline.log"

# 模型
IN_DIM = 1024
HIDDEN = 512
DROPOUT = 0.25
# 输出形态："sigmoid"（文档 5.3 公式）或 "linear"（D'Amato 官方实现：均值线性输出 + 评估截断）
OUTPUT_MODE = "sigmoid"
# 模型类型："meanpool" 或 "abmil_inst"（注意力加权实例分数，治梯度稀释）
MODEL_TYPE = "meanpool"
ATTN_HIDDEN = 256

# 训练
N_FOLDS = 5
SEEDS = [0, 1, 2]
LR = 2e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 200
EARLY_PATIENCE = 20
MIN_EPOCHS = 50
VAL_RATIO = 0.15          # 每折内部 train/val 划分
N_BINS = 5                # weak_fraction 分箱（分层用）

# 冒烟开关：True 时仅 1 折 × 1 种子 × 15 epoch（验证管道）
DEBUG_SMOKE = False
