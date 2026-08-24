# -*- coding: utf-8 -*-
"""M3 空间约束配置：继承 M2 基线，新增空间约束开关与变体矩阵。"""
from configs.baseline_m2 import *  # 复用 M2 全部路径与超参

# 空间约束开关（train_one_model 经 getattr 读取）
LAMBDA_SMOOTH = 0.0     # 平滑损失权重（V0 基线为 0）
USE_IND = False         # InD 免参 dropout 开关

# 变体矩阵：V0 基线 / V1 平滑 / V2 平滑+InD（V3 = V2 + 形态学后处理，不重训）
VARIANTS = [
    {"name": "V0_baseline", "LAMBDA_SMOOTH": 0.0, "USE_IND": False},
    {"name": "V1_smooth", "LAMBDA_SMOOTH": 0.01, "USE_IND": False},
    {"name": "V2_smooth_ind", "LAMBDA_SMOOTH": 0.01, "USE_IND": True},
]
DELIVERABLE_VARIANT = "V2_smooth_ind"

# 形态学后处理
MASK_THRESHOLD = 0.5
MIN_REGION_PATCHES = 4        # 最小保留连通域（patch 数）

# SRG 评估
SRG_STEPS = 100               # 百分位桶数
CKPT_DIR = OUTPUTS_DIR / "logs" / "checkpoints"   # 每折模型权重（SRG 复算用）
