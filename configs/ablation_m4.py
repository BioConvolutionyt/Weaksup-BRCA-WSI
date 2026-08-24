# -*- coding: utf-8 -*-
"""M4 消融实验配置：聚合器家族对照 + 校准 + 范式对照。

基准线（用户指定）：cells 标签 + 可信度加权（Pearson 0.227 / Spearman 0.170 / MAE 0.115）。
消融变体在交付配置（cells + 可信度 + 平滑 λ=0.01）上单变量替换聚合器。
"""
from configs.spatial_m3 import *  # 继承交付配置路径与超参

# 交付配置固化（M3.5 结论）：cells + 可信度 + 平滑 λ=0.01、关 InD
LABEL_FIELD = "percent_tumor_cells"
LAMBDA_SMOOTH = 0.01
USE_IND = False
CREDIBILITY = None                    # 由 run_ablations.py 从 label_credibility.csv 注入

# 聚合器变体矩阵（含瓶颈 2 分位聚合扫描）
AGGREGATOR_VARIANTS = [
    {"name": "ref_meanpool",   "MODEL_TYPE": "meanpool", "POOL_PARAM": None},
    {"name": "abmil_inst",     "MODEL_TYPE": "abmil_inst", "POOL_PARAM": None},
    {"name": "chowder_top10",  "MODEL_TYPE": "chowder",  "POOL_PARAM": None},
    {"name": "gmean_p2",       "MODEL_TYPE": "gmean",    "POOL_PARAM": 2.0},
    {"name": "gmean_p5",       "MODEL_TYPE": "gmean",    "POOL_PARAM": 5.0},
    {"name": "gmean_p20",      "MODEL_TYPE": "gmean",    "POOL_PARAM": 20.0},
    {"name": "quantile_q75",   "MODEL_TYPE": "quantile", "POOL_PARAM": 0.75},
    {"name": "quantile_q50",   "MODEL_TYPE": "quantile", "POOL_PARAM": 0.50},
    {"name": "quantile_q25",   "MODEL_TYPE": "quantile", "POOL_PARAM": 0.25},
]

# 范式对照：ABMIL-inst 变体保存权重与 attention 分数
PARADIGM_VARIANT = "abmil_inst"
ATTN_DIR = OUTPUTS_DIR / "logs" / "attn_scores"
