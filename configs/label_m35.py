# -*- coding: utf-8 -*-
"""M3.5 定向性能提升（瓶颈 1：标签口径错位）配置。

两个标签侧杠杆（溯源：Modeling Plan.md §M3.5）：
  A. 口径敏感性替换：percent_tumor_cells vs percent_tumor_nuclei（免费列，已在 weak_labels.csv）
  B. PANDA 软清洗前移：首轮 OOF 残差 → 逐切片可信度 → 加权 MSE 重训
实验矩阵：2 标签字段 × {无加权, 可信度加权} = 4 组（每组 5-fold × 3 种子）。
"""
from configs.baseline_m2 import *  # 继承 M2 路径与超参

LABEL_FIELD = "percent_tumor_cells"   # 可被变体覆盖为 percent_tumor_nuclei
CREDIBILITY = None                    # {slide_id: weight}；None = 不加权

# 可信度衰减参数：c_i = exp(-r_i / τ)，τ=None 时取 OOF 残差中位数（自适应尺度）
CREDIBILITY_TAU = None
