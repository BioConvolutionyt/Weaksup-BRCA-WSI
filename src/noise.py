# -*- coding: utf-8 -*-
"""6.3 噪声注入器（D'Amato 协议：仅污染训练标签，val/test 保持干净）。

两种噪声模型（回答文档 6.3"高斯 vs 离群值"之问）：
- uniform：y + U[−a, a]，截断到 [0,1]（现实量级 a≤0.3）；
- outlier：以概率 r 把标签替换为 U[0,1] 随机值（结构性污染，模拟坏标签）。

每个模型运行内噪声固定（按种子确定性生成），复现 D'Amato add_noise 的"固定污染训练集"协议。
"""
import numpy as np


def inject_one(y, model, level, rng):
    """对单个标签注入噪声。model ∈ {"uniform", "outlier"}；level = a 或 r。"""
    if model == "uniform":
        return float(np.clip(y + rng.uniform(-level, level), 0.0, 1.0))
    if model == "outlier":
        if rng.rand() < level:
            return float(rng.uniform(0.0, 1.0))
        return float(y)
    raise ValueError(f"未知噪声模型: {model}")


def build_noisy_labels(train_ids, labels, model, level, seed):
    """为训练集生成确定性噪声标签映射 {row_index: noisy_label}。"""
    rng = np.random.RandomState(seed)
    return {i: inject_one(labels["weak_fraction"].iloc[i], model, level, rng)
            for i in train_ids}
