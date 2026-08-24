# -*- coding: utf-8 -*-
"""临时：6.3 冒烟——验证噪声注入（训练污染/测试干净）与池化变体前向。"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs import baseline_m2 as cfg
from src.noise import build_noisy_labels

# 1. 噪声注入器单测
ref = pd.read_csv(PROJECT_ROOT / "data" / "area_reference.csv", encoding="utf-8-sig")
old25 = sorted(pd.read_csv(PROJECT_ROOT / "TCGA_Download" / "slide_rename_mapping.csv",
                           encoding="utf-8-sig")["slide_id"])
labels_df = ref[ref.slide_id.isin(old25)][["slide_id", "area_frac_soft"]].rename(
    columns={"area_frac_soft": "weak_fraction"}).reset_index(drop=True)

noisy = build_noisy_labels([0, 1, 2, 3], labels_df, "uniform", 0.3, seed=0)
orig = [labels_df["weak_fraction"].iloc[i] for i in [0, 1, 2, 3]]
print("uniform a=0.3: orig=%s → noisy=%s" % (np.round(orig, 3), np.round(list(noisy.values()), 3)))
noisy2 = build_noisy_labels([0, 1, 2, 3], labels_df, "outlier", 0.5, seed=0)
print("outlier r=0.5: orig=%s → noisy=%s" % (np.round(orig, 3), np.round(list(noisy2.values()), 3)))

# 确定性检查
noisy_again = build_noisy_labels([0, 1, 2, 3], labels_df, "uniform", 0.3, seed=0)
assert noisy == noisy_again, "噪声注入不确定！"
print("确定性: OK")

# 2. 池化变体前向冒烟
import torch
from src.models import make_model
x = torch.randn(500, 1024)
for mt, pp in [("meanpool", None), ("gmean", 5.0), ("lse", 1.0), ("topk", 50.0),
               ("max", 1.0), ("abmil_inst", None), ("quantile", 0.75)]:
    vcfg = types.SimpleNamespace(IN_DIM=1024, HIDDEN=512, DROPOUT=0.25,
                                 MODEL_TYPE=mt, POOL_PARAM=pp, ATTN_HIDDEN=256,
                                 OUTPUT_MODE="sigmoid")
    m = make_model(vcfg)
    pred, scores = m(x)
    print(f"  {mt}: pred={pred.item():.4f} scores{tuple(scores.shape)}")
print("SMOKE OK")
