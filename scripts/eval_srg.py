# -*- coding: utf-8 -*-
"""M3 SRG 热图忠实度评估（Idaji 2026 / xMIL 协议复用，自实现）。

协议：patch 按分数排序分 100 个百分位桶，分别按降序/升序逐步删除并用模型重算预测占比，
SRG = mean(升序曲线 − 降序曲线)；随机顺序为基线。
已知边界（写入注释与报告）：instance 架构下 patch 分数即贡献，SRG 近似天然成立——
本评估定位为 pipeline sanity check 与报告证据，不作为定位精度证明。

输出：outputs/logs/srg_results.csv（slide_id, srg, srg_random）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs import spatial_m3 as cfg
from src.models import make_model
from src.training import load_data, make_folds

# 交付模型类型（M4 聚合器对照后更新为分位聚合）
DELIVERABLE_MODEL_TYPE = "quantile"
DELIVERABLE_POOL_PARAM = 0.75


def load_fold_models(fold_idx, cfg, device):
    """加载该折 3 个种子的交付变体模型。"""
    models = []
    for s in cfg.SEEDS:
        vcfg = cfg
        vcfg.MODEL_TYPE = DELIVERABLE_MODEL_TYPE
        vcfg.POOL_PARAM = DELIVERABLE_POOL_PARAM
        m = make_model(vcfg)
        m.load_state_dict(torch.load(cfg.CKPT_DIR / f"fold{fold_idx}_seed{s}.pt",
                                     map_location=device, weights_only=True))
        m.eval()
        models.append(m.to(device))
    return models


def predict_fraction(models, x, device):
    if x.shape[0] == 0:
        return 0.0
    x = x.to(device)
    with torch.no_grad():
        return float(np.mean([m(x)[0].item() for m in models]))


def flip_curve(models, x, order, n_steps):
    """按 order 逐步删除 patch，记录预测曲线。"""
    n = x.shape[0]
    buckets = np.array_split(order, n_steps)
    kept = np.arange(n)
    curve = [predict_fraction(models, x, device_global)]
    for b in buckets:
        kept = np.setdiff1d(kept, b)
        curve.append(predict_fraction(models, x[kept], device_global))
    return np.array(curve)


def main():
    global device_global
    device_global = "cuda" if torch.cuda.is_available() else "cpu"
    df, feats = load_data(cfg)
    folds = make_folds(df, cfg)
    fold_of = {i: f for f, test_idx in enumerate(folds) for i in test_idx}

    rows = []
    for i, sid in enumerate(df["slide_id"]):
        scores = np.load(cfg.SCORES_DIR / f"{sid}.npy")
        x = feats[sid]
        models = load_fold_models(fold_of[i], cfg, device_global)

        desc = np.argsort(-scores)
        asc = np.argsort(scores)
        rand = np.random.RandomState(42).permutation(len(scores))

        c_desc = flip_curve(models, x, desc, cfg.SRG_STEPS)
        c_asc = flip_curve(models, x, asc, cfg.SRG_STEPS)
        c_rand_d = flip_curve(models, x, rand, cfg.SRG_STEPS)
        c_rand_a = flip_curve(models, x, rand[::-1], cfg.SRG_STEPS)

        srg = float((c_asc - c_desc).mean())
        srg_rand = float((c_rand_a - c_rand_d).mean())
        rows.append({"slide_id": sid, "srg": srg, "srg_random": srg_rand})
        print(f"  {sid}: SRG={srg:.4f} vs 随机 {srg_rand:.4f}")

    out = pd.DataFrame(rows)
    out_csv = cfg.OUTPUTS_DIR / "logs" / "srg_results.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[SRG] 平均 SRG={out.srg.mean():.4f}（随机基线 {out.srg_random.mean():.4f}）→ {out_csv}")


if __name__ == "__main__":
    main()
