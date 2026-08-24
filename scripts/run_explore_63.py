# -*- coding: utf-8 -*-
"""6.3 池化硬度 × 标签噪声鲁棒性实验（文档 6.3）。

矩阵：7 池化变体 ×（1 干净 + 2 均匀 + 2 离群）× 5 折 × 3 种子 = 35 组。
基座：面积参照标签（25 张）+ 仅面积约束 MSE（无可信度/无空间，隔离池化单变量）。
协议：D'Amato 固定污染（噪声仅入训练标签，val/test 干净）+ Grahn 纪律（固定种子/超参）。

产出（exploration/results/）：
  6.3_results.csv              —— 35 组 × 指标 + 保持率
  6.3_degradation_uniform.png  —— 均匀噪声退化曲线
  6.3_degradation_outlier.png  —— 离群噪声退化曲线
  6.3_hardness_variance.png    —— 有效实例数 vs 误差退化（理论对照）
  ckpts_63/                    —— 各池化干净基线 fold0/seed0 权重（n_eff 计算用）
"""
import sys
import time
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs import baseline_m2 as cfg
from src.training import run_cv

RESULTS_DIR = PROJECT_ROOT / "exploration" / "results"
OUT_CSV = RESULTS_DIR / "6.3_results.csv"
CKPT_DIR = RESULTS_DIR / "ckpts_63"

POOLERS = [
    {"name": "mean",     "MODEL_TYPE": "meanpool",  "POOL_PARAM": None},
    {"name": "gmean_p5", "MODEL_TYPE": "gmean",     "POOL_PARAM": 5.0},
    {"name": "lse_t1",   "MODEL_TYPE": "lse",       "POOL_PARAM": 1.0},
    {"name": "top50",    "MODEL_TYPE": "topk",      "POOL_PARAM": 50.0},
    {"name": "max",      "MODEL_TYPE": "max",       "POOL_PARAM": 1.0},
    {"name": "attention","MODEL_TYPE": "abmil_inst","POOL_PARAM": None},
    {"name": "q75",      "MODEL_TYPE": "quantile",  "POOL_PARAM": 0.75},
]
NOISE_GRID = [("clean", 0.0), ("uniform", 0.15), ("uniform", 0.30),
              ("outlier", 0.2), ("outlier", 0.5)]


def make_cfg(labels_df, pooler, noise_model, noise_level, ckpt_dir=None):
    vcfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    vcfg.LABELS_DF = labels_df
    vcfg.MODEL_TYPE = pooler["MODEL_TYPE"]
    if pooler["POOL_PARAM"] is not None:
        vcfg.POOL_PARAM = pooler["POOL_PARAM"]
    vcfg.NOISE_MODEL = None if noise_model == "clean" else noise_model
    vcfg.NOISE_LEVEL = noise_level
    vcfg.CREDIBILITY = None
    vcfg.LAMBDA_SMOOTH = 0.0
    vcfg.USE_IND = False
    vcfg.USE_PROP = False
    vcfg.AMPLIFY_N = None
    vcfg.CKPT_DIR = ckpt_dir
    return vcfg


def n_eff_of(model, x, device):
    """有效参与实例数：池化权重梯度的参与率 (Σ|w|)²/Σw²（显式 scores 路径）。"""
    x = x.to(device)
    h = model.encoder(x)
    logits = model.head(h).squeeze(-1)
    scores2 = torch.sigmoid(logits).detach().requires_grad_(True)  # detach 成叶子节点
    if hasattr(model, "attention"):
        pred2 = (model.attention(h) * scores2).sum()
    else:
        mt = getattr(model, "pooling", "mean")   # MeanPoolMIL 无 pooling 属性，默认 mean
        p = getattr(model, "param", 1.0)
        n = scores2.shape[0]
        if mt == "mean":
            pred2 = scores2.mean()
        elif mt == "gmean":
            pred2 = torch.exp((torch.logsumexp(p * torch.log(scores2), 0) - np.log(n)) / p)
        elif mt == "lse":
            pred2 = torch.logsumexp(p * scores2, 0) / p - np.log(n) / p
        elif mt == "topk":
            pred2 = scores2.topk(min(int(p), n)).values.mean()
        elif mt == "max":
            pred2 = scores2.max()
        else:
            pred2 = scores2.topk(max(1, int(round(n * p)))).values.mean()
    pred2.backward()
    w = scores2.grad.detach().abs().cpu().numpy()
    return float((w.sum() ** 2) / (w ** 2).sum())


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(PROJECT_ROOT / "data" / "area_reference.csv", encoding="utf-8-sig")
    old25 = sorted(pd.read_csv(PROJECT_ROOT / "TCGA_Download" / "slide_rename_mapping.csv",
                               encoding="utf-8-sig")["slide_id"])
    labels_df = ref[ref.slide_id.isin(old25)][["slide_id", "area_frac_soft"]].rename(
        columns={"area_frac_soft": "weak_fraction"}).reset_index(drop=True)
    print(f"[6.3] 25 张面积口径标签就绪 n={len(labels_df)}")

    rows = []
    for pooler in POOLERS:
        clean_metrics = None
        for noise_model, level in NOISE_GRID:
            # 干净基线保存 fold0/seed0 权重（n_eff 用）
            ck = CKPT_DIR / pooler["name"] if noise_model == "clean" else None
            vcfg = make_cfg(labels_df, pooler, noise_model, level, ckpt_dir=ck)
            print(f"[6.3] {pooler['name']} × {noise_model}={level}")
            _, _, m = run_cv(vcfg)
            if noise_model == "clean":
                clean_metrics = m
            rows.append({"pooler": pooler["name"], "noise_model": noise_model,
                         "noise_level": level, **metrics_with_retention(m, clean_metrics)})
            print(f"  → {m}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[6.3] 矩阵完成 {len(out)} 行 → {OUT_CSV}")

    make_figures(out)
    print(f"[6.3] 总耗时 {(time.time()-t0)/60:.1f} min")


def metrics_with_retention(m, clean):
    if clean is None:
        return {**m, "ret_mae": 1.0, "ret_pearson": 1.0}
    return {**m,
            "ret_mae": m["mae"] / max(clean["mae"], 1e-12),
            "ret_pearson": m["pearson"] / clean["pearson"] if abs(clean["pearson"]) > 1e-12 else np.nan}


PALETTE = ["#1f78b4", "#33a02c", "#e31a1c", "#fe8307", "#6a3d9a",
           "#fb9a99", "#91D1C2"]
# 跨图一致的池化器顺序（硬度由软到硬，学习型 attention 居中）
POOLER_ORDER = ["mean", "gmean_p5", "lse_t1", "q75", "attention", "top50", "max"]
POOLER_COLOR = {p: PALETTE[i] for i, p in enumerate(POOLER_ORDER)}


def make_figures(out):
    """退化曲线 ×2 + 硬度-方差对照。"""
    for nm, fname in [("uniform", "6.3_degradation_uniform.png"),
                      ("outlier", "6.3_degradation_outlier.png")]:
        sub = out[out.noise_model.isin(["clean", nm])].copy()
        sub.loc[sub.noise_model == "clean", "noise_level"] = 0.0
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for pooler in POOLER_ORDER:
            g = sub[sub.pooler == pooler].sort_values("noise_level")
            if not len(g):
                continue
            axes[0].plot(g["noise_level"], g["ret_mae"], marker="o",
                         color=POOLER_COLOR[pooler], label=pooler)
            axes[1].plot(g["noise_level"], g["ret_pearson"], marker="o",
                         color=POOLER_COLOR[pooler], label=pooler)
        axes[0].set(xlabel="noise level", ylabel="MAE retention (noisy/clean)",
                    title=f"{nm}: MAE degradation"); axes[0].legend(fontsize=8)
        axes[1].set(xlabel="noise level", ylabel="Pearson retention",
                    title=f"{nm}: Pearson retention"); axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / fname, dpi=150)
        plt.close(fig)
    print("[6.3] 退化曲线已存")


def redraw_hardness_variance():
    """由缓存的 6.3_n_eff.csv 重绘硬度-方差散点（统一调色板）。"""
    df = pd.read_csv(RESULTS_DIR / "6.3_n_eff.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in df.iterrows():
        ax.scatter(r["n_eff"], r["mae_deg_u03"], s=90,
                   color=POOLER_COLOR.get(r["pooler"], "#333333"), zorder=3)
        ax.annotate(r["pooler"], (r["n_eff"], r["mae_deg_u03"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("effective number of instances n_eff (log scale)")
    ax.set_ylabel("MAE degradation at uniform noise 0.3")
    ax.set_title("6.3: pooling hardness vs noise sensitivity")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "6.3_hardness_variance.png", dpi=150)
    plt.close(fig)
    print("[6.3] 硬度-方差散点已重绘")


if __name__ == "__main__":
    main()
