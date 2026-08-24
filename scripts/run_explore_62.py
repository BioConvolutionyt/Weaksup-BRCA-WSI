# -*- coding: utf-8 -*-
"""6.2 池化硬度 × WSI patch 数 N 的最优匹配（文档 6.2）。

文档四问：① 平均与最大之间是否存在最优硬度？② 硬度应随 N 变化吗？
③ 数据驱动的硬度调度（训练中自动调整）？④ 固定池化下误差-N 经验关系 + 理论解释。

矩阵 A（硬度 × N 曲面）：5 池化器（mean / gmean-p5 / quantile-q75 / top-50 / max，
按硬度阶梯排列）× 4 档 N（1000 / 2000 / 5000 / 全量）× 5 折 × 3 种子 = 300 次训练。
  - 嵌套子采样：同一切片按固定随机排列取前缀（1000⊂2000⊂5000⊂全量），N 是唯一变量；
  - 内置对照：q75=固定比例（硬度不随 N 变）vs top-50=固定个数（N 大则等效硬度升）
    ——两者在 N 扫描下的行为差异直接回答问题②。
矩阵 B（课程调度，Duffner & Garcia 2020 复现）：gmean p 随 epoch 线性 1→5，
  对照固定 p=1（数学上=mean）与固定 p=5——两端点复用矩阵 A 全量 N 行（精确相等，
  去重），仅新增 1 组 15 次训练。

协议与 6.3 一致：25 张面积口径标签；关空间约束/可信度/噪声（隔离变量）。

产出：exploration/results/6.2_results.csv + 6.2_hardness_N_surface.png +
      6.2_error_vs_N.png + 6.2_curriculum.png
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs import baseline_m2 as cfg
from src.training import run_cv

RESULTS_DIR = PROJECT_ROOT / "exploration" / "results"
OUT_CSV = RESULTS_DIR / "6.2_results.csv"

PALETTE = ["#1f78b4", "#33a02c", "#e31a1c", "#fe8307", "#6a3d9a",
           "#fb9a99", "#91D1C2"]

N_LEVELS = [1000, 2000, 5000, None]     # None = 全量
POOLERS = [("mean", "mean", 1.0), ("gmean_p5", "gmean", 5.0),
           ("q75", "quantile", 0.75), ("top50", "topk", 50.0), ("max", "max", 1.0)]
CURR_NAME = "gmean_curr_1to5"


def make_cfg(labels_df, max_patches, pooling, param, p_curriculum=None):
    vcfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    vcfg.LABELS_DF = labels_df
    vcfg.CREDIBILITY = None
    vcfg.MAX_PATCHES = max_patches
    vcfg.MODEL_TYPE = {"mean": "mean", "gmean": "gmean", "quantile": "quantile",
                       "topk": "topk", "max": "max"}[pooling]
    vcfg.POOL_PARAM = param
    vcfg.P_CURRICULUM = p_curriculum
    vcfg.LAMBDA_SMOOTH = 0.0
    vcfg.USE_IND = False
    vcfg.USE_PROP = False
    vcfg.AMPLIFY_N = None
    vcfg.NOISE_MODEL = None
    vcfg.NOISE_LEVEL = 0.0
    vcfg.CKPT_DIR = None
    return vcfg


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(PROJECT_ROOT / "data" / "area_reference.csv", encoding="utf-8-sig")
    old25 = sorted(pd.read_csv(PROJECT_ROOT / "TCGA_Download" / "slide_rename_mapping.csv",
                               encoding="utf-8-sig")["slide_id"])
    labels_df = ref[ref.slide_id.isin(old25)][["slide_id", "area_frac_soft"]].rename(
        columns={"area_frac_soft": "weak_fraction"}).reset_index(drop=True)

    rows = []
    for n_lv in N_LEVELS:
        for name, pooling, param in POOLERS:
            tag = f"N={n_lv or 'full'} pool={name}"
            print(f"[6.2] {tag}")
            _, _, metrics = run_cv(make_cfg(labels_df, n_lv, pooling, param))
            rows.append({"N": n_lv or "full", "pooler": name, **metrics})
            print(f"  → pearson={metrics['pearson']:.3f} mae={metrics['mae']:.4f}")

    # 矩阵 B：课程调度（全量 N）；端点 p=1/p=5 复用矩阵 A 的 mean/gmean_p5 行
    print("[6.2] 课程调度 gmean p:1→5（全量 N）")
    _, _, metrics = run_cv(make_cfg(labels_df, None, "gmean", 1.0,
                                    p_curriculum=(1.0, 5.0)))
    rows.append({"N": "full", "pooler": CURR_NAME, **metrics})
    print(f"  → pearson={metrics['pearson']:.3f} mae={metrics['mae']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[6.2] 完成 {len(out)} 组 → {OUT_CSV}；耗时 {(time.time()-t0)/60:.1f} min")
    make_figures(out)


def make_figures(out):
    main5 = out[out.pooler != CURR_NAME].copy()
    n_order = [1000, 2000, 5000, "full"]
    p_order = ["mean", "gmean_p5", "q75", "top50", "max"]

    # 1. 硬度 × N 曲面（Pearson / MAE 双面板热图；子图收窄近方形，利于放大展示）
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for ax, metric, ttl in [(axes[0], "pearson", "Pearson vs truth"),
                            (axes[1], "mae", "MAE")]:
        mat = np.full((len(p_order), len(n_order)), np.nan)
        for i, p in enumerate(p_order):
            for j, n in enumerate(n_order):
                v = main5[(main5.pooler == p) & (main5.N.astype(str) == str(n))]
                if len(v):
                    mat[i, j] = v[metric].iloc[0]
        # R thisplot 风格：Spectral 连续色带（ColorBrewer），NA 为 grey80；
        # 两面板统一"红 = 更优"：Pearson 大值红（Spectral_r），MAE 小值红（Spectral）
        cmap = plt.get_cmap("Spectral_r" if metric == "pearson"
                            else "Spectral").copy()
        cmap.set_bad("#CCCCCC")
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(n_order)), [str(n) for n in n_order])
        ax.set_yticks(range(len(p_order)), p_order)
        ax.set_xlabel("patches per slide (N)"); ax.set_title(ttl)
        norm = plt.Normalize(np.nanmin(mat), np.nanmax(mat))
        for i in range(len(p_order)):
            for j in range(len(n_order)):
                r, g_, b, _ = cmap(norm(mat[i, j]))
                lum = 0.299 * r + 0.587 * g_ + 0.114 * b
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="white" if lum < 0.55 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("6.2: pooling hardness x N surface")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "6.2_hardness_N_surface.png", dpi=150)
    plt.close(fig)

    # 2. 误差-N 经验关系（文档问题④）+ 固定比例 vs 固定个数对照（问题②）
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [1000, 2000, 5000, 30000]   # full 以 ~30k 代表（实际均值见笔记）
    for i, p in enumerate(p_order):
        ys = [main5[(main5.pooler == p) & (main5.N.astype(str) == str(n))]["mae"].iloc[0]
              for n in n_order]
        ax.plot(xs, ys, "o-", color=PALETTE[i], label=p)
    ax.set_xscale("log")
    ax.set_xlabel("patches per slide (N, log scale)"); ax.set_ylabel("MAE")
    ax.set_title("6.2: error vs N (fixed-fraction q75 vs fixed-k top50)")
    ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "6.2_error_vs_N.png", dpi=150)
    plt.close(fig)

    # 3. 课程调度对比（问题③）
    fig, ax = plt.subplots(figsize=(6, 4.5))
    p1 = main5[(main5.pooler == "mean") & (main5.N == "full")].iloc[0]
    p5 = main5[(main5.pooler == "gmean_p5") & (main5.N == "full")].iloc[0]
    cur = out[out.pooler == CURR_NAME].iloc[0]
    names = ["fixed p=1\n(=mean)", "fixed p=5", "curriculum\np:1->5"]
    vals = [p1["pearson"], p5["pearson"], cur["pearson"]]
    maes = [p1["mae"], p5["mae"], cur["mae"]]
    x = np.arange(3)
    ax.bar(x - 0.2, vals, 0.4, color=PALETTE[0], label="Pearson")
    ax.bar(x + 0.2, maes, 0.4, color=PALETTE[1], label="MAE")
    ax.set_xticks(x, names); ax.set_title("6.2: curriculum hardness scheduling (full N)")
    ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "6.2_curriculum.png", dpi=150)
    plt.close(fig)
    print("[6.2] 三张图已存")


if __name__ == "__main__":
    main()
