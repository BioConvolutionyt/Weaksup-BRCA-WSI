# -*- coding: utf-8 -*-
"""6.3 硬度-方差对照图：有效参与实例数 n_eff vs 噪声下的误差退化。

理论预测（推导见探索笔记）：袋级噪声方差 σ² 经池化反传的参数扰动 ∝ σ²/n_eff，
故误差退化（噪声 MAE − 干净 MAE）应随 1/n_eff 增大而增大。

n_eff 实证计算：加载各池化干净基线 fold0/seed0 权重，对若干参考切片用
参与率 (Σ|∂pred/∂s_i|)²/Σ(∂pred/∂s_i)² 估计。

产出：exploration/results/6.3_hardness_variance.png + 6.3_n_eff.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from configs import baseline_m2 as cfg
from src.models import make_model
from run_explore_63 import POOLERS, n_eff_of  # scripts 目录同源复用

RESULTS_DIR = PROJECT_ROOT / "exploration" / "results"
CKPT_DIR = RESULTS_DIR / "ckpts_63"


def main():
    out = pd.read_csv(RESULTS_DIR / "6.3_results.csv")
    ref = pd.read_csv(PROJECT_ROOT / "data" / "area_reference.csv", encoding="utf-8-sig")
    old25 = sorted(pd.read_csv(PROJECT_ROOT / "TCGA_Download" / "slide_rename_mapping.csv",
                               encoding="utf-8-sig")["slide_id"])
    sids = old25[:5]  # 5 张参考切片估计 n_eff
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for pooler in POOLERS:
        name = pooler["name"]
        ck = CKPT_DIR / name / "fold0_seed0.pt"
        if not ck.exists():
            print(f"  {name}: 缺权重，跳过")
            continue
        vcfg = types.SimpleNamespace(
            IN_DIM=cfg.IN_DIM, HIDDEN=cfg.HIDDEN, DROPOUT=cfg.DROPOUT,
            MODEL_TYPE=pooler["MODEL_TYPE"],
            POOL_PARAM=pooler["POOL_PARAM"] if pooler["POOL_PARAM"] is not None else 1.0,
            ATTN_HIDDEN=256, OUTPUT_MODE="sigmoid")
        model = make_model(vcfg)
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        model.eval().to(device)
        neffs = []
        for sid in sids:
            x = torch.load(cfg.FEATURES_DIR / f"{sid}.pt", map_location="cpu",
                           weights_only=True)
            neffs.append(n_eff_of(model, x, device))
        n_eff = float(np.mean(neffs))
        # 误差退化：最高均匀噪声档的 MAE 增量
        sub = out[out.pooler == name]
        clean = sub[sub.noise_model == "clean"]["mae"].iloc[0]
        deg = sub[(sub.noise_model == "uniform") & (sub.noise_level == 0.3)]["mae"].iloc[0] - clean
        rows.append({"pooler": name, "n_eff": n_eff, "mae_deg_u03": deg})
        print(f"  {name}: n_eff={n_eff:.0f}，MAE 增量(u0.3)={deg:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "6.3_n_eff.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = 1.0 / df["n_eff"]
    ax.scatter(x, df["mae_deg_u03"], s=60)
    for _, r in df.iterrows():
        ax.annotate(r["pooler"], (1 / r["n_eff"], r["mae_deg_u03"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("1 / n_eff (log scale; harder pooling to the right)")
    ax.set_ylabel("MAE degradation (uniform a=0.3 minus clean)")
    ax.set_title("6.3 hardness-variance: harder pooling degrades more under noise")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "6.3_hardness_variance.png", dpi=150)
    plt.close(fig)
    print(f"[6.3-hardness] 图与表已存 → {RESULTS_DIR}")


if __name__ == "__main__":
    import types  # noqa: 供 main 内使用
    main()
