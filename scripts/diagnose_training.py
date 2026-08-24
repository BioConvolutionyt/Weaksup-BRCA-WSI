# -*- coding: utf-8 -*-
"""训练动力学诊断（对最终交付配置）。

记录 5 折 × 3 种子下逐 epoch 的：
  1) 训练/验证 MSE 曲线（均值 ± 标准差带）——判断收敛与过拟合走向；
  2) 逐 Linear 层梯度范数曲线（对数轴）——判断梯度爆炸/消失。

产出：
  outputs/logs/training_history.csv
  outputs/figures/training_loss_curves.png
  outputs/figures/gradient_norm_curves.png
"""
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs import spatial_m3 as cfg
from src.training import run_cv

LOSS_FIG = cfg.OUTPUTS_DIR / "figures" / "training_loss_curves.png"
GRAD_FIG = cfg.OUTPUTS_DIR / "figures" / "gradient_norm_curves.png"
HIST_CSV = cfg.OUTPUTS_DIR / "logs" / "training_history.csv"


def build_cfg():
    vcfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    cred_df = pd.read_csv(cfg.OUTPUTS_DIR / "logs" / "label_credibility.csv")
    cred = cred_df[cred_df.label_field == "percent_tumor_cells"]
    vcfg.CREDIBILITY = dict(zip(cred.slide_id, cred.credibility))
    vcfg.LAMBDA_SMOOTH = 0.01
    vcfg.USE_IND = False
    vcfg.RECORD_HISTORY = True
    vcfg.CKPT_DIR = None
    return vcfg


def plot_loss(hist):
    epochs = sorted(hist["epoch"].unique())
    tr_mean, tr_std, va_mean, va_std, alive = [], [], [], [], []
    for e in epochs:
        sub = hist[hist["epoch"] == e]
        tr_mean.append(sub["train_loss"].mean()); tr_std.append(sub["train_loss"].std())
        va_mean.append(sub["val_loss"].mean()); va_std.append(sub["val_loss"].std())
        alive.append(len(sub))
    epochs = np.array(epochs)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, tr_mean, label="train MSE (mean)", color="tab:blue")
    ax.fill_between(epochs, np.array(tr_mean) - np.array(tr_std), np.array(tr_mean) + np.array(tr_std),
                    color="tab:blue", alpha=0.2)
    ax.plot(epochs, va_mean, label="val MSE (mean)", color="tab:orange")
    ax.fill_between(epochs, np.array(va_mean) - np.array(va_std), np.array(va_mean) + np.array(va_std),
                    color="tab:orange", alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(epochs, alive, color="gray", ls=":", lw=1, label="仍在训练的模型数")
    ax2.set_ylabel("alive models", color="gray")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    ax.set_title("Training dynamics: train/val MSE (5 folds × 3 seeds, mean ± std)")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(LOSS_FIG, dpi=150); plt.close(fig)


def plot_grad(hist):
    grad_cols = [c for c in hist.columns if c.startswith("grad_")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in grad_cols:
        g = hist.groupby("epoch")[c].mean()
        ax.plot(g.index, g.values, label=c.replace("grad_", "grad ") + " (mean)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("gradient norm (log scale)")
    ax.set_title("Per-layer gradient norms (5 folds × 3 seeds, mean)")
    ax.legend()
    fig.tight_layout(); fig.savefig(GRAD_FIG, dpi=150); plt.close(fig)


def main():
    vcfg = build_cfg()
    _, _, metrics, histories = run_cv(vcfg)
    hist = pd.DataFrame(histories)
    hist = hist[["fold", "seed", "epoch", "train_loss", "val_loss"]
                + [c for c in hist.columns if c.startswith("grad_")]]
    HIST_CSV.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(HIST_CSV, index=False, encoding="utf-8-sig")
    LOSS_FIG.parent.mkdir(parents=True, exist_ok=True)
    plot_loss(hist)
    plot_grad(hist)
    print(f"[DIAG] 历史 {len(hist)} 行（{hist[['fold','seed']].drop_duplicates().shape[0]} 模型）；"
          f"指标复核 {metrics}")
    print(f"[DIAG] 图：{LOSS_FIG.name}, {GRAD_FIG.name}")


if __name__ == "__main__":
    main()
