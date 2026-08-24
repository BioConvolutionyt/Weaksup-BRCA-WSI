# -*- coding: utf-8 -*-
"""M2 训练入口：5-fold × 3 种子基线训练 + OOF 指标 + 散点图 + 分数图落盘。

运行：python scripts/train_baseline.py
产出：
  outputs/predictions.csv        —— slide_id, weak_fraction, predicted_fraction, absolute_error
  outputs/regression_plot.png    —— 预测 vs 弱标签散点图
  outputs/logs/scores/*.npy      —— 每切片 OOF patch 分数（make_heatmaps.py 的输入）
  outputs/logs/train_baseline.log—— 训练日志
"""
import json
import sys
import time
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


def log(msg):
    line = str(msg)
    print(line, flush=True)
    cfg.TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.TRAIN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_regression_plot(preds_df, metrics):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(preds_df["weak_fraction"], preds_df["predicted_fraction"],
               s=45, edgecolors="k", linewidths=0.5, alpha=0.85)
    lim = [0, 1.05]
    ax.plot(lim, lim, "r--", lw=1, label="ideal")
    ax.set_xlabel("weak_fraction (GDC biospecimen)")
    ax.set_ylabel("predicted_fraction (OOF)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"Pearson={metrics['pearson']:.3f}  Spearman={metrics['spearman']:.3f}\n"
                 f"MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.REGRESSION_PLOT, dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    cfg.OUTPUTS_DIR.mkdir(exist_ok=True)
    cfg.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    if cfg.TRAIN_LOG.exists():
        cfg.TRAIN_LOG.unlink()

    log(f"[M2] 设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    preds_df, oof_scores, metrics = run_cv(cfg, log=log)

    preds_df.to_csv(cfg.PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
    for sid, sc in oof_scores.items():
        np.save(cfg.SCORES_DIR / f"{sid}.npy", sc.astype(np.float32))
    save_regression_plot(preds_df, metrics)

    log(f"[M2] 指标: {json.dumps(metrics, ensure_ascii=False)}")
    log(f"[M2] 耗时 {(time.time()-t0)/60:.1f} min")
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    import torch  # 供 log 中的设备判断
    main()
