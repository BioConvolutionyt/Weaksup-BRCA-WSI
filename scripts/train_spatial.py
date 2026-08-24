# -*- coding: utf-8 -*-
"""M3 训练入口：V0–V2 空间约束变体对照（V3 = V2 + 形态学后处理，不重训）。

运行：python scripts/train_spatial.py
产出：
  outputs/logs/spatial_variants.csv —— 变体 × (占比指标 + 空间质量指标)
  outputs/predictions.csv           —— 更新为交付变体（V2）的 OOF 预测
  outputs/logs/scores/*.npy         —— 更新为交付变体的 patch 分数（供热图/mask）
  outputs/logs/checkpoints/*.pt     —— 交付变体每折权重（SRG 复算用）
"""
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import spatial_m3 as cfg
from src.spatial import count_components, mean_neighbor_diff, scores_to_grid
from src.training import run_cv


def log(msg):
    print(msg, flush=True)


def spatial_metrics(oof_scores, cfg):
    """每变体的空间质量指标：连通域数（0.5 mask）与平均邻域分数差。"""
    comps, smooths = [], []
    for sid, sc in oof_scores.items():
        coords = pd.read_csv(cfg.COORDS_DIR / f"{sid}.csv")
        grid = scores_to_grid(sc, coords)
        comps.append(count_components(np.nan_to_num(grid, nan=0.0) >= cfg.MASK_THRESHOLD))
        smooths.append(mean_neighbor_diff(grid))
    return float(np.mean(comps)), float(np.mean(smooths))


def main():
    t0 = time.time()
    rows = []
    deliverable = None
    for v in cfg.VARIANTS:
        vcfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
        vcfg.LAMBDA_SMOOTH = v["LAMBDA_SMOOTH"]
        vcfg.USE_IND = v["USE_IND"]
        # 仅交付变体保存权重（SRG 复算用），控制文件数量
        vcfg.CKPT_DIR = cfg.CKPT_DIR if v["name"] == cfg.DELIVERABLE_VARIANT else None

        log(f"[M3] 训练变体 {v['name']}（λ={vcfg.LAMBDA_SMOOTH}, InD={vcfg.USE_IND}）")
        preds_df, oof_scores, metrics = run_cv(vcfg, log=log)
        n_comp, smooth = spatial_metrics(oof_scores, vcfg)
        rows.append({"variant": v["name"], **metrics,
                     "mean_components": n_comp, "mean_neighbor_diff": smooth})
        log(f"  → {metrics} | 连通域={n_comp:.1f} 平滑度={smooth:.4f}")

        if v["name"] == cfg.DELIVERABLE_VARIANT:
            deliverable = (preds_df, oof_scores)

    out = pd.DataFrame(rows)
    out_csv = cfg.OUTPUTS_DIR / "logs" / "spatial_variants.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 交付变体落盘
    preds_df, oof_scores = deliverable
    preds_df.to_csv(cfg.PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
    cfg.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    for sid, sc in oof_scores.items():
        np.save(cfg.SCORES_DIR / f"{sid}.npy", sc.astype(np.float32))

    print(out.round(4).to_string(index=False))
    print(f"[M3] 耗时 {(time.time()-t0)/60:.1f} min；变体表 → {out_csv}")


if __name__ == "__main__":
    main()
