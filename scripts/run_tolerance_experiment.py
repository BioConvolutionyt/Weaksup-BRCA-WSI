# -*- coding: utf-8 -*-
"""实验十二：容差式面积约束（候选 B）对照实验。

设计：基座 = 面积参照标签 + 分位聚合 q75 + 可信度加权 + 平滑 λ=0.01（当前交付形态）；
损失形态对照 = MSE（参照）vs log-barrier 容差 δ∈{0.03, 0.05, 0.10}（Silva-Rodríguez 公式）；
队列 = 25 / 75 张。动机：面积参照本身含噪，容差约束使训练对参照误差鲁棒（仅越界才惩罚）。

产出：outputs/logs/tolerance_experiment.csv
"""
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from configs import spatial_m3 as cfg
from src.training import run_cv
from src.spatial import spatial_metrics

OUT_CSV = cfg.OUTPUTS_DIR / "logs" / "tolerance_experiment.csv"

LOSS_VARIANTS = [
    {"name": "mse_ref",   "LOSS_TYPE": "mse",       "TOL_DELTA": None},
    {"name": "tol_0.03",  "LOSS_TYPE": "tolerance", "TOL_DELTA": 0.03},
    {"name": "tol_0.05",  "LOSS_TYPE": "tolerance", "TOL_DELTA": 0.05},
    {"name": "tol_0.10",  "LOSS_TYPE": "tolerance", "TOL_DELTA": 0.10},
]


def make_cfg(labels_df, credibility, v):
    vcfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    vcfg.LABELS_DF = labels_df
    vcfg.CREDIBILITY = credibility
    vcfg.LAMBDA_SMOOTH = 0.01
    vcfg.USE_IND = False
    vcfg.MODEL_TYPE = "quantile"
    vcfg.POOL_PARAM = 0.75
    vcfg.CKPT_DIR = None
    vcfg.LOSS_TYPE = v["LOSS_TYPE"]
    if v["TOL_DELTA"] is not None:
        vcfg.TOL_DELTA = v["TOL_DELTA"]
    return vcfg


def credibility_from_residuals(preds_df):
    r = preds_df["absolute_error"].values.astype(float)
    tau = np.median(r) + 1e-12
    c = np.exp(-r / tau)
    return dict(zip(preds_df["slide_id"], c / c.mean()))


def main():
    t0 = time.time()
    ref = pd.read_csv(PROJECT_ROOT / "data" / "area_reference.csv", encoding="utf-8-sig")
    area_labels = ref[["slide_id", "area_frac_soft"]].rename(
        columns={"area_frac_soft": "weak_fraction"})
    old25 = sorted(pd.read_csv(PROJECT_ROOT / "TCGA_Download" / "slide_rename_mapping.csv",
                               encoding="utf-8-sig")["slide_id"])

    rows = []
    for cohort, sids in [("25", old25), ("75", None)]:
        labels_df = area_labels if sids is None else area_labels[area_labels.slide_id.isin(sids)]
        # 可信度按队列自闭合（MSE 首轮残差，四个损失变体共用）
        preds0, _, _ = run_cv(make_cfg(labels_df, None, LOSS_VARIANTS[0]))
        cred = credibility_from_residuals(preds0)

        for v in LOSS_VARIANTS:
            print(f"[EXP12] cohort={cohort} 损失={v['name']}")
            _, oof_scores, metrics = run_cv(make_cfg(labels_df, cred, v))
            n_comp, smooth = spatial_metrics(oof_scores, cfg)
            rows.append({"cohort": cohort, "loss": v["name"], "n": len(labels_df),
                         **metrics, "mean_components": n_comp,
                         "mean_neighbor_diff": smooth})
            print(f"  → {metrics} | 连通域={n_comp:.1f} 平滑度={smooth:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n" + out.round(4).to_string(index=False))
    print(f"[EXP12] 耗时 {(time.time()-t0)/60:.1f} min → {OUT_CSV}")


if __name__ == "__main__":
    main()
