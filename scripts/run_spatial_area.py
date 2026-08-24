# -*- coding: utf-8 -*-
"""实验十一：面积口径下重做实验五（M3 空间约束对照）。

设计：面积参照为弱标签（已获批准）；队列 = 25 张 / 75 张；
变体 = V0 无空间约束 / V1 仅平滑(λ=0.01) / V2 平滑+InD——结构对齐实验五。
基座模型 = 当前交付形态：分位聚合 q75 + 可信度加权（可信度按队列首轮残差自闭合）。

产出：outputs/logs/spatial_variants_area.csv
  （cohort × variant × [pearson, spearman, mae, rmse, mean_components, mean_neighbor_diff]）
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

OUT_CSV = cfg.OUTPUTS_DIR / "logs" / "spatial_variants_area.csv"

VARIANTS = [
    {"name": "V0_none", "LAMBDA_SMOOTH": 0.0, "USE_IND": False},
    {"name": "V1_smooth", "LAMBDA_SMOOTH": 0.01, "USE_IND": False},
    {"name": "V2_smooth_ind", "LAMBDA_SMOOTH": 0.01, "USE_IND": True},
]


def make_cfg(labels_df, credibility, lam, use_ind):
    vcfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    vcfg.LABELS_DF = labels_df
    vcfg.CREDIBILITY = credibility
    vcfg.LAMBDA_SMOOTH = lam
    vcfg.USE_IND = use_ind
    vcfg.MODEL_TYPE = "quantile"
    vcfg.POOL_PARAM = 0.75
    vcfg.CKPT_DIR = None
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
        # 可信度按队列自闭合：V0 不加权首轮 → 残差 → 可信度（三个变体共用）
        preds0, _, _ = run_cv(make_cfg(labels_df, None, 0.0, False))
        cred = credibility_from_residuals(preds0)

        for v in VARIANTS:
            print(f"[EXP11] cohort={cohort} 变体={v['name']}")
            _, oof_scores, metrics = run_cv(
                make_cfg(labels_df, cred, v["LAMBDA_SMOOTH"], v["USE_IND"]))
            n_comp, smooth = spatial_metrics(oof_scores, cfg)
            rows.append({"cohort": cohort, "variant": v["name"], "n": len(labels_df),
                         **metrics, "mean_components": n_comp,
                         "mean_neighbor_diff": smooth})
            print(f"  → {metrics} | 连通域={n_comp:.1f} 平滑度={smooth:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n" + out.round(4).to_string(index=False))
    print(f"[EXP11] 耗时 {(time.time()-t0)/60:.1f} min → {OUT_CSV}")


if __name__ == "__main__":
    main()
