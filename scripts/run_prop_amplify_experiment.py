# -*- coding: utf-8 -*-
"""实验十三：图标签传播（候选 C）与低占比端放大（候选 D）组合实验。

设计（用户指定）：
  基座 = 面积参照 + 分位聚合 q75 + 可信度加权（可信度按队列首轮自闭合）；
  变体 = 平滑+传播 / 平滑+放大；队列 = 25 / 75；
  若两变体均改善空间指标（连通域数/平滑度）→ 加测 平滑+传播+放大 组合。
参照行 = 仅平滑（实验十一 V1 同构，本脚本内重跑以保证同协议可比）。

产出：outputs/logs/prop_amplify_experiment.csv
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

OUT_CSV = cfg.OUTPUTS_DIR / "logs" / "prop_amplify_experiment.csv"

VARIANTS = [
    {"name": "ref_smooth",        "USE_PROP": False, "AMPLIFY_N": None},
    {"name": "smooth_prop",       "USE_PROP": True,  "AMPLIFY_N": None},
    {"name": "smooth_amplify",    "USE_PROP": False, "AMPLIFY_N": 5},
    {"name": "smooth_prop_amplify", "USE_PROP": True, "AMPLIFY_N": 5},   # 条件触发
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
    vcfg.USE_PROP = v["USE_PROP"]
    vcfg.AMPLIFY_N = v["AMPLIFY_N"]
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
        preds0, _, _ = run_cv(make_cfg(labels_df, None, VARIANTS[0]))
        cred = credibility_from_residuals(preds0)

        cohort_rows = {}
        for v in VARIANTS[:3]:  # 先跑参照 + 两个单变体
            print(f"[EXP13] cohort={cohort} 变体={v['name']}")
            _, oof_scores, metrics = run_cv(make_cfg(labels_df, cred, v))
            n_comp, smooth = spatial_metrics(oof_scores, cfg)
            row = {"cohort": cohort, "variant": v["name"], "n": len(labels_df),
                   **metrics, "mean_components": n_comp, "mean_neighbor_diff": smooth}
            rows.append(row)
            cohort_rows[v["name"]] = row
            print(f"  → {metrics} | 连通域={n_comp:.1f} 平滑度={smooth:.4f}")

        # 条件触发：两单变体均改善空间指标（连通域或平滑度优于参照）才测组合
        r0 = cohort_rows["ref_smooth"]
        def improved(name):
            r = cohort_rows[name]
            return (r["mean_components"] < r0["mean_components"]
                    or r["mean_neighbor_diff"] < r0["mean_neighbor_diff"])
        if improved("smooth_prop") and improved("smooth_amplify"):
            v = VARIANTS[3]
            print(f"[EXP13] cohort={cohort} 条件满足，加测组合 {v['name']}")
            _, oof_scores, metrics = run_cv(make_cfg(labels_df, cred, v))
            n_comp, smooth = spatial_metrics(oof_scores, cfg)
            rows.append({"cohort": cohort, "variant": v["name"], "n": len(labels_df),
                         **metrics, "mean_components": n_comp, "mean_neighbor_diff": smooth})
            print(f"  → {metrics} | 连通域={n_comp:.1f} 平滑度={smooth:.4f}")
        else:
            print(f"[EXP13] cohort={cohort} 条件未满足，跳过组合变体")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n" + out.round(4).to_string(index=False))
    print(f"[EXP13] 耗时 {(time.time()-t0)/60:.1f} min → {OUT_CSV}")


if __name__ == "__main__":
    main()
