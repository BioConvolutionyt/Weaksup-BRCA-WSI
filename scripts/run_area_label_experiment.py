# -*- coding: utf-8 -*-
"""面积口径验证实验：三组相关分析 + 面积口径重训（25/75 两版）。

输入：data/area_reference.csv（WSInfer 面积参照）+ data/weak_labels.csv（GDC 弱标签）
      + outputs/logs/predictions_75_credibility.csv（我们模型 75 张 OOF 预测）。

产出：
  outputs/logs/area_correlations.csv      —— 三组相关（Pearson+Spearman）
  outputs/logs/area_label_experiment.csv  —— 面积口径 vs GDC 口径 × 25/75 的训练指标
"""
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs import baseline_m2 as cfg
from src.training import run_cv

OUT_CORR = cfg.OUTPUTS_DIR / "logs" / "area_correlations.csv"
OUT_EXP = cfg.OUTPUTS_DIR / "logs" / "area_label_experiment.csv"


def make_cfg(labels_df=None, credibility=None):
    vcfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    vcfg.LABELS_DF = labels_df
    vcfg.CREDIBILITY = credibility
    vcfg.CKPT_DIR = None
    # 基线 = M4 最优聚合器：分位聚合 q75（2026-08-22 起实验十改用此基线）
    vcfg.MODEL_TYPE = "quantile"
    vcfg.POOL_PARAM = 0.75
    return vcfg


def credibility_from_residuals(preds_df):
    r = preds_df["absolute_error"].values.astype(float)
    tau = np.median(r) + 1e-12
    c = np.exp(-r / tau)
    return dict(zip(preds_df["slide_id"], c / c.mean()))


def corr_pair(x, y):
    return round(pearsonr(x, y)[0], 4), round(spearmanr(x, y)[0], 4)


def main():
    t0 = time.time()
    # 0. 汇总 WSInfer 产出 → data/area_reference.csv（自包含；含救援后的 A5RX）
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_area_reference import aggregate, RESULTS_MAIN, RESULTS_SUPP, OUT_CSV
    rows = aggregate(RESULTS_MAIN, "wsinfer_breast_tumor_resnet34") + \
           aggregate(RESULTS_SUPP, "wsinfer_breast_tumor_resnet34")
    ref_all = pd.DataFrame(rows).drop_duplicates(subset="slide_id")
    ref_all.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[REF] 面积参照 {len(ref_all)}/75 → {OUT_CSV}")

    ref = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
    gdc = pd.read_csv(PROJECT_ROOT / "data" / "weak_labels.csv", encoding="utf-8-sig")

    # 0.5 以 q75 基线重新生成 75 张 GDC 口径 OOF 预测（两轮自闭合：残差→可信度→加权）
    print("[PRED75] q75 基线重新生成 75 张预测")
    preds_q75_0, _, _ = run_cv(make_cfg())                      # q75 不加权
    cred_q75 = credibility_from_residuals(preds_q75_0)
    pred75, _, m_pred75 = run_cv(make_cfg(credibility=cred_q75))  # q75 + 可信度
    pred75.to_csv(cfg.OUTPUTS_DIR / "logs" / "predictions_75_q75.csv",
                  index=False, encoding="utf-8-sig")
    print(f"  75 张 q75+可信度 指标: {m_pred75}")

    # ---------- 1. 三组相关（75 张水平） ----------
    m = ref.merge(gdc[["slide_id", "weak_fraction"]], on="slide_id") \
           .merge(pred75[["slide_id", "predicted_fraction"]], on="slide_id")
    print(f"[CORR] 三方对齐 n={len(m)}")
    rows = [
        {"pair": "A: 我们的预测 vs 面积参照", "n": len(m),
         "pearson": corr_pair(m["predicted_fraction"], m["area_frac_soft"])[0],
         "spearman": corr_pair(m["predicted_fraction"], m["area_frac_soft"])[1]},
        {"pair": "B: 我们的预测 vs GDC 弱标签", "n": len(m),
         "pearson": corr_pair(m["predicted_fraction"], m["weak_fraction"])[0],
         "spearman": corr_pair(m["predicted_fraction"], m["weak_fraction"])[1]},
        {"pair": "C: 面积参照 vs GDC 弱标签", "n": len(m),
         "pearson": corr_pair(m["area_frac_soft"], m["weak_fraction"])[0],
         "spearman": corr_pair(m["area_frac_soft"], m["weak_fraction"])[1]},
    ]
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(OUT_CORR, index=False, encoding="utf-8-sig")
    print(corr_df.to_string(index=False))

    # ---------- 2. 面积口径重训（25/75 两版） ----------
    old25 = set(pd.read_csv(PROJECT_ROOT / "TCGA_Download" / "slide_rename_mapping.csv",
                            encoding="utf-8-sig")["slide_id"])
    area_labels = ref[["slide_id", "area_frac_soft"]].rename(
        columns={"area_frac_soft": "weak_fraction"})
    exp_rows = []
    for cohort, sids in [("25", sorted(old25)), ("75", None)]:
        labels_df = area_labels if sids is None else area_labels[area_labels.slide_id.isin(sids)]
        # 第一轮：不加权出残差
        preds0, _, m0 = run_cv(make_cfg(labels_df=labels_df))
        cred = credibility_from_residuals(preds0)
        # 第二轮：可信度加权（与基线配置一致）
        _, _, m1 = run_cv(make_cfg(labels_df=labels_df, credibility=cred))
        exp_rows.append({"label_source": "area_reference", "cohort": cohort,
                         "n": len(labels_df), **m1})
        print(f"  面积口径/{cohort}张: {m1}")

    # 参照行：GDC 标签 + q75 基线版（既有结果：25 张=M4 交付指标；75 张=扩量实验 Run D）
    exp_rows.append({"label_source": "gdc_cells(参照)", "cohort": "25", "n": 25,
                     "pearson": 0.267, "spearman": 0.250, "mae": 0.114, "rmse": 0.149})
    exp_rows.append({"label_source": "gdc_cells(参照)", "cohort": "75", "n": 75,
                     "pearson": 0.004, "spearman": 0.034, "mae": 0.143, "rmse": 0.192})
    out = pd.DataFrame(exp_rows)
    out.to_csv(OUT_EXP, index=False, encoding="utf-8-sig")
    print("\n" + out.round(4).to_string(index=False))
    print(f"[AREA] 耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
