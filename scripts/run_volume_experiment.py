# -*- coding: utf-8 -*-
"""补充数据扩展对比实验（25 → 75 张）：检验数据量是否为性能瓶颈。

对照设计（同一 5-fold × 3 种子协议；标签口径均为 percent_tumor_cells）：
  Run A（既有，25 张 cells+可信度）：Pearson 0.227 / Spearman 0.170 / MAE 0.115 / RMSE 0.153
  Run B：75 张 cells 不加权 —— 隔离纯数据量效应
  Run C：75 张 cells + 可信度 —— 两轮自闭合（首轮残差→可信度→加权重训）
  Run D：75 张 + 分位聚合 q75 + 可信度 —— 当前交付聚合器在扩量后的参照行

产出：outputs/logs/volume_experiment.csv；outputs/logs/label_credibility_75.csv
"""
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs import baseline_m2 as cfg
from src.training import run_cv

OUT_CSV = cfg.OUTPUTS_DIR / "logs" / "volume_experiment.csv"
CRED_CSV = cfg.OUTPUTS_DIR / "logs" / "label_credibility_75.csv"


def make_cfg(**over):
    vcfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    for k, v in over.items():
        setattr(vcfg, k, v)
    vcfg.CKPT_DIR = None
    return vcfg


def credibility_from_residuals(preds_df):
    r = preds_df["absolute_error"].values.astype(float)
    tau = np.median(r) + 1e-12
    c = np.exp(-r / tau)
    c = c / c.mean()
    return dict(zip(preds_df["slide_id"], c)), tau


def main():
    t0 = time.time()
    rows = []

    print("[Run B] 75 张 / cells / 不加权")
    predsB, _, mB = run_cv(make_cfg(CREDIBILITY=None))
    rows.append({"run": "B_75_unweighted", **mB})
    print("  →", mB)

    cred, tau = credibility_from_residuals(predsB)
    pd.DataFrame([{"slide_id": k, "credibility": round(v, 4)} for k, v in cred.items()]
                 ).to_csv(CRED_CSV, index=False, encoding="utf-8-sig")
    print(f"  可信度 τ={tau:.4f}")

    print("[Run C] 75 张 / cells / 可信度加权")
    _, _, mC = run_cv(make_cfg(CREDIBILITY=cred))
    rows.append({"run": "C_75_credibility", **mC})
    print("  →", mC)

    print("[Run D] 75 张 / cells / 可信度 + 分位聚合 q75")
    _, _, mD = run_cv(make_cfg(CREDIBILITY=cred, MODEL_TYPE="quantile", POOL_PARAM=0.75))
    rows.append({"run": "D_75_credibility_q75", **mD})
    print("  →", mD)

    rows.insert(0, {"run": "A_25_credibility(既有)", "pearson": 0.227, "spearman": 0.170,
                    "mae": 0.115, "rmse": 0.153})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n" + out.round(4).to_string(index=False))
    print(f"[VOL] 耗时 {(time.time()-t0)/60:.1f} min → {OUT_CSV}")


if __name__ == "__main__":
    main()
