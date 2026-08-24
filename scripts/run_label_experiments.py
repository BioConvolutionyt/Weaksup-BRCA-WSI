# -*- coding: utf-8 -*-
"""M3.5 标签侧实验入口：2 标签口径 × {无加权, 可信度加权} = 4 组对照。

流程（每组标签口径）：
  1) 不加权训练（5-fold × 3 种子）→ OOF 指标 + 逐切片残差；
  2) 由残差计算可信度 c_i = exp(-r_i/τ)，τ = 残差中位数，并归一化到均值 1；
  3) 可信度加权重训 → OOF 指标。
  （B1 组的可信度来自本口径自己的不加权残差——自闭合两轮，符合 PANDA 清洗逻辑。）

产出：
  outputs/logs/label_experiments.csv  —— 4 组 × 4 指标
  outputs/logs/label_credibility.csv  —— 逐切片可信度（两口径各一份）
"""
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import label_m35 as cfg
from src.training import run_cv

LABEL_FIELDS = ["percent_tumor_cells", "percent_tumor_nuclei"]


def make_cfg(field, credibility):
    vcfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    vcfg.LABEL_FIELD = field
    vcfg.CREDIBILITY = credibility
    vcfg.CKPT_DIR = None
    return vcfg


def credibility_from_residuals(preds_df, tau=None):
    """PANDA 软清洗的软化版：残差 → 指数衰减可信度，归一化至均值 1。"""
    r = preds_df["absolute_error"].values.astype(float)
    if tau is None:
        tau = np.median(r) + 1e-12
    c = np.exp(-r / tau)
    c = c / c.mean()
    return dict(zip(preds_df["slide_id"], c)), tau


def main():
    t0 = time.time()
    rows = []
    cred_records = []
    for field in LABEL_FIELDS:
        print(f"\n[M3.5] 标签口径: {field}")

        preds0, _, m0 = run_cv(make_cfg(field, None))
        rows.append({"label_field": field, "weighting": "none", **m0})
        print(f"  不加权: {m0}")

        cred, tau = credibility_from_residuals(preds0, cfg.CREDIBILITY_TAU)
        for sid, c in cred.items():
            cred_records.append({"label_field": field, "slide_id": sid,
                                 "credibility": round(c, 4)})
        print(f"  可信度: τ={tau:.4f}，min={min(cred.values()):.3f}，max={max(cred.values()):.3f}")

        preds1, _, m1 = run_cv(make_cfg(field, cred))
        rows.append({"label_field": field, "weighting": "credibility", **m1})
        print(f"  可信度加权: {m1}")

    out = pd.DataFrame(rows)
    out_csv = cfg.OUTPUTS_DIR / "logs" / "label_experiments.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(cred_records).to_csv(
        cfg.OUTPUTS_DIR / "logs" / "label_credibility.csv", index=False, encoding="utf-8-sig")

    print("\n" + out.round(4).to_string(index=False))
    print(f"[M3.5] 耗时 {(time.time()-t0)/60:.1f} min；结果 → {out_csv}")


if __name__ == "__main__":
    main()
