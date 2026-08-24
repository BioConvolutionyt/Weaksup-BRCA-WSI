# -*- coding: utf-8 -*-
"""M4 消融统一入口：聚合器家族对照 + OOF 仿射校准 + 范式对照（attention vs 分数）。

运行：python scripts/run_ablations.py
产出：
  outputs/logs/ablation_aggregators.csv   —— 9 变体 × 4 指标（含瓶颈 2 分位聚合扫描）
  outputs/logs/calibration.csv            —— 校准前后双口径指标
  outputs/figures/regression_plot_calibrated.png
  outputs/logs/srg_paradigm.csv           —— attention 序 vs 分数序 SRG 逐切片对比
  outputs/logs/attn_scores/*.npy          —— ABMIL attention 权重图（逐切片）
  outputs/figures/paradigm_compare_*.png  —— 2 张 attention/分数并排对比图
  outputs/logs/checkpoints_abmil/*.pt     —— ABMIL 变体权重（SRG 复算用）
"""
import sys
import time
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs import ablation_m4 as cfg
from src.models import ABMILInstanceMIL, MeanPoolMIL
from src.training import fold_split, load_data, make_folds, run_cv


def log(msg):
    print(msg, flush=True)


def make_vcfg(v, credibility, ckpt_dir=None):
    vcfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    vcfg.MODEL_TYPE = v["MODEL_TYPE"]
    if v["POOL_PARAM"] is not None:
        vcfg.POOL_PARAM = v["POOL_PARAM"]
    vcfg.CREDIBILITY = credibility
    vcfg.CKPT_DIR = ckpt_dir
    return vcfg


# ---------- 1. 聚合器家族对照 ----------
def run_aggregators(credibility):
    rows = []
    for v in cfg.AGGREGATOR_VARIANTS:
        # 仅 ABMIL 变体保存权重（范式对照复算用）
        ckpt = cfg.OUTPUTS_DIR / "logs" / "checkpoints_abmil" if v["name"] == cfg.PARADIGM_VARIANT else None
        log(f"[M4-B] 训练变体 {v['name']}")
        _, _, metrics = run_cv(make_vcfg(v, credibility, ckpt))
        rows.append({"variant": v["name"], **metrics})
        log(f"  → {metrics}")
    out = pd.DataFrame(rows)
    out_csv = cfg.OUTPUTS_DIR / "logs" / "ablation_aggregators.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    log(f"[M4-B] 聚合器对照 → {out_csv}")
    return out


# ---------- 2. OOF 仿射校准 ----------
def run_calibration():
    """每折：用该折训练集的 (ensemble pred, weak) 拟合仿射，应用到该折测试 OOF 预测。"""
    vcfg = make_vcfg({"name": "deliverable", "MODEL_TYPE": "meanpool", "POOL_PARAM": None},
                     None, ckpt_dir=cfg.CKPT_DIR)
    df, feats = load_data(vcfg)
    folds = make_folds(df, vcfg)
    preds = pd.read_csv(cfg.PREDICTIONS_CSV, encoding="utf-8-sig")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    oof_cal = preds["predicted_fraction"].values.copy()
    affines = []
    for fold_idx, test_idx in enumerate(folds):
        train_ids, _ = fold_split(df, vcfg, test_idx, len(df))
        # 该折 3 种子模型在训练集上的集成预测
        train_pred = []
        for i in train_ids:
            sid = df["slide_id"].iloc[i]
            ps = []
            for s in vcfg.SEEDS:
                m = MeanPoolMIL(vcfg.IN_DIM, vcfg.HIDDEN, vcfg.DROPOUT)
                m.load_state_dict(torch.load(cfg.CKPT_DIR / f"fold{fold_idx}_seed{s}.pt",
                                             map_location=device, weights_only=True))
                m.eval().to(device)
                with torch.no_grad():
                    ps.append(m(feats[sid].to(device))[0].item())
            train_pred.append(np.mean(ps))
        train_pred = np.array(train_pred)
        y_train = df["weak_fraction"].iloc[train_ids].values
        a, b = np.polyfit(train_pred, y_train, 1)   # weak ≈ a·pred + b
        affines.append({"fold": fold_idx, "slope": a, "intercept": b})
        oof_cal[test_idx] = np.clip(a * preds["predicted_fraction"].values[test_idx] + b, 0, 1)
        log(f"  fold {fold_idx}: 仿射 weak≈{a:.3f}·pred+{b:.3f}")

    y = preds["weak_fraction"].values
    raw = preds["predicted_fraction"].values
    rows = []
    for tag, p in [("before", raw), ("after", oof_cal)]:
        rows.append({"stage": tag,
                     "pearson": pearsonr(y, p)[0], "spearman": spearmanr(y, p)[0],
                     "mae": np.abs(p - y).mean(), "rmse": float(np.sqrt(((p - y) ** 2).mean()))})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.OUTPUTS_DIR / "logs" / "calibration.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(affines).to_csv(cfg.OUTPUTS_DIR / "logs" / "calibration_affines.csv",
                                 index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, raw, s=45, alpha=0.6, edgecolors="k", linewidths=0.5, label="before calibration")
    ax.scatter(y, oof_cal, s=45, alpha=0.6, edgecolors="k", linewidths=0.5,
               marker="s", label="after calibration")
    ax.plot([0, 1.05], [0, 1.05], "r--", lw=1)
    ax.set_xlabel("weak_fraction"); ax.set_ylabel("predicted_fraction (OOF)")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05); ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.OUTPUTS_DIR / "figures" / "regression_plot_calibrated.png", dpi=150)
    plt.close(fig)
    log(f"[M4-C] 校准: MAE {rows[0]['mae']:.4f}→{rows[1]['mae']:.4f}，"
        f"RMSE {rows[0]['rmse']:.4f}→{rows[1]['rmse']:.4f}")


# ---------- 3. 范式对照：attention 序 vs 分数序 SRG ----------
def _flip_curve_abmil(models, x, order, n_steps, device):
    n = x.shape[0]
    buckets = np.array_split(order, n_steps)
    kept = np.arange(n)
    curve = []
    for b in [None] + list(buckets):
        if b is not None:
            kept = np.setdiff1d(kept, b)
        if len(kept) == 0:
            curve.append(0.0)
            continue
        with torch.no_grad():
            curve.append(float(np.mean([m(x[kept].to(device))[0].item() for m in models])))
    return np.array(curve)


def run_paradigm():
    """ABMIL 模型上：attention 权重序 vs instance 分数序的 patch flipping SRG。"""
    vcfg = make_vcfg({"name": "abmil", "MODEL_TYPE": "abmil_inst", "POOL_PARAM": None}, None)
    df, feats = load_data(vcfg)
    folds = make_folds(df, vcfg)
    fold_of = {i: f for f, t in enumerate(folds) for i in t}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = cfg.OUTPUTS_DIR / "logs" / "checkpoints_abmil"

    cfg.ATTN_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sid in enumerate(df["slide_id"]):
        models = []
        for s in vcfg.SEEDS:
            m = ABMILInstanceMIL(vcfg.IN_DIM, vcfg.HIDDEN, vcfg.ATTN_HIDDEN, vcfg.DROPOUT)
            m.load_state_dict(torch.load(ckpt / f"fold{fold_of[i]}_seed{s}.pt",
                                         map_location=device, weights_only=True))
            m.eval().to(device)
            models.append(m)
        x = feats[sid]
        with torch.no_grad():
            _, sc, at = zip(*[m.forward_with_attn(x.to(device)) for m in models])
        scores = torch.stack(sc).mean(0).cpu().numpy()
        attn = torch.stack(at).mean(0).cpu().numpy()
        np.save(cfg.ATTN_DIR / f"{sid}.npy", attn.astype(np.float32))

        c_sd = _flip_curve_abmil(models, x, np.argsort(-scores), cfg.SRG_STEPS, device)
        c_sa = _flip_curve_abmil(models, x, np.argsort(scores), cfg.SRG_STEPS, device)
        c_ad = _flip_curve_abmil(models, x, np.argsort(-attn), cfg.SRG_STEPS, device)
        c_aa = _flip_curve_abmil(models, x, np.argsort(attn), cfg.SRG_STEPS, device)
        rows.append({"slide_id": sid,
                     "srg_score": float((c_sa - c_sd).mean()),
                     "srg_attention": float((c_aa - c_ad).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.OUTPUTS_DIR / "logs" / "srg_paradigm.csv", index=False, encoding="utf-8-sig")
    win = (out.srg_score > out.srg_attention).mean()
    log(f"[M4-D] SRG：分数序 {out.srg_score.mean():.4f} vs attention 序 "
        f"{out.srg_attention.mean():.4f}；分数序胜出比例 {win:.0%}")


def main():
    t0 = time.time()
    cred_df = pd.read_csv(cfg.OUTPUTS_DIR / "logs" / "label_credibility.csv")
    cred = cred_df[cred_df.label_field == "percent_tumor_cells"]
    credibility = dict(zip(cred.slide_id, cred.credibility))

    run_aggregators(credibility)
    run_calibration()
    run_paradigm()
    print(f"[M4] 总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
