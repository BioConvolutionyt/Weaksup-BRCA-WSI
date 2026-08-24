# -*- coding: utf-8 -*-
"""6.1 标签可信度协同推断的可行性边界（文档 6.1）。

设计：真值可控的合成 bag——真实 UNI 特征 + 固定随机打分头 w* 产生真值 patch 分数，
真值袋占比 = q75 聚合（与交付形态一致）；按污染率 {0,20,35,50,65%} 把该比例袋标签
替换为 U[0,1]（离群污染，与 6.3 同模型）。

对照：无处理 vs 协同推断（PANDA 软清洗迭代版：round0 不加权 → OOF 残差 → 可信度 →
迭代 2 轮加权重训）。
评估（严谨性要点）：模型选择只用污染标签（真实协议），指标对**干净真值**计算；
可信度质量 = 残差识别污染标签的 AUC；收敛性 = 轮间可信度变化量。

矩阵：5 污染档 ×（1 无处理 + 3 轮协同）= 20 次 CV（300 次训练）。

产出：exploration/results/6.1_results.csv + 三张图。
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
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs import baseline_m2 as cfg
from src.training import run_cv

RESULTS_DIR = PROJECT_ROOT / "exploration" / "results"
OUT_CSV = RESULTS_DIR / "6.1_results.csv"

# 统一调色板（ColorBrewer 系），按图内实例顺序取色
PALETTE = ["#1f78b4", "#33a02c", "#e31a1c", "#fe8307", "#6a3d9a",
           "#fb9a99", "#91D1C2"]

LEVELS = [0.0, 0.2, 0.35, 0.5, 0.65]
N_ROUNDS = 3          # 协同推断轮数（round0 不加权 + 2 轮迭代）
SYNTH_SEED = 42
CONTAM_SEED = 7


def build_synthetic_truth(labels_df, feats, seed=SYNTH_SEED, target_std=0.12):
    """固定随机打分头 w* + 逐片 prevalency 偏移 → 真值 patch 分数 → q75 聚合。

    两处校准（严谨性）：
    1. w* 尺度校准到逐片分数 std≈0.15（patch 级判别信号）；
    2. 逐片偏移 δ_s ~ U[−d, d]，d 经二分校准使真值袋占比 std≈target_std
       （制造跨切片肿瘤负荷差异——否则真值趋同、任务退化）。
    """
    rng = np.random.RandomState(seed)
    w_star = rng.randn(1024)
    x0 = feats[labels_df["slide_id"].iloc[0]]
    raw0 = x0.numpy() @ w_star
    scale = 0.15 / max(raw0.std(), 1e-12)

    sids = labels_df["slide_id"].tolist()
    deltas = rng.uniform(-1, 1, len(sids))   # 单位偏移，幅度由 d 控制

    def truth_with_d(d):
        out = {}
        for sid, dv in zip(sids, deltas):
            z = feats[sid].numpy() @ w_star * scale + dv * d   # 偏移加在 sigmoid 输入单位
            s = 1.0 / (1.0 + np.exp(-z))
            k = max(1, int(round(len(s) * 0.75)))
            out[sid] = float(np.sort(s)[-k:].mean())
        return out

    # 二分校准 d 使真值 std ≈ target_std
    lo, hi = 0.0, 10.0
    for _ in range(25):
        mid = (lo + hi) / 2
        std_mid = np.std(list(truth_with_d(mid).values()))
        if abs(std_mid - target_std) < 0.005:
            break
        if std_mid < target_std:
            lo = mid
        else:
            hi = mid
    truth = truth_with_d(mid)
    print(f"[6.1] 真值校准: d={mid:.2f}，std={np.std(list(truth.values())):.3f}")
    return truth


def contaminate(truth, labels_df, rate, seed=CONTAM_SEED):
    """按污染率替换袋标签为 U[0,1]；返回 (污染标签表, 污染标记 dict)。"""
    rng = np.random.RandomState(seed)
    sids = labels_df["slide_id"].tolist()
    n_bad = int(round(len(sids) * rate))
    bad = set(rng.choice(sids, size=n_bad, replace=False)) if n_bad else set()
    flags = {s: int(s in bad) for s in sids}
    df = labels_df[["slide_id"]].copy()
    df["weak_fraction"] = [float(rng.uniform(0, 1)) if flags[s] else truth[s]
                           for s in sids]
    return df, flags


def make_cfg(labels_df, credibility=None):
    vcfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    vcfg.LABELS_DF = labels_df
    vcfg.CREDIBILITY = credibility
    vcfg.MODEL_TYPE = "quantile"
    vcfg.POOL_PARAM = 0.75
    vcfg.LAMBDA_SMOOTH = 0.0
    vcfg.USE_IND = False
    vcfg.USE_PROP = False
    vcfg.AMPLIFY_N = None
    vcfg.NOISE_MODEL = None
    vcfg.NOISE_LEVEL = 0.0
    vcfg.CKPT_DIR = None
    return vcfg


def eval_vs_truth(preds_df, truth):
    m = preds_df.merge(pd.DataFrame({"slide_id": list(truth.keys()),
                                     "truth": list(truth.values())}), on="slide_id")
    err = np.abs(m["predicted_fraction"] - m["truth"])
    return float(err.mean()), float(pd.Series(m["predicted_fraction"]).corr(m["truth"]))


def credibility_from_residuals(resid):
    r = np.asarray(list(resid.values()), dtype=float)
    tau = np.median(r) + 1e-12
    c = np.exp(-r / tau)
    c = c / c.mean()
    return dict(zip(resid.keys(), c))


def run_level(labels_df, truth, flags, level):
    """单污染档：无处理 + 协同推断 3 轮。"""
    rows = []
    # 无处理
    preds, _, _ = run_cv(make_cfg(labels_df))
    mae_t, pear_t = eval_vs_truth(preds, truth)
    rows.append({"level": level, "method": "none", "round": 0,
                 "mae_vs_truth": mae_t, "pearson_vs_truth": pear_t,
                 "cred_auc": np.nan, "delta_c": np.nan})
    print(f"  [6.1] level={level} 无处理: MAE_vs_truth={mae_t:.4f}")

    # 协同推断（迭代）
    cred = None
    prev_c = None
    for r in range(N_ROUNDS):
        preds, _, _ = run_cv(make_cfg(labels_df, credibility=cred))
        mae_t, pear_t = eval_vs_truth(preds, truth)
        resid = {sid: abs(float(preds.loc[preds.slide_id == sid, "predicted_fraction"].iloc[0])
                        - float(labels_df.loc[labels_df.slide_id == sid, "weak_fraction"].iloc[0]))
                 for sid in labels_df["slide_id"]}
        auc = roc_auc_score(list(flags.values()), list(resid.values()))
        cred = credibility_from_residuals(resid)
        delta_c = (np.mean([abs(cred[s] - prev_c[s]) for s in cred]) if prev_c else np.nan)
        rows.append({"level": level, "method": "coinfer", "round": r,
                     "mae_vs_truth": mae_t, "pearson_vs_truth": pear_t,
                     "cred_auc": float(auc), "delta_c": delta_c})
        print(f"  [6.1] level={level} 协同 round={r}: MAE={mae_t:.4f} AUC={auc:.3f}")
        prev_c = cred
    return rows


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(PROJECT_ROOT / "data" / "area_reference.csv", encoding="utf-8-sig")
    old25 = sorted(pd.read_csv(PROJECT_ROOT / "TCGA_Download" / "slide_rename_mapping.csv",
                               encoding="utf-8-sig")["slide_id"])
    labels_df = ref[ref.slide_id.isin(old25)][["slide_id"]].copy().reset_index(drop=True)

    import torch
    feats = {sid: torch.load(cfg.FEATURES_DIR / f"{sid}.pt", map_location="cpu",
                             weights_only=True) for sid in labels_df["slide_id"]}
    truth = build_synthetic_truth(labels_df, feats)
    print(f"[6.1] 合成真值就绪：n={len(truth)}，真值分布 "
          f"min={min(truth.values()):.3f} max={max(truth.values()):.3f}")

    all_rows = []
    for level in LEVELS:
        print(f"[6.1] 污染率 {level:.0%}")
        df_c, flags = contaminate(truth, labels_df, level)
        all_rows.extend(run_level(df_c, truth, flags, level))

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[6.1] 完成 {len(out)} 行 → {OUT_CSV}；耗时 {(time.time()-t0)/60:.1f} min")
    make_figures(out)


def make_figures(out):
    # 1. 污染率-误差曲线（无处理 vs 协同最终轮）
    fig, ax = plt.subplots(figsize=(7, 5))
    none_g = out[out.method == "none"].sort_values("level")
    coi = out[out.method == "coinfer"]
    coi_final = coi.groupby("level").apply(
        lambda g: g[g["round"] == g["round"].max()], include_groups=False).reset_index()
    ax.plot(none_g["level"], none_g["mae_vs_truth"], "o-", color=PALETTE[0],
            label="no treatment")
    ax.plot(coi_final["level"], coi_final["mae_vs_truth"], "s-", color=PALETTE[1],
            label="co-inference (final round)")
    ax.set_xlabel("contamination rate"); ax.set_ylabel("MAE vs ground truth")
    ax.set_title("6.1: co-inference feasibility boundary"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "6.1_contamination_curve.png", dpi=150)
    plt.close(fig)

    # 2. 可信度 AUC（协同推断各轮）
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, (r, g) in enumerate(sorted(coi.groupby("round"), key=lambda t: t[0])):
        g = g.sort_values("level")
        ax.plot(g["level"], g["cred_auc"], "o-", color=PALETTE[k],
                label=f"round {r}")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("contamination rate"); ax.set_ylabel("AUC (residual flags contamination)")
    ax.set_title("6.1: credibility identifies contaminated labels"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "6.1_credibility_auc.png", dpi=150)
    plt.close(fig)

    # 3. 收敛性（轮间可信度变化）
    fig, ax = plt.subplots(figsize=(7, 5))
    conv = coi.dropna(subset=["delta_c"])
    for k, (lv, g) in enumerate(sorted(conv.groupby("level"), key=lambda t: t[0])):
        g = g.sort_values("round")
        ax.plot(g["round"], g["delta_c"], "o-", color=PALETTE[k],
                label=f"level {lv:.0%}")
    ax.set_xlabel("round"); ax.set_ylabel("mean |Δc| between rounds")
    ax.set_title("6.1: co-inference convergence"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "6.1_convergence.png", dpi=150)
    plt.close(fig)
    print("[6.1] 三张图已存")


if __name__ == "__main__":
    main()
