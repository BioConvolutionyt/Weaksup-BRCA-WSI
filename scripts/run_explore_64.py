# -*- coding: utf-8 -*-
"""6.4 面积硬约束 × 空间平滑的"张力"（文档 6.4）。

形式化：x∈{0,1}^N，max Σ s_i·x_i − λ·Σ w_ij|x_i−x_j|，s.t. Σx=K（K=预测占比×N）。
无基数约束 → 子模二元能量，s-t 最小割精确可解；加基数约束 → NP-hard（平衡割类）。
本脚本对比四种近似策略（纯后处理，不重训）：
  S1 简单阈值 0.5 + 形态学（现状基线，无面积保证）
  S2 Top-K 贪心（面积构造上精确）+ 形态学
  S3 图传播平滑分数 + Top-K + 形态学（平滑先验融入分数再精确预算）
  S4 拉格朗日图割（PyMaxflow 可用时；Potts + 一元偏差 γ 二分逼近 K；代表切片子集）

分两阶段：PHASE="A" 用交付配置（面积标签+q75+可信度+平滑 λ=0.01，不开传播——
传播留给 S3 在后处理施加）跑 2 次 CV 生成 OOF patch 分数并缓存；
PHASE="B" 纯 CPU 后处理对比。阶段 A 同时导出交付管线逐切片可信度
（文档 6.5 要求的"推断出的标签可信度"中间变量，归 6.1 产物）。

产出：exploration/results/6.4_scores.npz、6.4_preds.csv、6.4_metrics.csv、
      6.4_mask_comparison.png、6.1_credibility_real75.csv
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs import spatial_m3 as cfg
from src.training import run_cv
from src.spatial import (build_propagation_graph, count_components, graph_propagate,
                         mean_neighbor_diff, morphology_cleanup, scores_to_grid)

RESULTS_DIR = PROJECT_ROOT / "exploration" / "results"
PALETTE = ["#1f78b4", "#33a02c", "#e31a1c", "#fe8307", "#6a3d9a",
           "#fb9a99", "#91D1C2"]
SCORES_NPZ = RESULTS_DIR / "6.4_scores.npz"
PREDS_CSV = RESULTS_DIR / "6.4_preds.csv"
METRICS_CSV = RESULTS_DIR / "6.4_metrics.csv"
CRED_CSV = RESULTS_DIR / "6.1_credibility_real75.csv"

PHASE = "D"   # "A"=GPU 分数生成；"B"=CPU 后处理对比；"C"=S4 β 扫描；"D"=仅重绘图
S4_SUBSET_N = 9          # 拉格朗日图割的代表切片数（按预测占比分层抽样）
MASK_THRESHOLD = 0.5


def make_cfg(labels_df, credibility, lam):
    vcfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    vcfg.LABELS_DF = labels_df
    vcfg.CREDIBILITY = credibility
    vcfg.LAMBDA_SMOOTH = lam
    vcfg.USE_IND = False
    vcfg.USE_PROP = False
    vcfg.AMPLIFY_N = None
    vcfg.NOISE_MODEL = None
    vcfg.NOISE_LEVEL = 0.0
    vcfg.MODEL_TYPE = "quantile"
    vcfg.POOL_PARAM = 0.75
    vcfg.CKPT_DIR = None
    return vcfg


def phase_a():
    """交付配置分数生成：round0 求可信度（并导出）→ round1 可信度+平滑。"""
    ref = pd.read_csv(PROJECT_ROOT / "data" / "area_reference.csv", encoding="utf-8-sig")
    labels_df = ref[["slide_id", "area_frac_soft"]].rename(
        columns={"area_frac_soft": "weak_fraction"})

    print("[6.4-A] round0：无加权求可信度")
    preds0, _, _ = run_cv(make_cfg(labels_df, None, 0.0))
    r = preds0["absolute_error"].values.astype(float)
    c = np.exp(-r / (np.median(r) + 1e-12))
    cred = dict(zip(preds0["slide_id"], c / c.mean()))
    pd.DataFrame({"slide_id": list(cred.keys()), "credibility": list(cred.values())}
                 ).to_csv(CRED_CSV, index=False, encoding="utf-8-sig")
    print(f"[6.4-A] 逐切片可信度已导出 → {CRED_CSV}（兼作 6.1 真实数据产物）")

    print("[6.4-A] round1：可信度 + 平滑 λ=0.01（交付配置，不开传播）")
    preds, oof_scores, metrics = run_cv(make_cfg(labels_df, cred, 0.01))
    print(f"[6.4-A] 交付配置指标：{metrics}")
    preds.to_csv(PREDS_CSV, index=False, encoding="utf-8-sig")
    np.savez_compressed(SCORES_NPZ, **{sid: sc for sid, sc in oof_scores.items()})
    print(f"[6.4-A] 分数缓存 → {SCORES_NPZ}")


def topk_mask(scores, k):
    """按分数取 top-K 的 patch 级二值 mask（K 截断到 [1, N]）。"""
    n = len(scores)
    k = int(np.clip(round(k), 1, n))
    idx = np.argpartition(-scores, k - 1)[:k]
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def lagrangian_graphcut(scores, coords, k_target, beta=0.3, iters=22):
    """S4：Potts 网格图 + 一元偏差 γ 二分逼近基数 K（PyMaxflow）。

    能量 E(x) = Σ D_i(x_i) + β·Σ_{(i,j)∈E8} [x_i≠x_j]，
    D_i(1) = −log s_i + γ，D_i(0) = −log(1−s_i)；γ 每加 1 等价于选区代价加 1，
    二分 γ 使 |{x=1}| ≈ K。每次求解为一次精确 s-t 最小割（无基数时的精确解），
    基数约束通过对偶偏差近似满足——展示"全局约束 × 局部平滑"的张力。
    """
    import maxflow
    gx = (coords["x"].values // 512).astype(np.int64)
    gy = (coords["y"].values // 512).astype(np.int64)
    H, W = gy.max() + 1, gx.max() + 1
    node_id = {(y, x): i for i, (y, x) in enumerate(zip(gy, gx))}
    s = np.clip(scores, 1e-6, 1 - 1e-6)
    logit1, logit0 = -np.log(s), -np.log(1 - s)

    d1_base = logit1.astype(np.float64)
    d0 = logit0.astype(np.float64)

    def solve(gamma):
        g = maxflow.Graph[float](len(s), 4 * len(s))
        g.add_nodes(len(s))
        for (y, x), i in node_id.items():
            for dy, dx in ((1, 0), (0, 1), (1, 1), (-1, 1)):
                j = node_id.get((y + dy, x + dx))
                if j is not None:
                    g.add_edge(i, j, beta, beta)
        d1 = d1_base + gamma
        for i in range(len(s)):   # 该版本 add_tedge 向量化不稳定，逐点标量添加
            g.add_tedge(i, float(d1[i]), float(d0[i]))
        g.maxflow()
        # get_segment==1 为 sink 侧：切割代价 = cap_source = D_i(1)，即 x_i=1（选中）
        return np.array([g.get_segment(i) == 1 for i in range(len(s))])

    lo, hi = -6.0, 6.0
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        m = solve(mid)
        cnt = int(m.sum())
        if best is None or abs(cnt - k_target) < abs(best.sum() - k_target):
            best = m
        if cnt > k_target:
            lo = mid
        else:
            hi = mid
    return best


def phase_b():
    """四策略后处理对比（纯 CPU）。"""
    t0 = time.time()
    preds = pd.read_csv(PREDS_CSV, encoding="utf-8-sig")
    npz = np.load(SCORES_NPZ)
    sids = [s for s in preds["slide_id"] if s in npz.files]
    pred_of = dict(zip(preds["slide_id"], preds["predicted_fraction"]))

    # S4 代表子集：按预测占比排序分层抽样
    sub = preds[preds.slide_id.isin(sids)].sort_values("predicted_fraction")
    s4_sids = set(sub.iloc[np.linspace(0, len(sub) - 1, S4_SUBSET_N).astype(int)]["slide_id"])
    try:
        import maxflow  # noqa: F401
        has_mf = True
    except ImportError:
        has_mf = False
        print("[6.4-B] PyMaxflow 不可用，S4 跳过（笔记中记为未来工作）")

    rows = []
    for sid in sids:
        scores = npz[sid].astype(np.float64)
        coords = pd.read_csv(cfg.COORDS_DIR / f"{sid}.csv")
        n = len(scores)
        k = pred_of[sid] * n
        grid = scores_to_grid(scores, coords)
        tissue = ~np.isnan(grid)

        # S3 的传播分数（全切片一次；图用 UNI 特征构建，参数同交付管线默认）
        feats_t = torch.load(cfg.FEATURES_DIR / f"{sid}.pt", map_location="cpu",
                             weights_only=True)
        ei, ew = build_propagation_graph(feats_t, coords, radius=2, topk=8)
        sc_prop = graph_propagate(torch.from_numpy(scores), ei, ew,
                                  alpha=0.5).numpy()

        # S3b：传播 3 轮（更平滑的场 → top-K 更成团）
        sc_p3 = torch.from_numpy(scores)
        for _ in range(3):
            sc_p3 = graph_propagate(sc_p3, ei, ew, alpha=0.5)
        sc_prop3 = sc_p3.numpy()

        variants = {"S1_threshold": scores >= MASK_THRESHOLD,
                    "S2_topk": topk_mask(scores, k),
                    "S3_prop_topk": topk_mask(sc_prop, k),
                    "S3b_prop3_topk": topk_mask(sc_prop3, k)}
        if has_mf and sid in s4_sids:
            t1 = time.time()
            variants["S4_graphcut"] = lagrangian_graphcut(scores, coords, k)
            s4_time = time.time() - t1
        else:
            s4_time = np.nan

        for name, m in variants.items():
            t1 = time.time()
            mg = np.full(grid.shape, False)
            gx = (coords["x"].values // 512).astype(np.int64)
            gy = (coords["y"].values // 512).astype(np.int64)
            mg[gy, gx] = m
            mg_clean = morphology_cleanup(mg, tissue_domain=tissue)
            mask_frac = float(mg_clean.sum()) / float(tissue.sum())
            rows.append({"slide_id": sid, "strategy": name,
                         "area_err": abs(mask_frac - pred_of[sid]),
                         "mask_frac": mask_frac, "pred_frac": pred_of[sid],
                         "n_components": count_components(mg_clean),
                         "mask_neighbor_diff": mean_neighbor_diff(
                             np.where(tissue, mg_clean.astype(np.float32), np.nan)),
                         "runtime_s": s4_time if name == "S4_graphcut"
                         else round(time.time() - t1, 3)})
        print(f"[6.4-B] {sid} 完成（{len(rows)} 行累计）")

    out = pd.DataFrame(rows)
    out.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    summ = out.groupby("strategy")[["area_err", "n_components",
                                    "mask_neighbor_diff"]].mean().round(4)
    print("\n" + summ.to_string())
    print(f"[6.4-B] 完成，耗时 {(time.time()-t0)/60:.1f} min → {METRICS_CSV}")
    make_figure(out, npz, preds, sids)


def make_figure(out, npz, preds, sids):
    """3 张代表切片 × (分数热图 + S1/S2/S3/S3b/S4) 对比图，3×2 竖向布局。"""
    sub = preds[preds.slide_id.isin(sids)].sort_values("predicted_fraction")
    picks = [sub.iloc[0]["slide_id"], sub.iloc[len(sub)//2]["slide_id"],
             sub.iloc[-1]["slide_id"]]
    for sid in picks:
        scores = npz[sid].astype(np.float64)
        coords = pd.read_csv(cfg.COORDS_DIR / f"{sid}.csv")
        grid = scores_to_grid(scores, coords)
        tissue = ~np.isnan(grid)
        n = len(scores)
        k = float(preds.loc[preds.slide_id == sid, "predicted_fraction"].iloc[0]) * n
        feats_t = torch.load(cfg.FEATURES_DIR / f"{sid}.pt", map_location="cpu",
                             weights_only=True)
        ei, ew = build_propagation_graph(feats_t, coords, radius=2, topk=8)
        sc_p = torch.from_numpy(scores)
        sc_prop = graph_propagate(sc_p, ei, ew, alpha=0.5).numpy()
        for _ in range(2):
            sc_p = graph_propagate(sc_p, ei, ew, alpha=0.5)
        sc_prop3 = sc_p.numpy()
        masks = [("S1 threshold", scores >= MASK_THRESHOLD),
                 ("S2 top-K", topk_mask(scores, k)),
                 ("S3 prop x1 + top-K", topk_mask(sc_prop, k)),
                 ("S3b prop x3 + top-K", topk_mask(sc_prop3, k))]
        try:
            import maxflow  # noqa: F401
            masks.append(("S4 graph cut", lagrangian_graphcut(scores, coords, k)))
        except ImportError:
            pass
        key_of = {"S1 threshold": "S1_threshold", "S2 top-K": "S2_topk",
                  "S3 prop x1 + top-K": "S3_prop_topk",
                  "S3b prop x3 + top-K": "S3b_prop3_topk",
                  "S4 graph cut": "S4_graphcut"}
        panels = [("score map", None)] + masks
        # 按网格真实宽高比动态定画布高，消除子图间竖向留白
        gw, gh = grid.shape[1], grid.shape[0]
        cell_w = 4.4
        fig_w = cell_w * 2
        fig_h = cell_w * (gh / gw) * 3 + 1.4
        fig, axes = plt.subplots(3, 2, figsize=(fig_w, fig_h))
        for ax, (name, m) in zip(axes.flat, panels):
            if m is None:
                ax.imshow(np.where(tissue, grid, np.nan), cmap="jet",
                          vmin=0, vmax=1)
                ax.set_title(f"{sid}\nscore map", fontsize=10)
            else:
                mg = np.full(grid.shape, False)
                gx = (coords["x"].values // 512).astype(np.int64)
                gy = (coords["y"].values // 512).astype(np.int64)
                mg[gy, gx] = m
                mg = morphology_cleanup(mg, tissue_domain=tissue)
                ax.imshow(np.where(tissue, mg.astype(float), np.nan),
                          cmap="gray", vmin=0, vmax=1)
                row = out[(out.slide_id == sid)
                          & (out.strategy == key_of[name])]
                if len(row):
                    ax.set_title(f"{name}\nerr={row['area_err'].iloc[0]:.3f} "
                                 f"comp={int(row['n_components'].iloc[0])}",
                                 fontsize=10)
                else:
                    ax.set_title(name, fontsize=10)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"6.4_mask_{sid[:23]}.png", dpi=130)
        plt.close(fig)
    print("[6.4] 对比图已存（3x2 竖向布局）")


def redraw_pareto():
    """由 6.4_pareto.csv + 6.4_metrics.csv 重绘 Pareto 图（对数横轴 + 交错标注）。"""
    out = pd.read_csv(RESULTS_DIR / "6.4_pareto.csv", encoding="utf-8-sig")
    m = pd.read_csv(RESULTS_DIR / "6.4_metrics.csv", encoding="utf-8-sig")
    g = out.groupby("beta")[["area_err", "n_components"]].mean().reset_index()
    ref = m.groupby("strategy")[["area_err", "n_components"]].mean()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(g["area_err"], g["n_components"], "o-", color=PALETTE[0],
            label="S4 graph cut (beta sweep)", zorder=3)
    # β 标签在左上空白区纵向排列，指引线连点；按点横坐标排序避免引线交叉
    pts = g.sort_values("area_err")
    label_ys = np.linspace(42, 28, len(pts))
    for (_, r), ly in zip(pts.iterrows(), label_ys):
        ax.annotate(f"beta={r['beta']}", xy=(r["area_err"], r["n_components"]),
                    xytext=(0.00215, ly), fontsize=9, va="center",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6,
                                    shrinkB=4))
    for k, (name, marker, lbl) in enumerate(
            [("S1_threshold", "s", "S1 threshold"),
             ("S2_topk", "^", "S2 top-K"),
             ("S3b_prop3_topk", "D", "S3b prop x3 + top-K")]):
        ax.scatter(ref.loc[name, "area_err"], ref.loc[name, "n_components"],
                   marker=marker, s=90, color=PALETTE[k + 1], label=lbl,
                   zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(0.002, 0.3)
    ax.set_ylim(9, 45)
    ax.axvline(0.03, color="gray", ls="--", lw=1)
    ax.annotate("area error target 0.03", (0.0315, 10.2), fontsize=8,
                color="gray", va="center")          # 虚线底部，远离图例
    ax.set_xlabel("area error |mask frac - predicted frac| (log scale)")
    ax.set_ylabel("connected components")
    ax.set_title("6.4: area-smoothness tension")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "6.4_pareto.png", dpi=150)
    plt.close(fig)
    print("[6.4] Pareto 图已重绘")


def phase_c():
    """S4 的 β 扫描：面积误差-连通域 Pareto 前沿（张力的定量可视化，9 切片子集）。"""
    preds = pd.read_csv(PREDS_CSV, encoding="utf-8-sig")
    npz = np.load(SCORES_NPZ)
    sids = [s for s in preds["slide_id"] if s in npz.files]
    sub = preds[preds.slide_id.isin(sids)].sort_values("predicted_fraction")
    s4_sids = list(sub.iloc[np.linspace(0, len(sub) - 1, S4_SUBSET_N).astype(int)]
                   ["slide_id"])
    rows = []
    for beta in [0.1, 0.2, 0.3, 0.45, 0.6, 0.9]:
        for sid in s4_sids:
            scores = npz[sid].astype(np.float64)
            coords = pd.read_csv(cfg.COORDS_DIR / f"{sid}.csv")
            k = float(preds.loc[preds.slide_id == sid, "predicted_fraction"].iloc[0]) \
                * len(scores)
            t1 = time.time()
            m = lagrangian_graphcut(scores, coords, k, beta=beta)
            rt = time.time() - t1
            grid = scores_to_grid(scores, coords)
            tissue = ~np.isnan(grid)
            mg = np.full(grid.shape, False)
            gx = (coords["x"].values // 512).astype(np.int64)
            gy = (coords["y"].values // 512).astype(np.int64)
            mg[gy, gx] = m
            mg = morphology_cleanup(mg, tissue_domain=tissue)
            pred_f = float(preds.loc[preds.slide_id == sid,
                                     "predicted_fraction"].iloc[0])
            rows.append({"beta": beta, "slide_id": sid,
                         "area_err": abs(mg.sum() / tissue.sum() - pred_f),
                         "n_components": count_components(mg),
                         "runtime_s": rt})
        b = [r for r in rows if r["beta"] == beta]
        print(f"[6.4-C] β={beta}: area_err={np.mean([r['area_err'] for r in b]):.4f} "
              f"comp={np.mean([r['n_components'] for r in b]):.1f} "
              f"runtime={np.mean([r['runtime_s'] for r in b]):.1f}s")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "6.4_pareto.csv", index=False, encoding="utf-8-sig")
    redraw_pareto()


def phase_d():
    """仅重绘图：3 张代表切片掩膜对比图（3×2 布局）+ Pareto 图（由缓存 CSV）。"""
    preds = pd.read_csv(PREDS_CSV, encoding="utf-8-sig")
    npz = np.load(SCORES_NPZ)
    sids = [s for s in preds["slide_id"] if s in npz.files]
    m = pd.read_csv(METRICS_CSV, encoding="utf-8-sig")
    make_figure(m, npz, preds, sids)
    redraw_pareto()


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if PHASE == "A":
        phase_a()
    elif PHASE == "B":
        phase_b()
    elif PHASE == "C":
        phase_c()
    else:
        phase_d()
