# -*- coding: utf-8 -*-
"""
M1 数据 QC 三件套（溯源：Modeling Plan.md §2.1）：
  1) 特征空间 UMAP（UBC-OCEAN 法）：每切片 UNI 均值/ std 特征降维散点，按 weak_fraction 着色，
     并用 1024 维空间中到质心的 L2 距离 z-score 标记离群切片（染色/倍率/质量异常）。
  2) imagehash 近重复检查（PANDA 法）：OpenSlide 低倍缩略图 phash 两两汉明距离，
     排查同 case 多 vial 或重复扫描造成的隐性泄漏。
  3) 弱标签对齐审核：registry × weak_labels.csv 合并；标签风险标记
     （多来源 range 大、cells/nuclei 口径倒挂、高坏死、高淋浸）。

输出：
  outputs/figures/qc_feature_umap.png
  data/qc_imagehash.csv
  data/qc_label_audit.csv
"""

from pathlib import Path

import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch

# ---------------- 配置（全部在代码内，不用命令行参数） ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"
WSI_DIR = PROJECT_ROOT / "TCGA_Download"
WEAK_LABELS = PROJECT_ROOT / "data" / "weak_labels.csv"
REGISTRY = PROJECT_ROOT / "data" / "preprocess_registry.csv"
FIG_OUT = PROJECT_ROOT / "outputs" / "figures" / "qc_feature_umap.png"
HASH_CSV = PROJECT_ROOT / "data" / "qc_imagehash.csv"
AUDIT_CSV = PROJECT_ROOT / "data" / "qc_label_audit.csv"

PHASH_FLAG_THRESHOLD = 10     # 汉明距离 ≤10 视为疑似近重复（PANDA 惯例阈值附近）
OUTLIER_Z = 2.5               # 质心距离 z-score 阈值
THUMB_MAX_SIDE = 1024         # 缩略图最长边


def load_weak_labels():
    return pd.read_csv(WEAK_LABELS, encoding="utf-8-sig")


def feature_umap(labels):
    """每切片均值/std 特征 → UMAP 散点 + 质心距离离群检测。"""
    slide_ids, means, stds = [], [], []
    for pt in sorted(FEATURES_DIR.glob("*.pt")):
        feats = torch.load(pt, map_location="cpu", weights_only=True)
        slide_ids.append(pt.stem)
        means.append(feats.mean(dim=0).numpy())
        stds.append(feats.std(dim=0).numpy())
    if not slide_ids:
        print("[UMAP] features/ 为空，跳过（预处理未完成）")
        return None
    X = np.stack(means)

    # 1024 维空间的离群检测（不依赖 UMAP 也可解释）
    centroid = X.mean(axis=0)
    dists = np.linalg.norm(X - centroid, axis=1)
    z = (dists - dists.mean()) / (dists.std() + 1e-12)
    outliers = [s for s, zz in zip(slide_ids, z) if zz > OUTLIER_Z]

    merged = pd.DataFrame({"slide_id": slide_ids, "centroid_dist": dists, "dist_z": z}).merge(
        labels[["slide_id", "weak_fraction"]], on="slide_id", how="left"
    )

    # UMAP 降维（失败则 PCA 兜底）
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=min(10, len(X) - 1), min_dist=0.1, random_state=42)
        emb = reducer.fit_transform(X)
        method = "UMAP"
    except Exception as e:
        from sklearn.decomposition import PCA
        print(f"[UMAP] 失败（{e}），回退 PCA")
        emb = PCA(n_components=2, random_state=42).fit_transform(X)
        method = "PCA"

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=merged["weak_fraction"], cmap="viridis", s=60, edgecolors="k", linewidths=0.5)
    for (x, y), s in zip(emb, merged["slide_id"]):
        short = s.replace("TCGA-", "").replace("-01Z-00-DX1", "")
        ax.annotate(short, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_title(f"Per-slide UNI mean-feature {method} (color = weak_fraction)")
    ax.set_xlabel(f"{method} 1"); ax.set_ylabel(f"{method} 2")
    fig.colorbar(sc, label="weak_fraction")
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    plt.close(fig)

    print(f"[UMAP] 方法={method}，图已存 {FIG_OUT}")
    print(f"[UMAP] 离群切片（z>{OUTLIER_Z}）: {outliers if outliers else '无'}")
    return merged[["slide_id", "centroid_dist", "dist_z"]]


def imagehash_check(labels):
    """缩略图 phash 两两距离 + 同 case 标记。"""
    rows = []
    hashes = {}
    for _, r in labels.iterrows():
        sid = r["slide_id"]
        # 优先用 weak_labels.csv 的 wsi_path（补充数据为全路径），退回首批目录惯例
        path = Path(r["wsi_path"]) if "wsi_path" in r and pd.notna(r.get("wsi_path")) else WSI_DIR / f"{sid}.svs"
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            continue
        slide = openslide.open_slide(str(path))
        w, h = slide.dimensions
        scale = min(1.0, THUMB_MAX_SIDE / max(w, h))
        thumb = slide.get_thumbnail((int(w * scale), int(h * scale))).convert("RGB")
        hashes[sid] = imagehash.phash(thumb)
        slide.close()

    sids = list(hashes.keys())
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            a, b = sids[i], sids[j]
            d = hashes[a] - hashes[b]
            same_case = a.rsplit("-", 3)[0] == b.rsplit("-", 3)[0]  # 前三段=participant
            rows.append({
                "slide_a": a, "slide_b": b, "phash_hamming": int(d),
                "same_case": same_case, "flag_near_duplicate": bool(d <= PHASH_FLAG_THRESHOLD),
            })
    df = pd.DataFrame(rows).sort_values("phash_hamming")
    df.to_csv(HASH_CSV, index=False, encoding="utf-8-sig")
    flagged = df[df["flag_near_duplicate"]]
    print(f"[HASH] 切片对数={len(df)}，最小距离={df['phash_hamming'].min() if len(df) else 'N/A'}，"
          f"疑似近重复对={len(flagged)}")
    if len(flagged):
        print(flagged.to_string(index=False))
    return df


def label_audit(labels, umap_df):
    """registry × weak_labels × QC 风险标记。"""
    reg = pd.read_csv(REGISTRY, encoding="utf-8-sig") if REGISTRY.exists() else pd.DataFrame()
    df = labels.copy()
    if len(reg):
        df = df.merge(reg[["slide_id", "n_patches", "est_tissue_area_mm2", "level0_magnification"]],
                      on="slide_id", how="left")
    if umap_df is not None:
        df = df.merge(umap_df, on="slide_id", how="left")

    risks = []
    for _, r in df.iterrows():
        notes = []
        if not (0.0 <= r["weak_fraction"] <= 1.0):
            notes.append("fraction越界")
        if r.get("n_source_slides", 1) > 1:
            notes.append(f"多来源切片({int(r['n_source_slides'])})")
        ptc, ptn = r.get("percent_tumor_cells"), r.get("percent_tumor_nuclei")
        if pd.notna(ptc) and pd.notna(ptn) and ptn > ptc:
            notes.append(f"cells({ptc})<nuclei({ptn})口径倒挂")
        if pd.notna(r.get("percent_necrosis")) and r["percent_necrosis"] >= 15:
            notes.append(f"高坏死({r['percent_necrosis']}%)")
        if pd.notna(r.get("percent_lymphocyte_infiltration")) and r["percent_lymphocyte_infiltration"] >= 30:
            notes.append(f"高淋浸({r['percent_lymphocyte_infiltration']}%)")
        if pd.notna(r.get("dist_z")) and r["dist_z"] > OUTLIER_Z:
            notes.append(f"特征离群(z={r['dist_z']:.1f})")
        risks.append("; ".join(notes) if notes else "")

    df["qc_risk_flags"] = risks
    df.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    flagged = df[df["qc_risk_flags"] != ""]
    print(f"[AUDIT] 风险标记切片 {len(flagged)}/{len(df)}：")
    for _, r in flagged.iterrows():
        print(f"  - {r['slide_id']}: {r['qc_risk_flags']}")
    return df


def main():
    labels = load_weak_labels()
    print(f"弱标签: {len(labels)} 张，weak_fraction 分布 "
          f"min={labels['weak_fraction'].min():.2f} median={labels['weak_fraction'].median():.2f} "
          f"max={labels['weak_fraction'].max():.2f}")
    umap_df = feature_umap(labels)
    imagehash_check(labels)
    label_audit(labels, umap_df)
    print("[QC] 完成")


if __name__ == "__main__":
    main()
