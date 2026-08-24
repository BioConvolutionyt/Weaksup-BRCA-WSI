# -*- coding: utf-8 -*-
"""M4 案例分析：按 |误差| 取最小/最大各 5 例，生成三联图（缩略图+热图+mask overlay）与归因表。

产出：
  outputs/figures/cases/{case_type}_{rank}_{slide_id}.png —— 10 张三联图
  outputs/logs/case_analysis.csv —— 案例表（morphology_reading/attribution 列由人工判读后补录）
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs import spatial_m3 as cfg

CASES_DIR = cfg.OUTPUTS_DIR / "figures" / "cases"
N_PER_TYPE = 5
THUMB_MAX_SIDE = 1200


def make_triptych(sid, case_type, rank):
    slide = openslide.open_slide(str(cfg.WSI_DIR / f"{sid}.svs"))
    w0, h0 = slide.dimensions
    scale = min(1.0, THUMB_MAX_SIDE / max(w0, h0))
    thumb = np.asarray(slide.get_thumbnail((int(w0 * scale), int(h0 * scale))).convert("RGB"))
    slide.close()
    heat = np.asarray(Image.open(cfg.OUTPUTS_DIR / "heatmaps" / f"{sid}_heatmap.png"))
    mask = np.asarray(Image.open(cfg.OUTPUTS_DIR / "masks" / f"{sid}_overlay.png"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, [thumb, heat, mask],
                              ["thumbnail", "tumor score heatmap", "binary mask overlay"]):
        ax.imshow(img); ax.set_title(title, fontsize=10); ax.axis("off")
    fig.suptitle(f"{case_type} #{rank}: {sid}", fontsize=11)
    fig.tight_layout()
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CASES_DIR / f"{case_type}_{rank}_{sid}.png", dpi=110)
    plt.close(fig)


def main():
    preds = pd.read_csv(cfg.PREDICTIONS_CSV, encoding="utf-8-sig")
    audit = pd.read_csv(PROJECT_ROOT / "data" / "qc_label_audit.csv", encoding="utf-8-sig")
    reg = pd.read_csv(PROJECT_ROOT / "data" / "preprocess_registry.csv", encoding="utf-8-sig")
    df = preds.merge(audit[["slide_id", "qc_risk_flags"]], on="slide_id", how="left") \
              .merge(reg[["slide_id", "n_patches"]], on="slide_id", how="left")
    df["qc_risk_flags"] = df["qc_risk_flags"].fillna("")

    rows = []
    for case_type, sub in [("success", df.nsmallest(N_PER_TYPE, "absolute_error")),
                           ("failure", df.nlargest(N_PER_TYPE, "absolute_error"))]:
        for rank, (_, r) in enumerate(sub.iterrows(), 1):
            sid = r["slide_id"]
            try:
                make_triptych(sid, case_type, rank)
            except Exception as e:
                print(f"  {sid} 三联图失败: {e}")
            rows.append({"case_type": case_type, "rank": rank, "slide_id": sid,
                         "weak_fraction": r["weak_fraction"],
                         "predicted_fraction": round(r["predicted_fraction"], 4),
                         "absolute_error": round(r["absolute_error"], 4),
                         "n_patches": int(r["n_patches"]),
                         "qc_risk_flags": r["qc_risk_flags"],
                         "morphology_reading": "", "attribution": ""})
            print(f"  {case_type} #{rank}: {sid} (err={r['absolute_error']:.3f})")

    out = pd.DataFrame(rows)
    out.to_csv(cfg.OUTPUTS_DIR / "logs" / "case_analysis.csv", index=False, encoding="utf-8-sig")
    print(f"[M4-案例] 10 张三联图 → {CASES_DIR}；表 → outputs/logs/case_analysis.csv")


if __name__ == "__main__":
    main()
