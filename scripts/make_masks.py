# -*- coding: utf-8 -*-
"""M3 二值 mask 与 overlay：score≥0.5 → 形态学清理（去小连通域+填洞）→ mask/overlay 落盘。

输入：outputs/logs/scores/*.npy（交付变体 patch 分数）+ coords/*.csv。
输出：
  outputs/masks/{slide_id}_mask.png       —— 二值肿瘤 mask（网格放大到缩略图尺寸）
  outputs/masks/{slide_id}_overlay.png    —— mask 半透明叠加缩略图
  outputs/masks/mask_fractions.csv        —— mask 面积占比（mask patch 数 / 组织 patch 数）
"""
import sys
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs import spatial_m3 as cfg
from src.spatial import morphology_cleanup, scores_to_grid

THUMB_MAX_SIDE = 2000
MASK_DIR = cfg.OUTPUTS_DIR / "masks"


def render_one(slide_id, pred_fraction):
    scores = np.load(cfg.SCORES_DIR / f"{slide_id}.npy")
    coords = pd.read_csv(cfg.COORDS_DIR / f"{slide_id}.csv")
    grid = scores_to_grid(scores, coords)

    # 面积一致阈值：取分数的 (1 − predicted_fraction) 分位数，使 mask 面积 ≈ 模型预测占比
    # （修复 sigmoid 分数压缩导致固定 0.5 阈值失效的问题；阈值按模型自身预测自适应）
    thr = np.quantile(scores, np.clip(1.0 - pred_fraction, 0.0, 1.0))
    mask_raw = np.nan_to_num(grid, nan=0.0) >= thr
    mask_clean = morphology_cleanup(mask_raw, min_region=cfg.MIN_REGION_PATCHES,
                                    tissue_domain=~np.isnan(grid))

    slide = openslide.open_slide(str(cfg.WSI_DIR / f"{slide_id}.svs"))
    w0, h0 = slide.dimensions
    scale = min(1.0, THUMB_MAX_SIDE / max(w0, h0))
    thumb = np.asarray(slide.get_thumbnail((int(w0 * scale), int(h0 * scale))).convert("RGB"))
    slide.close()

    mask_img = Image.fromarray((mask_clean * 255).astype(np.uint8)).resize(
        (thumb.shape[1], thumb.shape[0]), Image.NEAREST)
    mask_img.save(MASK_DIR / f"{slide_id}_mask.png")

    # overlay：mask 区域半透明红色叠加
    m = np.asarray(mask_img) > 127
    overlay = thumb.astype(np.float32) / 255.0
    red = np.zeros_like(overlay); red[..., 0] = 1.0
    overlay = np.where(m[..., None], overlay * 0.55 + red * 0.45, overlay)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(MASK_DIR / f"{slide_id}_overlay.png")

    n_tissue = int((~np.isnan(grid)).sum())
    return int(mask_clean.sum()), n_tissue


def main():
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(cfg.PREDICTIONS_CSV, encoding="utf-8-sig").set_index("slide_id")
    rows = []
    for f in sorted(cfg.SCORES_DIR.glob("*.npy")):
        sid = f.stem
        try:
            n_mask, n_tissue = render_one(sid, float(preds.loc[sid, "predicted_fraction"]))
            rows.append({"slide_id": sid, "mask_patches": n_mask,
                         "tissue_patches": n_tissue,
                         "mask_fraction": round(n_mask / max(n_tissue, 1), 4)})
            print(f"  {sid}: mask={n_mask}/{n_tissue} ✓")
        except Exception as e:
            print(f"  {sid}: 失败 {type(e).__name__}: {e}")
    pd.DataFrame(rows).to_csv(MASK_DIR / "mask_fractions.csv", index=False, encoding="utf-8-sig")
    print(f"[MASK] 输出目录 {MASK_DIR}")


if __name__ == "__main__":
    main()
