# -*- coding: utf-8 -*-
"""M2 首版热图：OOF patch 分数按坐标网格重排 → jet 着色 → 半透明叠加到切片缩略图。

输入：outputs/logs/scores/{slide_id}.npy（train_baseline.py 产出）+ coords/{slide_id}.csv。
输出：outputs/heatmaps/{slide_id}_heatmap.png（overlay 图）。
注意：patch 分数来自该切片所在测试折的 3 种子集成均值（OOF，无泄漏）。
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
from configs import baseline_m2 as cfg

# ---------------- 配置（全部在代码内） ----------------
PATCH_SIZE_LEVEL0 = 512   # M1 registry：40x 原倍率下 512px = 20x 目标倍率下 256px
THUMB_MAX_SIDE = 2000     # 缩略图最长边
ALPHA = 0.45              # 热图叠加强度
HEATMAP_DIR = cfg.OUTPUTS_DIR / "heatmaps"


def render_one(slide_id):
    scores = np.load(cfg.SCORES_DIR / f"{slide_id}.npy")
    coords = pd.read_csv(cfg.COORDS_DIR / f"{slide_id}.csv")
    assert len(scores) == len(coords), f"{slide_id}: 分数与坐标行数不一致"

    # 网格重排（level0 像素坐标 → 网格行列）
    gx = (coords["x"].values // PATCH_SIZE_LEVEL0).astype(int)
    gy = (coords["y"].values // PATCH_SIZE_LEVEL0).astype(int)
    W, H = gx.max() + 1, gy.max() + 1
    grid = np.full((H, W), np.nan, dtype=np.float32)
    grid[gy, gx] = scores

    # 切片缩略图
    slide = openslide.open_slide(str(cfg.WSI_DIR / f"{slide_id}.svs"))
    w0, h0 = slide.dimensions
    scale = min(1.0, THUMB_MAX_SIDE / max(w0, h0))
    thumb = np.asarray(slide.get_thumbnail((int(w0 * scale), int(h0 * scale))).convert("RGB"))
    slide.close()

    # 热图着色并缩放到缩略图尺寸（level0 网格与缩略图的比例由各自相对 level0 的缩放决定）
    cmap = plt.get_cmap("jet")
    heat_rgba = cmap(np.nan_to_num(grid, nan=0.0))  # (H, W, 4) float [0,1]
    mask = ~np.isnan(grid)                          # 仅组织 patch 区域参与叠加
    heat_img = Image.fromarray((heat_rgba[..., :3] * 255).astype(np.uint8)).resize(
        (thumb.shape[1], thumb.shape[0]), Image.BILINEAR)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (thumb.shape[1], thumb.shape[0]), Image.BILINEAR)
    heat = np.asarray(heat_img).astype(np.float32) / 255.0
    m = (np.asarray(mask_img) > 127)[..., None]

    overlay = thumb.astype(np.float32) / 255.0
    overlay = np.where(m, overlay * (1 - ALPHA) + heat * ALPHA, overlay)

    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(HEATMAP_DIR / f"{slide_id}_heatmap.png")
    return float(np.nanmax(grid)), int(mask.sum())


def main():
    score_files = sorted(cfg.SCORES_DIR.glob("*.npy"))
    print(f"待渲染 {len(score_files)} 张")
    for f in score_files:
        sid = f.stem
        try:
            vmax, n = render_one(sid)
            print(f"  {sid}: patches={n}, max_score={vmax:.3f} ✓")
        except Exception as e:
            print(f"  {sid}: 失败 {type(e).__name__}: {e}")
    print(f"[HEATMAP] 输出目录 {HEATMAP_DIR}")


if __name__ == "__main__":
    main()
