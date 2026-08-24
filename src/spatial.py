# -*- coding: utf-8 -*-
"""M3 空间约束三件套（溯源：Modeling Plan.md §2.1）。

1. 网格 8 邻域平滑损失（SMMILe MRF 的蒸馏版：superpatch→规则网格，免 SLIC）；
2. InD 免参实例 dropout（SMMILe drop_with_score 官方公式：高分 patch 更可能被丢弃）；
3. 形态学后处理（文档 5.4 选项 5：连通域去小区域 + 填洞）。
"""
import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import remove_small_holes, remove_small_objects


def scores_to_grid(scores, coords_df, patch_size_level0=512):
    """patch 分数按 level0 坐标重排为二维网格（非组织位置为 NaN）。"""
    gx = (coords_df["x"].values // patch_size_level0).astype(np.int64)
    gy = (coords_df["y"].values // patch_size_level0).astype(np.int64)
    grid = np.full((gy.max() + 1, gx.max() + 1), np.nan, dtype=np.float32)
    grid[gy, gx] = scores
    return grid


def build_neighbor_pairs(coords_df, patch_size_level0=512):
    """由 level0 像素坐标构建网格邻接对（4 方向去重：右/下/右下/左下）。

    返回 LongTensor (2, E)：每列是一对相邻 patch 的行索引。
    """
    gx = (coords_df["x"].values // patch_size_level0).astype(np.int64)
    gy = (coords_df["y"].values // patch_size_level0).astype(np.int64)
    index_of = {(x, y): i for i, (x, y) in enumerate(zip(gx, gy))}
    pairs = []
    for x, y, i in zip(gx, gy, range(len(gx))):
        for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
            j = index_of.get((x + dx, y + dy))
            if j is not None:
                pairs.append((i, j))
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(np.array(pairs).T, dtype=torch.long)


def smoothness_loss(scores, pairs):
    """L_smooth = mean_{(i,j)∈E} (s_i − s_j)²；无边时返回 0。"""
    if pairs.shape[1] == 0:
        return scores.new_zeros(())
    return ((scores[pairs[0]] - scores[pairs[1]]) ** 2).mean()


def ind_dropout_mask(scores):
    """InD（SMMILe）：keep = rand > min-max(score)；高分更易被丢弃；至少保留 1 个。"""
    s = scores.detach()
    norm = (s - s.min()) / (s.max() - s.min() + 1e-10)
    keep = torch.rand_like(norm) > norm
    if not bool(keep.any()):
        keep[torch.argmax(norm)] = True
    return keep


def morphology_cleanup(mask_grid, min_region=4, tissue_domain=None, max_hole=16):
    """网格二值 mask 的形态学清理：小洞填充（限量）+ 去孤立小连通域。

    min_region：最小保留连通域面积（patch 数），默认 4（≈0.26 mm²@20x）。
    max_hole：可填充空洞的最大面积（patch 数），默认 16（≈1.0 mm²）——
      限量填充是为了避免 scipy binary_fill_holes 把 mask 包围的大片低分区整体吞没
      （实测会把 71% 的 mask 膨胀到 100%）。
    tissue_domain：组织区域布尔掩码；提供时最终结果限制在组织域内。
    """
    out = remove_small_holes(mask_grid, area_threshold=max_hole)
    out = remove_small_objects(out, min_size=min_region)
    if tissue_domain is not None:
        out = out & tissue_domain
    return out


def build_propagation_graph(feats, coords_df, radius=2, topk=8, patch_size_level0=512):
    """图标签传播的图构建（PaM-MIL 式：空间邻近 × 特征相似）。

    候选边：网格上 Chebyshev 距离 ≤ radius 的 patch 对；
    边权：候选中按 UNI 特征余弦相似度取 topk，softmax 归一。
    返回 (edge_index LongTensor(2,E), edge_weight FloatTensor(E))。
    """
    gx = (coords_df["x"].values // patch_size_level0).astype(np.int64)
    gy = (coords_df["y"].values // patch_size_level0).astype(np.int64)
    index_of = {(x, y): i for i, (x, y) in enumerate(zip(gx, gy))}

    feats_n = torch.nn.functional.normalize(feats, dim=1)
    src, dst, w = [], [], []
    for i, (x, y) in enumerate(zip(gx, gy)):
        cand = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                j = index_of.get((x + dx, y + dy))
                if j is not None:
                    cand.append(j)
        if not cand:
            continue
        cand = torch.tensor(cand, dtype=torch.long)
        sim = (feats_n[i] * feats_n[cand]).sum(dim=1)          # 余弦相似（已归一）
        k = min(topk, len(cand))
        top = sim.topk(k).values
        wij = torch.softmax(top, dim=0)
        src.extend([i] * k)
        dst.extend(cand[sim.topk(k).indices].tolist())
        w.extend(wij.tolist())
    if not src:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(w, dtype=torch.float32)


def graph_propagate(scores, edge_index, edge_weight, alpha=0.5):
    """p̃ = α·p + (1−α)·W·p（沿图做一次传播；W 行归一）。"""
    n = scores.shape[0]
    if edge_index.shape[1] == 0:
        return scores
    agg = torch.zeros_like(scores)
    agg.index_add_(0, edge_index[0], edge_weight * scores[edge_index[1]])
    return alpha * scores + (1 - alpha) * agg


def count_components(mask_grid):
    """mask 的连通域数量（8 连通）。"""
    _, n = ndimage.label(mask_grid, structure=np.ones((3, 3), dtype=int))
    return int(n)


def spatial_metrics(oof_scores, cfg):
    """空间质量指标：连通域数（0.5 mask）与平均邻域分数差，逐切片均值。

    自 scripts/train_spatial.py 迁入（该入口脚本不随仓库提交，见 README）。
    """
    import pandas as pd
    comps, smooths = [], []
    for sid, sc in oof_scores.items():
        coords = pd.read_csv(cfg.COORDS_DIR / f"{sid}.csv")
        grid = scores_to_grid(sc, coords)
        comps.append(count_components(np.nan_to_num(grid, nan=0.0) >= cfg.MASK_THRESHOLD))
        smooths.append(mean_neighbor_diff(grid))
    return float(np.mean(comps)), float(np.mean(smooths))


def mean_neighbor_diff(scores_grid):
    """分数图的平均相邻差异（平滑度指标，越小越平滑；NaN 区域不参与）。"""
    g = scores_grid
    diffs = []
    for axis in (0, 1):
        d = np.abs(np.diff(g, axis=axis))          # 相邻差
        # 只保留两端均非 NaN 的差分
        if axis == 0:
            valid = ~np.isnan(g[:-1, :]) & ~np.isnan(g[1:, :])
        else:
            valid = ~np.isnan(g[:, :-1]) & ~np.isnan(g[:, 1:])
        diffs.append(d[valid])
    diffs = [d for d in diffs if d.size]
    return float(np.concatenate(diffs).mean()) if diffs else 0.0
