# -*- coding: utf-8 -*-
"""
M1 全量预处理：25 张 TCGA-BRCA WSI → tissue 分割 → patch 坐标 → UNI v1 特征。

流程：
  1) 调用 TRIDENT（源码仓 TRIDENT-main）批处理，--task all，断点续跑由其内部保证
     （已完成的 seg/coords/feat 输出会被自动跳过）。
  2) 将 TRIDENT 产出的 h5 转存为本项目交付结构：
       features/{slide_id}.pt   —— torch.float32, (N, 1024)
       coords/{slide_id}.csv    —— x, y（level0 像素坐标）
  3) 生成 data/preprocess_registry.csv（每切片的 patch 数、估计组织面积、尺寸、mpp）。

设计决策（溯源：Modeling Plan.md §2.1）：
  - 分割器 grandqc（TRIDENT 内建，含 artifact 质检能力）；20x / 256px / overlap 0。
  - 特征编码器 uni_v1，本地权重经 TRIDENT 注册表 local_ckpts.json 指向 UNI/pytorch_model.bin。
  - 5.3 公式中 tissue_area_i 权重：patch 等尺寸且已过组织阈值过滤 → 均匀权重近似，
    切片级组织面积 ≈ n_patches × (256px × 0.5µm)² 记入 registry。
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

# ---------------- 配置（全部在代码内，不用命令行参数） ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRIDENT_DIR = PROJECT_ROOT / "TRIDENT-main"
WSI_DIR = PROJECT_ROOT / "TCGA_Download"
JOB_DIR = PROJECT_ROOT / "processed"
FEATURES_OUT = PROJECT_ROOT / "features"
COORDS_OUT = PROJECT_ROOT / "coords"
REGISTRY_CSV = PROJECT_ROOT / "data" / "preprocess_registry.csv"

PATCH_ENCODER = "uni_v1"
SEGMENTER = "grandqc"
MAG = 20
PATCH_SIZE = 256
OVERLAP = 0
BATCH_SIZE = 32          # 4060 8GB 实测安全；OOM 时降至 16
COORDS_SUBDIR = f"{MAG}x_{PATCH_SIZE}px_{OVERLAP}px_overlap"
PATCH_AREA_MM2 = (PATCH_SIZE * 0.5e-3) ** 2  # 20x 下 0.5 µm/px


def run_trident_batch(wsi_dir=WSI_DIR, search_nested=False):
    """调用 TRIDENT 批处理（task=all），流式打印日志。"""
    cmd = [
        sys.executable,
        str(TRIDENT_DIR / "run_batch_of_slides.py"),
        "--task", "all",
        "--wsi_dir", str(wsi_dir),
        "--wsi_ext", ".svs",
        "--job_dir", str(JOB_DIR),
        "--segmenter", SEGMENTER,
        "--patch_encoder", PATCH_ENCODER,
        "--mag", str(MAG),
        "--patch_size", str(PATCH_SIZE),
        "--batch_size", str(BATCH_SIZE),
        "--skip_errors",
    ]
    if search_nested:
        cmd.append("--search_nested")
    print("[TRIDENT]", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(TRIDENT_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    print(f"[TRIDENT] 批处理结束，exit={proc.returncode}，耗时 {(time.time()-t0)/60:.1f} min")
    return proc.returncode


def convert_outputs(mapping_csv=None, append_registry=False):
    """h5 → features/*.pt + coords/*.csv，并生成 preprocess_registry.csv。

    参数化（2026-08-22 补充数据扩展）：
    - mapping_csv 缺省为 TCGA_Download 首批映射表；可传 data/supplementary_mapping.csv。
    - 补充数据的 h5 文件名含 GDC UUID 后缀（barcode.UUID.h5），按 slide_id 前缀匹配。
    - append_registry=True 时合并进既有 registry（去重），否则覆盖。
    """
    feat_dir = JOB_DIR / COORDS_SUBDIR / f"features_{PATCH_ENCODER}"
    patch_dir = JOB_DIR / COORDS_SUBDIR / "patches"
    FEATURES_OUT.mkdir(exist_ok=True)
    COORDS_OUT.mkdir(exist_ok=True)

    if mapping_csv is None:
        mapping_csv = WSI_DIR / "slide_rename_mapping.csv"
    mapping = pd.read_csv(mapping_csv, encoding="utf-8-sig")
    slide_ids = mapping["slide_id"].tolist()

    def find_h5(directory, sid, suffix):
        exact = directory / f"{sid}{suffix}"
        if exact.exists():
            return exact
        hits = sorted(directory.glob(f"{sid}.*{suffix}"))
        return hits[0] if hits else None

    rows, missing = [], []
    for sid in slide_ids:
        feat_h5 = find_h5(feat_dir, sid, ".h5")
        patch_h5 = find_h5(patch_dir, sid, "_patches.h5")
        if feat_h5 is None or patch_h5 is None:
            missing.append(sid)
            continue

        with h5py.File(feat_h5, "r") as f:
            feats = torch.from_numpy(np.asarray(f["features"], dtype=np.float32))
        with h5py.File(patch_h5, "r") as f:
            coords = np.asarray(f["coords"])
            attrs = dict(f["coords"].attrs)

        torch.save(feats, FEATURES_OUT / f"{sid}.pt")
        pd.DataFrame(coords, columns=["x", "y"]).to_csv(COORDS_OUT / f"{sid}.csv", index=False)

        n = feats.shape[0]
        rows.append({
            "slide_id": sid,
            "n_patches": n,
            "feature_dim": feats.shape[1],
            "est_tissue_area_mm2": round(n * PATCH_AREA_MM2, 3),
            "level0_width": attrs.get("level0_width", ""),
            "level0_height": attrs.get("level0_height", ""),
            "level0_magnification": attrs.get("level0_magnification", ""),
            "target_magnification": attrs.get("target_magnification", MAG),
            "patch_size": attrs.get("patch_size", PATCH_SIZE),
            "patch_size_level0": attrs.get("patch_size_level0", ""),
            "encoder": PATCH_ENCODER,
        })

    reg = pd.DataFrame(rows)
    if append_registry and REGISTRY_CSV.exists():
        old = pd.read_csv(REGISTRY_CSV, encoding="utf-8-sig")
        reg = pd.concat([old, reg]).drop_duplicates(subset="slide_id", keep="last")
    reg.to_csv(REGISTRY_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[CONVERT] 完成 {len(rows)}/{len(slide_ids)} 张；缺失: {missing if missing else '无'}")
    print(reg[["slide_id", "n_patches", "est_tissue_area_mm2"]].to_string(index=False))
    return reg, missing


def main(mapping_csv=None, wsi_dir=WSI_DIR, search_nested=False, append_registry=False):
    t0 = time.time()
    rc = run_trident_batch(wsi_dir=wsi_dir, search_nested=search_nested)
    if rc != 0:
        print("[WARN] TRIDENT 批处理返回非零，仍尝试转换已完成部分（skip_errors 语义）", flush=True)
    reg, missing = convert_outputs(mapping_csv=mapping_csv, append_registry=append_registry)

    summary = {
        "n_slides_total": len(reg),
        "n_slides_done": int(len(reg)),
        "missing": missing,
        "total_patches": int(reg["n_patches"].sum()) if len(reg) else 0,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    with open(JOB_DIR / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
