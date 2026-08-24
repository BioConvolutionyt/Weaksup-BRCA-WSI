# -*- coding: utf-8 -*-
"""A5RX 面积参照救援：WSInfer 因该切片元数据缺失 mpp 而跳过，此处以 monkeypatch
注入 mpp=0.25（与 TRIDENT 救援同一假设）单独补跑。

合规声明不变：仅用于评估参照，不进入训练与主推理流程。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- 在导入 wsinfer 前先准备好 monkeypatch ----
import wsinfer.patchlib as patchlib
import wsinfer.wsi as wsi_mod

_ASSUMED_MPP = 0.25  # TCGA-BRCA DX 切片典型 40x 扫描；QC 已验证其特征非离群

_orig_get_avg_mpp = patchlib.get_avg_mpp


def _patched_get_avg_mpp(slide_path):
    try:
        return _orig_get_avg_mpp(slide_path)
    except Exception:
        print(f"[RESCUE] mpp 元数据缺失，注入假设值 {_ASSUMED_MPP}: {slide_path}")
        return _ASSUMED_MPP


patchlib.get_avg_mpp = _patched_get_avg_mpp
wsi_mod.get_avg_mpp = _patched_get_avg_mpp

import wsinfer_zoo.client
from wsinfer.modellib.run_inference import run_inference

STAGE = PROJECT_ROOT / "processed" / "_wsinfer_supp"          # 沿用暂存目录（仅 A5RX）
RESULTS = PROJECT_ROOT / "processed" / "_wsinfer_supp_out"    # 沿用补充批输出目录
SLIDE = STAGE / "TCGA-OL-A5RX-01Z-00-DX1.svs"


def main():
    # 确保暂存目录中只有 A5RX
    if not SLIDE.exists():
        import os
        m = None
        import pandas as pd
        supp = pd.read_csv(PROJECT_ROOT / "data" / "supplementary_mapping.csv")
        row = supp[supp.slide_id == "TCGA-OL-A5RX-01Z-00-DX1"].iloc[0]
        os.link(row["wsi_path"], SLIDE)
        print("[RESCUE] 硬链接已建")

    models = wsinfer_zoo.client.load_registry()
    reg_model = models.get_model_by_name("breast-tumor-resnet34.tcga-brca")
    model_info = reg_model.load_model_torchscript()
    cfg_obj = model_info.config
    print(f"[RESCUE] 模型配置: patch_size_px={cfg_obj.patch_size_pixels}, "
          f"spacing={cfg_obj.spacing_um_px}")

    # 1) patching（monkeypatch 已注入 mpp）
    patchlib.segment_and_patch_directory_of_slides(
        wsi_dir=STAGE, save_dir=RESULTS,
        patch_size_px=cfg_obj.patch_size_pixels,
        patch_spacing_um_px=cfg_obj.spacing_um_px,
    )
    # 2) inference
    failed_patch, failed_infer = run_inference(
        wsi_dir=STAGE, results_dir=RESULTS, model_info=model_info,
        batch_size=32, num_workers=4)
    print(f"[RESCUE] 完成。patch 失败: {failed_patch} | 推理失败: {failed_infer}")


if __name__ == "__main__":
    main()
