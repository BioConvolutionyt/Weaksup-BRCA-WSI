# -*- coding: utf-8 -*-
"""单切片 TRIDENT 直驱（绕过 run_batch_of_slides 的 mpp 传递缺陷）。

背景：TCGA-OL-A5RX 的 SVS 元数据缺失 mpp 字段；run_batch_of_slides 在
selected_wsi_paths 分支下不会把 custom_list_of_wsis 的 mpp 传给 WSI 对象
（TRIDENT 该版本的缺陷），故用 Processor 直驱（已验证 mpp=0.25 生效）。

假设声明：mpp=0.25 为人工设定（TCGA-BRCA DX 切片典型 40x 扫描），
若特征 QC 显示该切片为离群点，将剔除并记录。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "TRIDENT-main"))

from trident import Processor
from trident.patch_encoder_models.load import encoder_factory
from trident.segmentation_models.load import segmentation_model_factory

# ---------------- 配置（全部在代码内） ----------------
JOB_DIR = PROJECT_ROOT / "processed"
SUPP_DIR = PROJECT_ROOT / "Supplementary Data"
MPP_LIST_CSV = JOB_DIR / "a5rx_mpp_list.csv"   # wsi(相对路径), mpp
DEVICE = "cuda:0"
BATCH_SIZE = 32


def main():
    processor = Processor(
        job_dir=str(JOB_DIR),
        wsi_source=str(SUPP_DIR),
        custom_list_of_wsis=str(MPP_LIST_CSV),
        skip_errors=False,
    )
    try:
        seg_model = segmentation_model_factory("grandqc", confidence_thresh=0.5)
        processor.run_segmentation_job(seg_model, seg_mag=seg_model.target_mag,
                                       holes_are_tissue=True, batch_size=BATCH_SIZE,
                                       device=DEVICE)
        processor.run_patching_job(target_magnification=20, patch_size=256, overlap=0)
        encoder = encoder_factory("uni_v1")  # 本地权重经 local_ckpts.json 注册
        processor.run_patch_feature_extraction_job(
            coords_dir="20x_256px_0px_overlap", patch_encoder=encoder,
            device=DEVICE, saveas="h5", batch_limit=BATCH_SIZE)
    finally:
        processor.release()
    print("[RESCUE] A5RX 处理完成")


if __name__ == "__main__":
    main()
