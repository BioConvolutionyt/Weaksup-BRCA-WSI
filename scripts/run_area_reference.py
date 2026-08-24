# -*- coding: utf-8 -*-
"""面积口径参照生成：WSInfer 预训练乳腺肿瘤模型对 75 张切片推理。

合规声明：该模型仅用于评估参照（验证口径错位假设），不进入训练与主推理流程——
同 SMMILe/Geo-MIL"像素标注仅用于评估"的性质，报告中显式声明。

流程：
  1) 为 50 张补充切片建硬链接暂存目录（零额外磁盘，输出名即为条形码）；
  2) WSInfer 分两批运行（首批 25 张 TCGA_Download / 补充 50 张暂存目录）；
  3) 汇总 model-outputs-csv → data/area_reference.csv：
     area_frac_hard = count(prob_Tumor>0.5)/n_patches；area_frac_soft = mean(prob_Tumor)。
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ---------------- 配置（全部在代码内） ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TCGA_DIR = PROJECT_ROOT / "TCGA_Download"
SUPP_MAP = PROJECT_ROOT / "data" / "supplementary_mapping.csv"
STAGE_DIR = PROJECT_ROOT / "processed" / "_wsinfer_supp"      # 暂存硬链接（用后清理）
RESULTS_MAIN = PROJECT_ROOT / "processed" / "_wsinfer_main"
RESULTS_SUPP = PROJECT_ROOT / "processed" / "_wsinfer_supp_out"
OUT_CSV = PROJECT_ROOT / "data" / "area_reference.csv"

MODEL = "breast-tumor-resnet34.tcga-brca"
BATCH_SIZE = 32
NUM_WORKERS = 4


def run_wsinfer(wsi_dir, results_dir):
    cmd = [sys.executable, "-m", "wsinfer", "run",
           "--wsi-dir", str(wsi_dir), "--results-dir", str(results_dir),
           "--model", MODEL, "--batch-size", str(BATCH_SIZE),
           "--num-workers", str(NUM_WORKERS)]
    print("[WSInfer]", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    print(f"[WSInfer] exit={proc.returncode} → {results_dir}")
    return proc.returncode


def build_stage():
    """为补充切片建硬链接暂存目录（文件名 = 条形码）。"""
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(SUPP_MAP, encoding="utf-8-sig")
    n_new = 0
    for _, r in m.iterrows():
        dst = STAGE_DIR / f"{r['slide_id']}.svs"
        if not dst.exists():
            import os
            os.link(r["wsi_path"], dst)
            n_new += 1
    print(f"[STAGE] 硬链接 {n_new} 个（共 {len(m)}）")


def aggregate(results_dir, source_tag):
    """汇总一批 WSInfer 输出为参照记录。"""
    rows = []
    out_dir = results_dir / "model-outputs-csv"
    for f in sorted(out_dir.glob("*.csv")):
        df = pd.read_csv(f)
        n = len(df)
        if n == 0:
            continue
        rows.append({
            "slide_id": f.stem,
            "n_patches_ref": n,
            "area_frac_hard": round(float((df["prob_Tumor"] > 0.5).mean()), 4),
            "area_frac_soft": round(float(df["prob_Tumor"].mean()), 4),
            "source": source_tag,
        })
    return rows


def main():
    build_stage()
    rc1 = run_wsinfer(TCGA_DIR, RESULTS_MAIN)
    rc2 = run_wsinfer(STAGE_DIR, RESULTS_SUPP)

    rows = aggregate(RESULTS_MAIN, "wsinfer_breast_tumor_resnet34") + \
           aggregate(RESULTS_SUPP, "wsinfer_breast_tumor_resnet34")
    out = pd.DataFrame(rows).drop_duplicates(subset="slide_id")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[REF] {len(out)}/75 张面积参照 → {OUT_CSV}；wsinfer exit: {rc1}/{rc2}")

    # 清理暂存硬链接
    for f in STAGE_DIR.glob("*.svs"):
        f.unlink()
    print("[STAGE] 暂存硬链接已清理")


if __name__ == "__main__":
    main()
