# -*- coding: utf-8 -*-
"""临时：WSInfer 单切片冒烟（硬链接暂存 + 输出格式验证）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\PythonProject\Fudan Internship")
STAGE = ROOT / "processed" / "_wsinfer_test"
STAGE.mkdir(parents=True, exist_ok=True)

src = ROOT / "TCGA_Download" / "TCGA-AC-A6IV-01Z-00-DX1.svs"
dst = STAGE / "TCGA-AC-A6IV-01Z-00-DX1.svs"
if not dst.exists():
    import os
    os.link(src, dst)  # 硬链接，零额外磁盘
    print("hardlink created")

cmd = [
    sys.executable, "-m", "wsinfer", "run",
    "--wsi-dir", str(STAGE),
    "--results-dir", str(ROOT / "processed" / "_wsinfer_test_out"),
    "--model", "breast-tumor-resnet34.tcga-brca",
    "--batch-size", "32", "--num-workers", "4",
]
print("CMD:", " ".join(cmd))
proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
print("exit:", proc.returncode)
print(proc.stdout[-2000:])
print(proc.stderr[-2000:])
