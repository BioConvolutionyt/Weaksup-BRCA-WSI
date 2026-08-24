# -*- coding: utf-8 -*-
"""
从 GDC biospecimen metadata 获取 25 张 TCGA-BRCA WSI 的 slide-level 弱标签。

方案说明（关键事实）：
- percent_* 字段挂在测序用组织切片（TS slide，如 ...-01A-01-TS1）上；
  我们手上的诊断切片（DX，即 .svs 对应的 slide 实体）不携带 percent 字段。
- 因此采用 case 级聚合：取同一 case 下**原发肿瘤样本（sample code 01）**中
  携带 percent_tumor_cells 的切片，取均值作为该 case 唯一 DX 切片的弱标签。
  （曾放宽至 01-09，因 A6IX/A1ES 混入转移灶样本 06A 而收紧；A1ES 由 67.5 修正为 75。）
- 该切面错位（TS ≠ DX）是已声明的标签噪声来源，逐行记录于 qc_note。

弱标签定义决策（2026-08-20 定稿）：
- 主弱标签 weak_fraction = percent_tumor_cells / 100（GDC 官方字段，最可靠）。
- 备选/敏感性检验：percent_tumor_nuclei 列已保留在 CSV 中，可替换主标签
  重算 Pearson/Spearman/MAE 作口径鲁棒性对照。
- 口径声明：percent_tumor_cells 为"细胞数占比"（BCR 病理医生目测估计），
  与 pipeline 预测的"面积占比"存在系统性错位；须在技术报告中显式声明，
  并作为失败案例分析的固定归因项（文献依据：D'Amato 2025 主张面积口径更优，
  其噪声实验表明 ±30% 噪声内回归框架仍稳健）。

运行结果摘要（2026-08-20，GDC /cases expand=samples.portions.slides）：
- 覆盖率 25/25；weak_fraction 分布 min=0.475, median=0.85, mean=0.81, max=1.00。
- 分布集中在中高区间、无低占比样本 → D'Amato 放大技术（五次根变换）弃用。
- QC 标记（失败案例分析的优先审查对象）：
  * TCGA-AN-A0XP：头号噪声嫌疑——两来源切片估计 25 vs 70（均值 47.5），
    且 cells(47.5) < nuclei(80) 口径倒挂。
  * 高淋巴细胞浸润(40%)：C8-A131 / E9-A5UO / Z7-A8R5。
  * 高坏死(20%)：C8-A131 / GI-A2C8。

输出：
- data/weak_labels.csv      主交付物（schema 对齐项目文档 3.3 + 辅助字段）
- data/weak_labels_raw.json GDC 原始响应（溯源用）
"""

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------- 配置（全部在代码内，不用命令行参数） ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAPPING_CSV = PROJECT_ROOT / "TCGA_Download" / "slide_rename_mapping.csv"
OUT_CSV = PROJECT_ROOT / "data" / "weak_labels.csv"
RAW_JSON = PROJECT_ROOT / "data" / "weak_labels_raw.json"

GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"

PERCENT_FIELDS = [
    "percent_tumor_cells",
    "percent_tumor_nuclei",
    "percent_normal_cells",
    "percent_stromal_cells",
    "percent_necrosis",
    "percent_lymphocyte_infiltration",
]

CASE_FIELDS = [
    "submitter_id",
    "samples.submitter_id",
    "samples.tissue_type",
    "samples.portions.submitter_id",
    "samples.portions.slides.submitter_id",
    "samples.portions.slides.section_location",
] + [f"samples.portions.slides.{f}" for f in PERCENT_FIELDS]

HTTP_TIMEOUT = 60
N_RETRIES = 3


def read_mapping(mapping_csv):
    """读取映射表，返回 [(slide_barcode, wsi_path)]。

    兼容两种来源：TCGA_Download/slide_rename_mapping.csv（new_filename 列）
    与 data/supplementary_mapping.csv（wsi_path 列，2026-08-22 扩展）。
    """
    rows = []
    with open(mapping_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            wsi_path = r.get("wsi_path") or f"TCGA_Download/{r['new_filename']}"
            rows.append((r["slide_id"], wsi_path))
    return rows


def case_of(barcode):
    """TCGA-AC-A6IV-01Z-00-DX1 -> TCGA-AC-A6IV"""
    return "-".join(barcode.split("-")[:3])


def fetch_cases(case_ids):
    """按 case 批量查询 GDC /cases，expand 到 slides 层级。"""
    payload = {
        "filters": json.dumps({
            "op": "in",
            "content": {"field": "cases.submitter_id", "value": sorted(case_ids)},
        }),
        "fields": ",".join(CASE_FIELDS),
        "expand": "samples.portions.slides",
        "format": "json",
        "size": "100",
    }
    url = GDC_CASES_ENDPOINT + "?" + urllib.parse.urlencode(payload)
    last_err = None
    for attempt in range(N_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as resp:
                return json.loads(resp.read())
        except Exception as e:  # noqa: BLE001 - 简单重试即可
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GDC API 请求失败（重试 {N_RETRIES} 次后放弃）: {last_err}")


def collect_case_labels(case_hit):
    """
    从单个 case 的 biospecimen 树中收集弱标签。
    只接受原发肿瘤样本（barcode 第 4 段为 01）下带 percent 值的切片。
    返回 {field: [values]} 与来源切片列表。
    """
    values = {f: [] for f in PERCENT_FIELDS}
    source_slides = []
    for sample in case_hit.get("samples", []):
        sample_id = sample.get("submitter_id", "")
        code = sample_id.split("-")[3][:2] if len(sample_id.split("-")) >= 4 else ""
        if code != "01":
            continue  # 只取原发性肿瘤样本（01）；排除复发(02)/转移(06)及正常(10/11)样本
        for portion in sample.get("portions", []):
            for slide in portion.get("slides", []):
                if slide.get("percent_tumor_cells") is None:
                    continue  # DX 切片等无标注实体在此被跳过
                source_slides.append(slide.get("submitter_id", ""))
                for f in PERCENT_FIELDS:
                    v = slide.get(f)
                    if v is not None:
                        values[f].append(float(v))
    return values, source_slides


def main(mapping_csv=MAPPING_CSV, out_csv=OUT_CSV, raw_json=RAW_JSON):
    mapping_csv, out_csv, raw_json = Path(mapping_csv), Path(out_csv), Path(raw_json)
    mapping = read_mapping(mapping_csv)
    case_ids = sorted({case_of(b) for b, _ in mapping})
    print(f"WSI 数量: {len(mapping)}，涉及 case 数量: {len(case_ids)}")

    resp = fetch_cases(case_ids)
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(resp, f, ensure_ascii=False, indent=1)
    hits = {h["submitter_id"]: h for h in resp["data"]["hits"]}
    print(f"GDC 返回 case 数: {len(hits)} / {len(case_ids)}")

    header = [
        "slide_id", "wsi_path", "case_id", "weak_fraction", "label_source",
    ] + PERCENT_FIELDS + [
        "n_source_slides", "source_slide_ids", "qc_note",
    ]

    rows, missing = [], []
    for barcode, wsi_name in mapping:
        cid = case_of(barcode)
        hit = hits.get(cid)
        if hit is None:
            missing.append((barcode, "case 未在 GDC 返回中"))
            continue
        values, source_slides = collect_case_labels(hit)
        if not values["percent_tumor_cells"]:
            missing.append((barcode, "该 case 无携带 percent_tumor_cells 的切片"))
            continue

        rec = {
            "slide_id": barcode,
            "wsi_path": wsi_name,
            "case_id": cid,
            "weak_fraction": round(sum(values["percent_tumor_cells"]) / len(values["percent_tumor_cells"]) / 100.0, 4),
            "label_source": "biospecimen_metadata_case_level_TS",
        }
        for f in PERCENT_FIELDS:
            rec[f] = round(sum(values[f]) / len(values[f]), 2) if values[f] else ""
        rec["n_source_slides"] = len(source_slides)
        rec["source_slide_ids"] = "|".join(source_slides)
        notes = []
        if len(source_slides) > 1:
            vals = values["percent_tumor_cells"]
            notes.append(f"多来源切片取均值(range {min(vals):.0f}-{max(vals):.0f})")
        notes.append("TS切面标签用于DX切片(口径=细胞数占比)")
        rec["qc_note"] = "; ".join(notes)
        rows.append(rec)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    print(f"\n已写出 {out_csv}：{len(rows)} / {len(mapping)} 张切片获得弱标签")
    if missing:
        print("未获得标签的切片：")
        for b, why in missing:
            print(f"  - {b}: {why}")

    fracs = sorted(r["weak_fraction"] for r in rows)
    if fracs:
        n = len(fracs)
        med = fracs[n // 2] if n % 2 else (fracs[n // 2 - 1] + fracs[n // 2]) / 2
        print(f"\nweak_fraction 分布: min={fracs[0]:.2f}, median={med:.2f}, "
              f"max={fracs[-1]:.2f}, mean={sum(fracs) / n:.2f}")


if __name__ == "__main__":
    main()
