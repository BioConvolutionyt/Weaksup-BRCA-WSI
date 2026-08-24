# 基于弱监督学习的 TCGA-BRCA 乳腺癌 WSI 肿瘤区域定位与面积占比估计

在不使用任何像素级肿瘤标注的前提下，以切片级弱标签（GDC 元数据中的肿瘤百分比字段及面积口径参照）为唯一监督信号，建立「UNI 特征提取 → MIL 占比回归 → 热图与掩膜生成」的完整可复现管线，输出 patch 级肿瘤概率、WSI 热图、二值掩膜与预测面积占比，并完成池化硬度、标签噪声、可信度协同推断、面积硬约束等四项探索实验（6.1–6.4）。

- 数据：75 张 TCGA-BRCA H&E 全切片
- 方法：UNI 基础模型提取 patch 特征，分位数聚合（q75）MIL 回归，可信度加权 + 空间平滑 + 图传播的后处理交付配置
- 交付模型主要指标（75 张，5 折 OOF）：Pearson 0.807 / Spearman 0.794 / MAE 0.108 / RMSE 0.137（以面积口径为参照）

## 环境安装

```bash
conda create -n wsi python=3.11 -y
conda activate wsi
pip install -r requirements.txt
```

另需三项手动安装：

1. **OpenSlide 系统动态库**（openslide-python 的底层依赖）：Windows 从 [OpenSlide 官方发布页](https://openslide.org/download/)下载二进制并把 bin 目录加入 PATH；Linux 使用 `apt install libopenslide0`。
2. **TRIDENT**（WSI 分割、patch 提取、UNI 特征提取框架）：`git clone https://github.com/mahmoodlab/TRIDENT`，本项目基于 2026-08 的 main 分支源码运行。预处理脚本默认在 `TRIDENT-main/` 目录下调用其源码。
3. **UNI v1 权重**：在 HuggingFace 申请并接受 [mahmoodlab/UNI](https://huggingface.co/mahmoodlab/UNI) 的许可后下载，放置于 `UNI/` 目录。

开发环境参考：Windows 11 + RTX 4060 8GB + CUDA 12.1。

## 数据目录结构

数据不入库，需自行下载并按如下结构放置：

```
TCGA_Download/            # 25 张主队列 .svs（GDC 下载）+ slide_rename_mapping.csv
Supplementary Data/       # 50 张补充队列 .svs + gdc_manifest.txt
data/
  weak_labels.csv                # 75 张切片弱标签（GDC 口径，脚本生成）
  area_reference.csv             # 面积口径参照（WSInfer 生成，评估用）
  supplementary_mapping.csv      # 补充队列映射
  preprocess_registry.csv        # 预处理登记表
```

WSI 下载流程：GDC 门户筛选 TCGA-BRCA 的 Diagnostic Slide → 导出 manifest → 以 gdc-client 批量下载。完整流程记录见技术报告第 0 节。

## 复现流程

以下命令均在仓库根目录执行，所有配置项在各 `configs/*.py` 与脚本顶部常量中定义，不接受命令行参数。

### 1. 提取 patch（组织分割 + 坐标）

```bash
python scripts/run_preprocessing.py
```

调用 TRIDENT 完成组织分割与 20× 放大级别下 512×512 patch 的坐标提取，产物写入 `processed/` 与 `coords/`，并登记 `data/preprocess_registry.csv`。缺少 MPP 元数据的个别切片由 `scripts/run_trident_processor.py` 以估计 MPP 单独处理。

### 2. 提取 UNI 特征

同一脚本 `run_preprocessing.py` 的特征提取阶段自动完成（UNI v1，1024 维，按切片缓存为 `features/{slide_id}.pt`）。特征提取前请确认 `UNI/` 权重就位。

### 3. 获取弱标签与面积参照

```bash
python scripts/fetch_weak_labels.py        # GDC biospecimen 字段 → data/weak_labels.csv
python scripts/run_area_reference.py       # WSInfer 肿瘤面积参照 → data/area_reference.csv
python scripts/qc_checks.py                # 数据质量检查（UMAP / imagehash / 标签审核）
```

### 4. 训练弱监督模型

```bash
python scripts/train_baseline.py           # 基线（MeanPool，GDC 标签）
python scripts/run_spatial_area.py         # 交付配置（面积口径 + q75 + 可信度加权 + 平滑）
python scripts/run_prop_amplify_experiment.py  # 图标签传播变体（交付配置的完整形态）
```

训练协议为 5 折分层交叉验证 × 3 种子集成，OOF 预测与 patch 分数写入 `outputs/`。

### 5. 生成 heatmap 和 mask

```bash
python scripts/make_heatmaps.py            # outputs/heatmaps/ 热图叠加
python scripts/make_masks.py               # outputs/masks/ 二值掩膜与叠加
```

输入为训练阶段保存的 `outputs/logs/scores/{slide_id}.npy` 与 `coords/`。6.4 探索提出的传播×3 + Top-K 掩膜策略见 `scripts/run_explore_64.py`。

### 6. 复现实验结果

| 实验 | 脚本 | 主要产物 |
|---|---|---|
| 弱标签获取与 QC | `fetch_weak_labels.py` / `qc_checks.py` | `data/weak_labels.csv`、`data/qc_*.csv` |
| M2 基线与诊断 | `train_baseline.py` / `diagnose_training.py` | `outputs/logs/training_history.csv` |
| M3.5 标签侧实验 | `run_label_experiments.py` | `outputs/logs/label_experiments.csv` |
| M4 聚合器消融 | `run_ablations.py` | `outputs/logs/ablation_aggregators.csv` |
| 面积口径验证 | `run_area_label_experiment.py` | `outputs/logs/area_label_experiment.csv` |
| 数据量扩展（25→75） | `run_volume_experiment.py` | `outputs/logs/volume_experiment.csv` |
| 空间约束（面积口径重做） | `run_spatial_area.py` | `outputs/logs/spatial_variants_area.csv` |
| 容差约束 | `run_tolerance_experiment.py` | `outputs/logs/tolerance_experiment.csv` |
| 传播与放大 | `run_prop_amplify_experiment.py` | `outputs/logs/prop_amplify_experiment.csv` |
| SRG 热图忠实度 | `eval_srg.py` | `outputs/logs/srg_results.csv` |
| 5+5 案例分析 | `case_analysis.py` | `outputs/figures/cases/` |
| 探索 6.1–6.4 | `run_explore_61.py` … `run_explore_64.py` | `exploration/results/` |
| 探索报告（Word） | `make_exploration_report.py` | `exploration/探索实验报告.docx` |

## 仓库结构

```
configs/          # 实验配置（baseline_m2 / spatial_m3 / label_m35 / ablation_m4）
src/              # 核心库：models.py（MIL 聚合器族）、training.py（CV 训练循环）、
                  # spatial.py（平滑/传播/形态学）、noise.py（标签噪声注入）
scripts/          # 各阶段入口脚本（见上表）
data/             # 弱标签、面积参照、映射与 QC 记录（小型 CSV/JSON）
exploration/      # 6.1–6.4 探索笔记、results/ 图表与数据、探索报告 docx
```

`outputs/`（8.2 结果文件，另行打包提交）、`features/`、`processed/`、`coords/`、`UNI/`、`TCGA_Download/` 等大体量或可再生内容不入库，按上文「复现流程」各节命令即可在本地完整再生，详见 `.gitignore`。

## 探索实验（6.1–6.4）

| 章节 | 问题 | 结论摘要 |
|---|---|---|
| 6.1 | 标签可信度协同推断的可行性边界 | 残差识别污染 AUC 0.70–0.84，污染率 ≥35% 启用有净收益，能减损不能复原 |
| 6.2 | 池化硬度 × patch 数量 | 最优硬度为中间偏软（gmean-p5），硬度应以比例而非个数参数化，误差与 N 近似无关 |
| 6.3 | 池化硬度 × 标签噪声 | 袋级回归噪声下池化越硬越敏感，误差方差 ∝ σ²/n_eff |
| 6.4 | 面积硬约束 × 空间平滑 | 基数约束 Potts 为 NP 难，传播×3+Top-K 与拉格朗日图割的面积误差优于目标 4–8 倍 |

详见 `exploration/` 下四份探索笔记与《探索实验报告.docx》。
