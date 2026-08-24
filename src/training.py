# -*- coding: utf-8 -*-
"""M2 训练与评估：5-fold（case 级 + weak_fraction 分箱分层）× 多种子。

协议溯源：D'Amato 2025（分层 5-fold / case 分组 / 早停）；集成：UBC 种子 + PANDA 多折。
"""
import copy

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.models import ABMILInstanceMIL, MeanPoolMIL, make_model


def log_barrier(z, t=5.0):
    """扩展 log-barrier（Silva-Rodríguez 2022 / jusiro mil_histology 官方实现形态）。

    约束 z ≤ 0 为满足侧：z ≤ −1/t² 时取 −log(−z)/t（对数段），
    否则取线性延拓 t·z + log(t²)/t + 1/t（保持可微）。
    用于容差式面积约束：z = |pred − weak| − δ。
    """
    if z <= -1.0 / t ** 2:
        return -torch.log(-z) / t
    return t * z + np.log(1.0 / t ** 2) / t + 1.0 / t


def load_data(cfg):
    """读弱标签表与全部 UNI 特征，返回 (labels_df, feats_dict)。

    M3.5：cfg.LABEL_FIELD 可切换标签口径（默认 percent_tumor_cells）。
    面积口径实验：cfg.LABELS_DF 可直接注入标签 DataFrame（含 slide_id/weak_fraction 列）。
    """
    labels_df = getattr(cfg, "LABELS_DF", None)
    if labels_df is None:
        df = pd.read_csv(cfg.WEAK_LABELS_CSV, encoding="utf-8-sig")
    else:
        df = labels_df.copy()
    label_field = getattr(cfg, "LABEL_FIELD", "percent_tumor_cells")
    if label_field != "percent_tumor_cells":
        df["weak_fraction"] = pd.to_numeric(df[label_field], errors="coerce") / 100.0
        n_missing = int(df["weak_fraction"].isna().sum())
        if n_missing:
            raise ValueError(f"{label_field} 存在 {n_missing} 个缺失值")
    feats = {}
    for sid in df["slide_id"]:
        feats[sid] = torch.load(cfg.FEATURES_DIR / f"{sid}.pt",
                                map_location="cpu", weights_only=True)
    # 6.2：cfg.MAX_PATCHES 子采样 patch 数上限（嵌套设计——同一切片按固定随机
    # 排列取前缀，保证 N=1000 ⊂ N=2000 ⊂ N=5000 ⊂ 全量，N 是唯一变量）
    max_patches = getattr(cfg, "MAX_PATCHES", None)
    if max_patches:
        import zlib
        for sid in df["slide_id"]:
            n = feats[sid].shape[0]
            if n > max_patches:
                rng = np.random.RandomState(zlib.crc32(sid.encode()) & 0xFFFFFFFF)
                idx = np.sort(rng.permutation(n)[:max_patches])
                feats[sid] = feats[sid][torch.from_numpy(idx)]
    return df, feats


def make_folds(df, cfg):
    """按 weak_fraction 分箱分层 + case 级（25 张均为不同 case，QC 已确认）。"""
    y = df["weak_fraction"].values
    bins = pd.qcut(y, cfg.N_BINS, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=42)
    folds = []
    for _, test_idx in skf.split(df["slide_id"].values, bins):
        folds.append(sorted(test_idx.tolist()))
    return folds


def fold_split(df, cfg, test_idx, n):
    """由测试集索引推出该折的 train/val 划分（确定性，供 run_cv 与校准复用）。"""
    rest = np.setdiff1d(np.arange(n), test_idx)
    rest_bins = pd.qcut(df["weak_fraction"].iloc[rest], 3, labels=False, duplicates="drop")
    fold_idx = [i for i, t in enumerate(make_folds(df, cfg)) if np.array_equal(sorted(t), sorted(test_idx))][0]
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.VAL_RATIO, random_state=fold_idx)
        tr_pos, val_pos = next(sss.split(rest, rest_bins))
    except ValueError:
        from sklearn.model_selection import ShuffleSplit
        tr_pos, val_pos = next(ShuffleSplit(n_splits=1, test_size=cfg.VAL_RATIO,
                                            random_state=fold_idx).split(rest))
    return rest[tr_pos], rest[val_pos]


def train_one_model(train_ids, val_ids, feats, labels, seed, cfg, device, log=print,
                    neighbors=None, prop_graphs=None):
    """单折单种子训练，早停于 val MSE，返回最优状态的模型。

    空间约束（M3，经 cfg 属性开关）：
    - LAMBDA_SMOOTH > 0：叠加网格邻域平滑损失（需 neighbors）；
    - USE_IND=True：叠加 InD 丢弃后的占比损失（SMMILe 双路监督形态）。
    """
    from src.spatial import graph_propagate, ind_dropout_mask, smoothness_loss
    lam = getattr(cfg, "LAMBDA_SMOOTH", 0.0)
    use_ind = getattr(cfg, "USE_IND", False)
    credibility = getattr(cfg, "CREDIBILITY", None)  # M3.5：{slide_id: 损失权重}
    use_prop = getattr(cfg, "USE_PROP", False)       # 实验十三：图标签传播
    prop_alpha = getattr(cfg, "PROP_ALPHA", 0.5)
    amplify_n = getattr(cfg, "AMPLIFY_N", None)      # 实验十三：低占比端放大（D'Amato 五次根）
    record = getattr(cfg, "RECORD_HISTORY", False)   # 训练动力学诊断开关
    history = []

    # 6.3：噪声注入（仅训练标签，val/test 干净；按种子确定性，D'Amato 固定污染协议）
    noise_model = getattr(cfg, "NOISE_MODEL", None)
    noise_level = getattr(cfg, "NOISE_LEVEL", 0.0)
    noisy_labels = None
    if noise_model and noise_level > 0:
        from src.noise import build_noisy_labels
        noisy_labels = build_noisy_labels(train_ids, labels, noise_model, noise_level, seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()

    # 6.2：课程式硬度调度（Duffner & Garcia 2020）——gmean 的 p 随 epoch
    # 从 P_CURRICULUM[0] 线性升至 P_CURRICULUM[1]（先软后硬）
    p_curr = getattr(cfg, "P_CURRICULUM", None)

    best_val, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(cfg.MAX_EPOCHS):
        if p_curr is not None and getattr(model, "pooling", None) == "gmean":
            t = epoch / max(cfg.MAX_EPOCHS - 1, 1)
            model.param = p_curr[0] + (p_curr[1] - p_curr[0]) * t
        model.train()
        order = np.random.permutation(train_ids)
        train_loss_sum, grad_sum = 0.0, {}
        for i in order:
            sid = labels["slide_id"].iloc[i]
            x = feats[sid].to(device)
            y_val = noisy_labels[i] if (noisy_labels is not None and i in noisy_labels) \
                else labels["weak_fraction"].iloc[i]
            y = torch.tensor(y_val, dtype=torch.float32, device=device)
            if amplify_n:
                y = y ** (1.0 / amplify_n)      # 放大技术：目标变换 ŷ = y^(1/n)
            pred, scores = model(x)
            if use_prop and prop_graphs is not None and prop_graphs[sid][2] > 0:
                ei, ew, _ = prop_graphs[sid]
                scores = graph_propagate(scores, ei.to(device), ew.to(device), alpha=prop_alpha)
                k = max(1, int(round(scores.shape[0] * getattr(cfg, "POOL_PARAM", 0.75))))
                pred = scores.topk(k).values.mean()
            mse_raw = loss_fn(pred, y)          # 未加权的纯 MSE，用于可解释的训练曲线
            # 容差式面积约束（实验十二）：LOSS_TYPE="tolerance" 时以 log-barrier 替代 MSE
            if getattr(cfg, "LOSS_TYPE", "mse") == "tolerance":
                z = (pred - y).abs() - getattr(cfg, "TOL_DELTA", 0.05)
                loss = log_barrier(z, t=getattr(cfg, "TOL_T", 5.0))
            else:
                loss = mse_raw
            if credibility is not None:
                loss = loss * credibility.get(sid, 1.0)
            if use_ind:
                keep = ind_dropout_mask(scores)
                loss = loss + loss_fn(scores[keep].mean(), y)
            if lam > 0 and neighbors is not None and neighbors[sid][1] > 0:
                pairs = neighbors[sid][0].to(device)
                loss = loss + lam * smoothness_loss(scores, pairs)
            opt.zero_grad()
            loss.backward()
            if record:
                train_loss_sum += mse_raw.item()
                for name, mod in model.named_modules():
                    if isinstance(mod, torch.nn.Linear) and mod.weight.grad is not None:
                        grad_sum[name] = grad_sum.get(name, 0.0) + mod.weight.grad.norm().item()
            opt.step()

        if record:
            row = {"epoch": epoch + 1,
                   "train_loss": train_loss_sum / len(order),
                   "val_loss": None}
            for name, g in grad_sum.items():
                row[f"grad_{name}"] = g / len(order)
            history.append(row)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in val_ids:
                sid = labels["slide_id"].iloc[i]
                pred, _ = model(feats[sid].to(device))
                val_loss += loss_fn(pred, torch.tensor(
                    labels["weak_fraction"].iloc[i], dtype=torch.float32, device=device)).item()
        val_loss /= max(len(val_ids), 1)
        if record:
            history[-1]["val_loss"] = val_loss

        if epoch + 1 >= cfg.MIN_EPOCHS:
            if val_loss < best_val - 1e-6:
                best_val, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.EARLY_PATIENCE:
                    break
    if best_state is None:  # 未过 MIN_EPOCHS 即收敛时兜底
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    log(f"    seed={seed} 训练结束 @epoch {epoch+1}，best val MSE={best_val:.4f}")
    return model, history


def predict(model, x, device, output_mode="sigmoid", cfg=None, prop_graph=None):
    """返回 (predicted_fraction, patch_scores ndarray)。

    patch_scores 恒为 sigmoid 概率（供热图/mask 使用）；
    linear 模式下 predicted_fraction 为 logits 均值（评估处截断）。
    实验十三：prop_graph 提供时先图传播再聚合；cfg.AMPLIFY_N 提供时逆变换（pred^n）。
    """
    model.eval()
    with torch.no_grad():
        pred, scores = model(x.to(device))
        if prop_graph is not None and prop_graph[2] > 0:
            from src.spatial import graph_propagate
            ei, ew, _ = prop_graph
            scores = graph_propagate(scores, ei.to(device), ew.to(device),
                                     alpha=getattr(cfg, "PROP_ALPHA", 0.5))
            k = max(1, int(round(scores.shape[0] * getattr(cfg, "POOL_PARAM", 0.75))))
            pred = scores.topk(k).values.mean()
        if cfg is not None and getattr(cfg, "AMPLIFY_N", None):
            pred = pred.clamp(min=0) ** cfg.AMPLIFY_N   # 放大技术的逆变换
    scores = scores.detach().cpu().numpy()
    if output_mode == "linear":
        scores = 1.0 / (1.0 + np.exp(-scores))
    return pred.item(), scores


def run_cv(cfg, log=print):
    """5-fold × 多种子主循环。返回 (preds_df, oof_scores, metrics_dict)。

    M3 扩展：cfg.LAMBDA_SMOOTH>0 或 USE_IND 时加载网格邻接；cfg.CKPT_DIR 存在时保存每折权重。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, feats = load_data(cfg)
    folds = make_folds(df, cfg)

    neighbors = None
    if getattr(cfg, "LAMBDA_SMOOTH", 0.0) > 0:
        from src.spatial import build_neighbor_pairs
        neighbors = {}
        for sid in df["slide_id"]:
            coords = pd.read_csv(cfg.COORDS_DIR / f"{sid}.csv")
            pairs = build_neighbor_pairs(coords)
            neighbors[sid] = (pairs, pairs.shape[1])

    # 实验十三：图标签传播的图构建（空间邻近 × 特征相似）
    prop_graphs = None
    if getattr(cfg, "USE_PROP", False):
        from src.spatial import build_propagation_graph
        prop_graphs = {}
        for sid in df["slide_id"]:
            coords = pd.read_csv(cfg.COORDS_DIR / f"{sid}.csv")
            ei, ew = build_propagation_graph(
                feats[sid], coords,
                radius=getattr(cfg, "PROP_RADIUS", 2),
                topk=getattr(cfg, "PROP_TOPK", 8))
            prop_graphs[sid] = (ei, ew, ei.shape[1])
    if cfg.DEBUG_SMOKE:
        folds = folds[:1]
        cfg.SEEDS = cfg.SEEDS[:1]
        cfg.MAX_EPOCHS = 15
        cfg.MIN_EPOCHS = 5
        log("[SMOKE] 调试模式：1 折 × 1 种子 × 15 epoch")

    n = len(df)
    oof_pred = np.full(n, np.nan)
    oof_scores = {}
    histories = []

    for fold_idx, test_idx in enumerate(folds):
        train_ids, val_ids = fold_split(df, cfg, test_idx, n)

        fold_preds, fold_scores = [], []
        for s in cfg.SEEDS:
            model, hist = train_one_model(train_ids, val_ids, feats, df, seed=1000 * fold_idx + s, cfg=cfg,
                                          device=device, log=log, neighbors=neighbors,
                                          prop_graphs=prop_graphs)
            if hist:
                for row in hist:
                    row["fold"] = fold_idx
                    row["seed"] = s
                histories.extend(hist)
            ckpt_dir = getattr(cfg, "CKPT_DIR", None)
            if ckpt_dir is not None:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / f"fold{fold_idx}_seed{s}.pt")
            for i in test_idx:
                sid = df["slide_id"].iloc[i]
                p, sc = predict(model, feats[sid], device, output_mode=cfg.OUTPUT_MODE,
                                cfg=cfg,
                                prop_graph=prop_graphs.get(sid) if prop_graphs else None)
                fold_preds.append((i, p))
                fold_scores.append((sid, sc))
        # 种子集成：同折内取均值
        for i in test_idx:
            sid = df["slide_id"].iloc[i]
            oof_pred[i] = np.mean([p for (ii, p) in fold_preds if ii == i])
            oof_scores[sid] = np.mean([sc for (s2, sc) in fold_scores if s2 == sid], axis=0)
        log(f"  fold {fold_idx} 完成（test={len(test_idx)}）")

    if cfg.OUTPUT_MODE == "linear":
        oof_pred = np.clip(oof_pred, 0.0, 1.0)  # D'Amato 式截断：评估时限定到 [0,1]

    preds_df = pd.DataFrame({
        "slide_id": df["slide_id"],
        "weak_fraction": df["weak_fraction"],
        "predicted_fraction": oof_pred,
    })
    preds_df["absolute_error"] = (preds_df["predicted_fraction"] - preds_df["weak_fraction"]).abs()

    valid = ~np.isnan(preds_df["predicted_fraction"].values)  # SMOKE 模式下仅部分折有预测
    y_true = preds_df["weak_fraction"].values[valid]
    y_pred = preds_df["predicted_fraction"].values[valid]
    metrics = {
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred)[0]),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if getattr(cfg, "RECORD_HISTORY", False):
        return preds_df, oof_scores, metrics, histories
    return preds_df, oof_scores, metrics
