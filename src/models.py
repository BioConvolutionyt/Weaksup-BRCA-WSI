# -*- coding: utf-8 -*-
"""M2/M4 模型：instance 打分头 + 占比回归聚合。

- MeanPoolMIL：文档 5.3 公式（逐 patch sigmoid → 均值）。
- ABMILInstanceMIL：R1 仓 ABMIL_Instance 改造（门控注意力加权实例分数），
  针对 MeanPool 的梯度稀释问题（注意力集中梯度到高信息 patch）。
"""
import math

import torch
from torch import nn


class MeanPoolMIL(nn.Module):
    """输入 (N, in_dim) 的 patch 特征袋，输出 (predicted_fraction, patch_scores)。

    output_mode:
      - "sigmoid"：文档 5.3 公式，逐 patch sigmoid 后取均值（patch 分数 ∈[0,1]，可直接做 mask）
      - "linear"：D'Amato 官方实现形态，logits 直接取均值（评估时截断到 [0,1]）
    """

    def __init__(self, in_dim=1024, hidden=512, dropout=0.25, output_mode="sigmoid"):
        super().__init__()
        assert output_mode in ("sigmoid", "linear")
        self.output_mode = output_mode
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        logits = self.head(self.encoder(x)).squeeze(-1)  # (N,)
        if self.output_mode == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = logits
        return scores.mean(), scores


class GatedAttention(nn.Module):
    """Ilse et al. 2018 门控注意力：tanh(Vh) ⊙ sigmoid(Uh) 门控后线性映射到标量。"""

    def __init__(self, dim=512, hidden=256):
        super().__init__()
        self.lin_v = nn.Linear(dim, hidden)
        self.lin_u = nn.Linear(dim, hidden)
        self.lin_w = nn.Linear(hidden, 1)

    def forward(self, h):                     # h: (N, dim)
        a = self.lin_w(torch.tanh(self.lin_v(h)) * torch.sigmoid(self.lin_u(h)))
        return torch.softmax(a.squeeze(-1), dim=0)   # (N,) 袋内归一


class ABMILInstanceMIL(nn.Module):
    """注意力加权实例分数：pred = Σ a_i · sigmoid(logit_i)（R1 仓 ABMIL_Instance 改造）。

    返回 (predicted_fraction, patch_scores)；patch_scores 恒为 sigmoid 概率供热图。
    attention 权重仅作分析对照（Phase 1 证据：回归中 attention 热图不可信）。
    """

    def __init__(self, in_dim=1024, hidden=512, attn_hidden=256, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = GatedAttention(hidden, attn_hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.encoder(x)                        # (N, hidden)
        a = self.attention(h)                      # (N,)
        scores = torch.sigmoid(self.head(h).squeeze(-1))  # (N,)
        pred = (a * scores).sum()                  # Σa=1，无需除法
        return pred, scores

    def forward_with_attn(self, x):
        """范式对照用：额外返回 attention 权重。"""
        h = self.encoder(x)
        a = self.attention(h)
        scores = torch.sigmoid(self.head(h).squeeze(-1))
        return (a * scores).sum(), scores, a


class PooledMIL(nn.Module):
    """参数化池化打分头（M4 消融：池化硬度/稀释免疫扫描）。

    pooling:
      - "mean"：scores.mean()（参照）
      - "gmean"：广义均值 (mean s^p)^(1/p)，param=p（Duffner 旋钮；logsumexp 数值稳定实现）
      - "quantile"：top-q 分位均值，param=q∈(0,1]（瓶颈 2：对超大切片良性区域稀释免疫）
      - "lse"：LogSumExp (1/τ)·log(mean e^{τs})，param=τ（Ramon & De Raedt 平滑 max）
      - "topk"：top-k 均值，param=k（Chowder 式计数轴）
      - "max"：scores.max()（最硬端点）
    """

    def __init__(self, in_dim=1024, hidden=512, dropout=0.25, pooling="mean", param=1.0):
        super().__init__()
        assert pooling in ("mean", "gmean", "quantile", "lse", "topk", "max")
        self.pooling, self.param = pooling, float(param)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        scores = torch.sigmoid(self.head(self.encoder(x)).squeeze(-1))  # (N,) ∈(0,1)
        n = scores.shape[0]
        if self.pooling == "mean":
            pred = scores.mean()
        elif self.pooling == "gmean":
            p = self.param
            pred = torch.exp((torch.logsumexp(p * torch.log(scores), dim=0)
                              - math.log(n)) / p)   # = (mean s^p)^(1/p)
        elif self.pooling == "lse":
            tau = self.param
            pred = torch.logsumexp(tau * scores, dim=0) / tau - math.log(n) / tau
        elif self.pooling == "topk":
            k = max(1, int(self.param))
            pred = scores.topk(min(k, n)).values.mean()
        elif self.pooling == "max":
            pred = scores.max()
        else:  # quantile
            k = max(1, int(round(n * self.param)))
            pred = scores.topk(k).values.mean()
        return pred, scores


class ChowderMIL(nn.Module):
    """UBC 冠军架构改造：TilesMLP 逐 patch 打分 → top-k+bottom-k → MLP 聚合。

    与本项目差异标注：极端层仅取 2k/万级 patch，面积语义下系统性低估占比——
    定位为聚合器对照，不作主聚合器（Phase 3 KSR 第二部分）。
    """

    def __init__(self, in_dim=1024, tiles_hidden=192, n_top=10, n_bottom=10,
                 mlp_hidden=96, dropout=0.3):
        super().__init__()
        self.n_top, self.n_bottom = n_top, n_bottom
        self.tiles_mlp = nn.Sequential(
            nn.Linear(in_dim, tiles_hidden), nn.ReLU(), nn.Linear(tiles_hidden, 1))
        self.mlp = nn.Sequential(
            nn.Linear(n_top + n_bottom, mlp_hidden), nn.Sigmoid(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1))

    def forward(self, x):
        scores = torch.sigmoid(self.tiles_mlp(x).squeeze(-1))      # (N,)
        n = scores.shape[0]
        top = scores.topk(min(self.n_top, n)).values
        bottom = scores.topk(min(self.n_bottom, n), largest=False).values
        extreme = torch.cat([top, bottom])
        pred = torch.sigmoid(self.mlp(extreme).squeeze())
        return pred, scores


def make_model(cfg):
    """模型工厂：按 cfg.MODEL_TYPE 分发（train_one_model 统一调用）。"""
    mt = getattr(cfg, "MODEL_TYPE", "meanpool")
    if mt == "abmil_inst":
        return ABMILInstanceMIL(cfg.IN_DIM, cfg.HIDDEN, cfg.ATTN_HIDDEN, cfg.DROPOUT)
    if mt == "gmean":
        return PooledMIL(cfg.IN_DIM, cfg.HIDDEN, cfg.DROPOUT, pooling="gmean",
                         param=cfg.POOL_PARAM)
    if mt == "quantile":
        return PooledMIL(cfg.IN_DIM, cfg.HIDDEN, cfg.DROPOUT, pooling="quantile",
                         param=cfg.POOL_PARAM)
    if mt in ("lse", "topk", "max"):
        return PooledMIL(cfg.IN_DIM, cfg.HIDDEN, cfg.DROPOUT, pooling=mt,
                         param=cfg.POOL_PARAM)
    if mt == "chowder":
        return ChowderMIL(cfg.IN_DIM)
    return MeanPoolMIL(cfg.IN_DIM, cfg.HIDDEN, cfg.DROPOUT, output_mode=cfg.OUTPUT_MODE)
