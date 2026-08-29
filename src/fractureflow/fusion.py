# -*- coding: utf-8 -*-
"""融合推理 (meta_v2 软权重): 多候选估计 + 模型选择器 → 最终预测。

候选集 (与 models/meta_v2.pt names 严格对齐):
  maj3..maj6 (majority_dirs), sem4b0.1 / sem6b0.5 (semantic_dirs), prop, nn
特征: [aniso, log1p(最近观测距离), purity16, kappa(prop 置信), log1p(L)]
权重: softmax( head(tanh(proj(h)), X) / tau ), tau=2.0 (test 泛化最优)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .inference import majority_dirs, semantic_dirs

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

CAND_ORDER = ("maj3", "maj4", "maj5", "maj6",
              "sem4b0.1", "sem6b0.5", "prop", "nn")


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class MetaFusion:
    """融合推理。proj/head 缺失时退化为 majority K=4。"""

    def __init__(self, device=None, h_dim=256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proj = None
        self.head = None
        self.names = CAND_ORDER
        self.temper = 2.0
        path = os.path.join(MODEL_DIR, "meta_v2.pt")
        if os.path.exists(path):
            ck = torch.load(path, map_location=self.device, weights_only=False)
            self.names = tuple(ck["names"])
            self.proj = nn.Linear(h_dim, 16)
            self.head = nn.Sequential(nn.Linear(16 + 5, 64), nn.Tanh(),
                                      nn.Linear(64, len(self.names)))
            self.proj.load_state_dict(ck["proj"])
            self.head.load_state_dict(ck["head"])
            self.proj.to(self.device).eval()
            self.head.to(self.device).eval()

    # ---- 候选 ----
    def candidates(self, pos, nrm, mask, prop_dir=None):
        occ = mask.astype(bool)
        nrm = np.asarray(nrm, dtype=np.float64)
        out = {}
        for K in (3, 4, 5, 6):
            pd, _ = majority_dirs(pos, nrm, occ, K=K, knn=16, min_frac=0.6)
            out[f"maj{K}"] = pd.astype(np.float64)
        pd, _ = semantic_dirs(pos, nrm, occ, 4, 0.1)
        out["sem4b0.1"] = pd.astype(np.float64)
        pd, _ = semantic_dirs(pos, nrm, occ, 6, 0.5)
        out["sem6b0.5"] = pd.astype(np.float64)
        d2 = ((pos[~occ][:, None] - pos[occ][None]) ** 2).sum(-1)
        nn_dir = np.zeros_like(nrm)
        if len(d2):
            nn_dir[~occ] = unit(nrm)[occ][d2.argmin(1)]
        out["nn"] = nn_dir
        out["prop"] = np.zeros_like(nrm)
        if prop_dir is not None:
            out["prop"][~occ] = unit(np.asarray(prop_dir))[~occ]
        return out

    # ---- 特征 ----
    def features(self, pos, mask, aniso, kappa, L):
        occ = mask.astype(bool)
        n_h = int((~occ).sum())
        d2 = ((pos[~occ][:, None] - pos[occ][None]) ** 2).sum(-1)
        if len(d2) == 0:
            return np.zeros((0, 5), dtype=np.float32), d2
        if d2.shape[1] >= 16:
            kk_all = np.argsort(((pos[~occ][:, None] - pos[None]) ** 2).sum(-1), 1)
            purity = mask[kk_all[:, :16]].mean(1)
        else:
            purity = np.full(n_h, 0.0)
        f = np.stack([aniso[~occ],
                      np.log1p(np.sqrt(d2.min(1))),
                      purity,
                      kappa[~occ],
                      np.full(n_h, np.log1p(L))], 1)
        return f.astype(np.float32), d2

    def predict(self, pos, nrm, mask, aniso, kappa, h=None, prop_dir=None, L=None):
        pos = np.asarray(pos, dtype=np.float64)
        nrm_u = unit(np.asarray(nrm, dtype=np.float64))
        occ = mask.astype(bool)
        L = L or len(pos)
        if occ.sum() == 0:
            return np.repeat(np.zeros(3)[None], len(pos), axis=0).astype(np.float32)
        if occ.sum() < 4:
            v = unit(nrm_u[occ].mean(0))
            return np.repeat(v[None], len(pos), axis=0).astype(np.float32)

        cands = self.candidates(pos, nrm, mask, prop_dir)
        if self.proj is None or (h is None and prop_dir is None):
            dirs = cands["maj4"][~occ]
        else:
            harr = None if h is None else np.asarray(h, dtype=np.float32)[~occ]
            X, d2 = self.features(pos, mask, aniso, kappa, L)
            Xv = torch.tensor(X, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                if harr is not None:
                    hv = torch.tensor(harr, device=self.device)
                    hp = F.tanh(self.proj(hv))
                    w = F.softmax(self.head(torch.cat([hp, Xv], 1)) / self.temper,
                                  dim=1).cpu().numpy()
                else:
                    w = np.ones((X.shape[0], len(self.names))) / len(self.names)
            acc = np.zeros((int((~occ).sum()), 3))
            for i, c in enumerate(self.names):
                acc += w[:, i][:, None] * cands[c][~occ]
            dirs = unit(acc)
        out = np.zeros_like(nrm)
        out[occ] = nrm_u[occ]
        out[~occ] = dirs
        return out.astype(np.float32)