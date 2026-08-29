# -*- coding: utf-8 -*-
"""Åland 露头迹线 -> 模型协议适配器 (路线 A / B 可直接消费)。

数据形态: 每个 20m 露头窗口 = 数千条迹线 (LineString), 每条迹线是一个裂隙,
带 Id (fracture_id 同类) 与可计算的走向。azimuth_sets.json 给出 3 个真实方位组。

关键物理约束: 露头迹线只给走向 (strike), 不给倾角(dip)。本适配器默认假设近垂直
倾角 (dip=90) 还原水平法向 n=[sin(az+90), cos(az+90), 0]。这是诚实但保守的近似:
- 若客户裂隙近直立, 此近似几乎无偏;
- 若需真实 3D 法向, 须补露头面产状(DEM/正射)或显式 dip 字段。

输出协议与项目其余解码器一致: pos[L,3], nrm[L,3], set_ids[L], fracture_id[L],
obs_mask[L]。可直接喂 inference.set_aware_dirs / connectivity.fracture_aware_dirs /
inference.l1_local_dirs。
"""
import json
import os
import glob
import math

import numpy as np

DEFAULT_TRACES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'external', 'aland_islands_repo', 'data', 'trace_data', 'traces', '20m')

# 走向方位区间 (azimuth_sets.json) 与中点
SET_RANGES = [(155, 25), (25, 75), (85, 135)]
SET_CENTERS = [170.0, 50.0, 110.0]


def _strike_of(coords):
    c = np.asarray(coords, float)
    if c.shape[0] < 2 or c.shape[1] < 2:
        return None
    d = c[-1] - c[0]
    if np.linalg.norm(d[:2]) < 1e-9 and c.shape[0] > 2:
        # 折线长度方向 (首末退化时用质心到末端)
        d = c[-1] - c[c.shape[0] // 2]
    az = math.degrees(math.atan2(d[0], d[1])) % 180
    return az


def _normal_from_strike(az, vertical_dip=True):
    if vertical_dip:
        rad = math.radians(az + 90.0)
        return np.array([math.sin(rad), math.cos(rad), 0.0], float)
    # 非垂直倾角需 dip 输入 (预留)
    raise NotImplementedError('non-vertical dip requires dip field (not in trace data)')


def _assign_set(az):
    best, bi = 1e9, -1
    for i, c in enumerate(SET_CENTERS):
        dd = min(abs(az - c), 180 - abs(az - c))
        if dd < best:
            best, bi = dd, i
    return bi


def load_trace_window(geojson_path, vertical_dip=True):
    """解析单个迹线窗口 geojson -> net dict (协议一致)。"""
    g = json.load(open(geojson_path, encoding='utf-8'))
    feats = g.get('features', [])
    pos, nrm, set_ids, fids, azs = [], [], [], [], []
    for idx, f in enumerate(feats):
        c = f.get('geometry', {}).get('coordinates')
        if not isinstance(c, list) or len(c) < 2:
            continue
        az = _strike_of(c)
        if az is None:
            continue
        cc = np.asarray(c, float)
        centroid = cc[:, :2].mean(0)
        nrm_v = _normal_from_strike(az, vertical_dip)
        sid = _assign_set(az)
        p = f.get('properties', {}) or {}
        fid = p.get('Id', None)
        fid = int(fid) if isinstance(fid, (int, float)) else idx
        pos.append([centroid[0], centroid[1], 0.0])
        nrm.append(nrm_v)
        set_ids.append(sid)
        fids.append(fid)
        azs.append(az)
    if not pos:
        return None
    return dict(
        pos=np.asarray(pos, float),
        nrm=np.asarray(nrm, float),
        set_ids=np.asarray(set_ids, int),
        fracture_id=np.asarray(fids, int),
        az=np.asarray(azs, float),
        name=os.path.basename(geojson_path),
        L=len(pos),
    )


def aland_networks(traces_dir=DEFAULT_TRACES_DIR, vertical_dip=True):
    """返回所有 20m 窗口的 net dict 列表。"""
    fs = sorted(glob.glob(os.path.join(traces_dir, '*.geojson')))
    nets = []
    for fp in fs:
        net = load_trace_window(fp, vertical_dip)
        if net is not None and net['L'] >= 20:
            nets.append(net)
    return nets


def make_obs_mask(L, seed=999, obs_frac=0.4):
    rng = np.random.default_rng(seed)
    return rng.random(L) < obs_frac


def eval_net(net, k=8, seed=999, obs_frac=0.4):
    """诚实评测单窗口: 隐伏迹线靠空间最近观测投票定组 (Route A 诚实版)。
    返回 dict(mean, median, p90, oracle_mean, L, n_hid)。
    """
    from .inference import _unit, _sign_align
    pos, nrm, sid_true = net['pos'], net['nrm'], net['set_ids']
    L = net['L']
    occ = make_obs_mask(L, seed, obs_frac)
    hid = ~occ
    if hid.sum() == 0 or occ.sum() == 0:
        return None
    sid_obs = sid_true[occ]
    obs_idx = np.where(occ)[0]

    def predict(vote_from_obs):
        pred = np.zeros((L, 3))
        pred[occ] = nrm[occ]
        for i in np.where(hid)[0]:
            dxy = np.linalg.norm(pos[obs_idx] - pos[i], axis=1)
            js = np.argsort(dxy)[:k]
            vote = np.bincount(sid_obs[js]).argmax()
            sel = sid_obs == vote
            if sel.sum() == 0:
                sel = np.ones(sel.shape, bool)
            ns = nrm[obs_idx][sel]
            s = np.sign((ns * ns.mean(0)).sum(-1, keepdims=True))
            s[s == 0] = 1
            pred[i] = (ns * s).mean(0)
        return pred

    pred = predict(True)
    err = _acos_err(pred[hid], nrm[hid])
    # oracle: 隐伏用其真实组
    por = np.zeros((L, 3)); por[occ] = nrm[occ]
    for i in np.where(hid)[0]:
        sel = sid_obs == sid_true[i]
        if sel.sum() == 0:
            sel = np.ones(sel.shape, bool)
        ns = nrm[obs_idx][sel]; s = np.sign((ns * ns.mean(0)).sum(-1, keepdims=True)); s[s == 0] = 1
        por[i] = (ns * s).mean(0)
    err_or = _acos_err(por[hid], nrm[hid])
    return dict(mean=float(err.mean()), median=float(np.median(err)),
                p90=float(np.percentile(err, 90)), oracle_mean=float(err_or.mean()),
                L=L, n_hid=int(hid.sum()))


def _acos_err(pred, true):
    pred = pred / np.linalg.norm(pred, axis=-1, keepdims=True)
    true = true / np.linalg.norm(true, axis=-1, keepdims=True)
    c = np.clip(np.abs((pred * true).sum(-1)), 0, 1)
    return np.degrees(np.arccos(c))
