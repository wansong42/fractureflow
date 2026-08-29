# -*- coding: utf-8 -*-
"""Phase 1.1 / 1.2 / 1.3  FORGE FMI LAS 解析 -> 真实多井 3D 法向数据集.

纯 numpy 解析, 禁装 lasio (任务红线). 输出:
  - 每井: MD(ft) -> 真实 3D 井迹 pos (East/North/Up, ft) 通过 HAzi/HDev 积分
  - 每裂隙: 真倾角 Dip_TRU + 真方位 Azimuth(倾向) -> 无向单位法向 nrm
  - 类型归一化: natural / induced, 剔除非裂隙 (Bed_Boundary/Fault/MicroFault/Open Fracture)
  - 存 data/external/utah_forge_fmi/forge_fmi_2wells.pt

用法 (CLI):
  python scripts/read_forge_las.py            # 生成 .pt + 打印摘要
  from scripts.read_forge_las import read_forge_las, build_forge_nets, FORGE_FILES
"""
import os
import re
import json
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(os.path.dirname(_ROOT), "data") if False else _ROOT

FORGE_FILES = {
    "16B": dict(
        path=os.path.join(_ROOT, "data/external/utah_forge_fmi/FORGE_16B_FMI_Reinterpretation.las"),
        md=0, az=2, dip=4, hazi=5, hdev=6,
        type_quoted=True, type_idx=None,
        wellhead=(0.0, 0.0, 0.0),
        surface_label="16B (deviated, HDev~30deg)",
    ),
    "58-32": dict(
        path=os.path.join(_ROOT, "data/external/utah_forge_fmi/FORGE_58_32_FMI_Run1_2226_7550ft.las"),
        md=0, az=2, dip=5, hazi=9, hdev=10,
        type_quoted=False, type_idx=-1,
        wellhead=(2000.0, 0.0, 0.0),  # 两井不同地表位置, 仅用于相对可视化
        surface_label="58-32 (near-vertical, HDev~1deg)",
    ),
}

# 类型归一化规则: 含 Tensile -> induced; Conductive/Resistive -> natural; 其余剔除
_REJECT_HINTS = ("Bed_Boundary", "Fault", "MicroFault", "Open Fracture")


def _norm_type(raw: str):
    if raw is None or raw == "":
        return "natural"   # 无类型信息 -> 默认天然 (保留)
    r = raw.strip().strip('"').strip()
    low = r.lower()
    if any(h.lower() in low for h in _REJECT_HINTS):
        return None  # 非裂隙, 剔除
    if "tensile" in low:
        return "induced"
    if r.startswith("Conductive") or r.startswith("Resistive"):
        return "natural"
    return "natural"  # 未知类型默认保留为天然


def _parse_las(path: str, cfg: dict):
    """返回 dict: md, az, dip, hazi, hdev (ft/deg 数组), ftype(list), raw_type(list).

    关键: 数值列一律按 ~Curve 全量列序的【原位置】索引 (parts[idx]), 绝不先剔除
    字符串列再按新下标取 —— 否则会像初版那样把 58-32 的 HAzi/HDev 取错列.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    ascii_start = None
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("~ASCII"):
            ascii_start = i + 1
            break
    if ascii_start is None:
        raise RuntimeError(f"no ~ASCII in {path}")
    md, az, dip, hazi, hdev = [], [], [], [], []
    ftype, raw_type = [], []
    type_re = re.compile(r'"([^"]*)"')
    for ln in lines[ascii_start:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) <= max(cfg["md"], cfg["az"], cfg["dip"], cfg["hazi"], cfg["hdev"]):
            continue
        # 数值列按全量原位置取 (可能个别含非数字 -> 跳过该行)
        try:
            md_v = float(parts[cfg["md"]])
            az_v = float(parts[cfg["az"]])
            dip_v = float(parts[cfg["dip"]])
            hazi_v = float(parts[cfg["hazi"]])
            hdev_v = float(parts[cfg["hdev"]])
        except ValueError:
            continue
        # 类型抽取
        if cfg.get("type_quoted"):
            m = type_re.search(s)
            rt = m.group(1) if m else None   # 16B: 无引号 -> 天然 (Conductive)
        else:
            last = parts[-1]
            try:
                float(last)
                rt = None                     # 58-32 末列是数字 -> 无显式类型
            except ValueError:
                rt = last
        ft = _norm_type(rt)
        if ft is None:
            continue  # 剔除非裂隙
        md.append(md_v); az.append(az_v); dip.append(dip_v)
        hazi.append(hazi_v); hdev.append(hdev_v)
        ftype.append(ft)
        raw_type.append(rt if rt is not None else "natural")
    return dict(
        md=np.array(md, float),
        az=np.array(az, float),
        dip=np.array(dip, float),
        hazi=np.array(hazi, float),
        hdev=np.array(hdev, float),
        ftype=ftype,
        raw_type=raw_type,
    )


def build_3d_trajectory(md, hazi, hdev, origin=(0.0, 0.0, 0.0)):
    """MD(ft) + 逐点 HAzi(井方位,deg) + HDev(井斜,deg) -> 3D 井迹 (East,North,Up, ft).

    约定 (East, North, Up=正上):
      dE = dMD * sin(HDev) * sin(HAzi)
      dN = dMD * sin(HDev) * cos(HAzi)
      dU = -dMD * cos(HDev)            # 向下为负
    HAzi 自北顺时针, HDev 自铅垂偏离.
    """
    md = np.asarray(md, float)
    hazi = np.deg2rad(np.asarray(hazi, float))
    hdev = np.deg2rad(np.asarray(hdev, float))
    pos = np.zeros((len(md), 3))
    pos[0] = np.array(origin, float)
    dmd = np.diff(md)
    # 中点处井斜用于步长积分
    hmid = (hdev[:-1] + hdev[1:]) / 2.0
    amid = (hazi[:-1] + hazi[1:]) / 2.0
    dE = dmd * np.sin(hmid) * np.sin(amid)
    dN = dmd * np.sin(hmid) * np.cos(amid)
    dU = -dmd * np.cos(hmid)
    inc = np.stack([dE, dN, dU], -1)
    pos[1:] = pos[0] + np.cumsum(inc, axis=0)
    return pos


def dip_dipdir_to_normal(dip_dir, dip):
    """真倾角 dip(deg, 自水平) + 真方位 dip_dir(deg, 自北顺时针, 倾向) -> 单位法向.

    约定 (East, North, Up): 法向水平投影沿 dip_dir, 垂直分量 = cos(dip) (up 正).
    无向 (±n 同面), 符合评测 |cos| 口径.

    几何验证:
      dip=0   (水平面)  -> 法向竖直 (nU=1, 水平分量=0)
      dip=90  (垂直面)  -> 法向水平 (nU=0, 水平分量=1)
      dip=45           -> |nz|=cos45≈0.707
    """
    a = np.deg2rad(np.asarray(dip_dir, float))
    d = np.deg2rad(np.asarray(dip, float))
    nE = np.sin(d) * np.sin(a)
    nN = np.sin(d) * np.cos(a)
    nU = np.cos(d)
    n = np.stack([nE, nN, nU], -1)
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)
    return n


def read_forge_las(which="all"):
    """解析 FORGE 双井, 返回 {wid: parsed_dict}."""
    if which == "all":
        keys = list(FORGE_FILES.keys())
    else:
        keys = [which]
    out = {}
    for k in keys:
        cfg = FORGE_FILES[k]
        p = _parse_las(cfg["path"], cfg)
        p["wid"] = k
        p["surface_label"] = cfg["surface_label"]
        out[k] = p
    return out


def build_forge_nets(parsed):
    """parsed({wid: dict}) -> list[net_dict] 协议格式 (与 loaded_real_nets 一致)."""
    nets = []
    widx = {}
    for i, (wid, p) in enumerate(parsed.items()):
        n = len(p["md"])
        pos = build_3d_trajectory(p["md"], p["hazi"], p["hdev"], FORGE_FILES[wid]["wellhead"])
        nrm = dip_dipdir_to_normal(p["az"], p["dip"])
        ft = np.array(p["ftype"])
        net = dict(
            pos=pos.astype(np.float64),
            nrm=nrm.astype(np.float64),         # 评测用: 全量真值法向
            nrm_full=nrm.astype(np.float64),
            len=np.ones(n, dtype=np.float64),
            lith=np.zeros(n, dtype=np.int64),
            s1=np.ones(n, dtype=np.float64),
            s3=np.ones(n, dtype=np.float64),
            src="forge_fmi",
            wid=wid,
            widx=i,
            md_ft=p["md"].astype(np.float64),
            md_m=(p["md"] * 0.3048).astype(np.float64),
            dip=p["dip"].astype(np.float64),
            dip_dir=p["az"].astype(np.float64),
            ftype=ft,
            raw_type=np.array(p["raw_type"]),
            n=n,
        )
        widx[wid] = i
        nets.append(net)
    return nets, widx


def read_single_las(path: str):
    """通用单井 LAS 解析 (按 ~CURVE 助记符定位列, 不依赖硬编码下标).

    支持任意 FORGE 风格 FMI LAS: 自动找 MD/TDEP, Azimuth(真倾向), Dip_TRU(真倾角),
    HAzi(井方位), HDev(井斜), Type. 返回与 _parse_las 同构的 dict (单井).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    curve = None
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("~CURVE"):
            curve = []
            for j in range(i + 1, len(lines)):
                if lines[j].strip().startswith("~"):
                    break
                p = lines[j]
                m = p.split(".", 1)[0].strip()
                if m and not m.startswith("#"):   # 跳过 ~CURVE 注释行
                    curve.append(m)
            break
    if curve is None:
        raise RuntimeError(f"no ~CURVE in {path}")
    # 去掉重复/空
    def fidx(sub):
        for k, m in enumerate(curve):
            if sub.lower() in m.lower():
                return k
        return None
    # 注意: 不能写 `fidx("MD") or fidx("TDEP")` —— MD 在第 0 列时 0 为 falsy,
    # 会误判为缺失而回退 TDEP/None (通用 LAS 的 MD 常就是第 0 列).
    md_i = fidx("MD")
    if md_i is None:
        md_i = fidx("TDEP")
    az_i = fidx("Azimuth")
    dip_i = fidx("Dip_TRU")
    hazi_i = fidx("HAzi")
    hdev_i = fidx("HDev")
    type_i = fidx("Type")
    # ascii
    ascii_start = next((i for i, ln in enumerate(lines) if ln.strip().upper().startswith("~ASCII")), None)
    if ascii_start is None:
        raise RuntimeError(f"no ~ASCII in {path}")
    type_re = re.compile(r'"([^"]*)"')
    md, az, dip, hazi, hdev, ftype, raw_type = [], [], [], [], [], [], []
    for ln in lines[ascii_start + 1:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        try:
            md_v = float(parts[md_i]) if md_i is not None else 0.0
            az_v = float(parts[az_i])
            dip_v = float(parts[dip_i])
            hazi_v = float(parts[hazi_i]) if hazi_i is not None else 0.0
            hdev_v = float(parts[hdev_i]) if hdev_i is not None else 0.0
        except (IndexError, ValueError, TypeError):
            # T91: TypeError 覆盖缺 Azimuth/Dip 列时 parts[None] 的场景 -> 该行跳过
            continue
        # 类型抽取 (鲁棒): 引号串 -> 末列非数字串 -> Type 助记符列
        rt = None
        m = type_re.search(s)
        if m:
            rt = m.group(1)
        else:
            last = parts[-1]
            try:
                float(last)
            except ValueError:
                rt = last
        if rt is None and type_i is not None and type_i < len(parts):
            try:
                float(parts[type_i])
            except ValueError:
                rt = parts[type_i]
        ft = _norm_type(rt)
        if ft is None:
            continue
        md.append(md_v); az.append(az_v); dip.append(dip_v)
        hazi.append(hazi_v); hdev.append(hdev_v)
        ftype.append(ft); raw_type.append(rt if rt is not None else "natural")
    if not md:
        raise RuntimeError(f"no fracture rows parsed from {path}")
    return dict(
        md=np.array(md, float), az=np.array(az, float), dip=np.array(dip, float),
        hazi=np.array(hazi, float), hdev=np.array(hdev, float),
        ftype=ftype, raw_type=raw_type, wid=os.path.basename(path),
    )


def main():
    parsed = read_forge_las("all")
    summary = {}
    for wid, p in parsed.items():
        summary[wid] = dict(
            n=len(p["md"]),
            md_ft_range=[float(p["md"].min()), float(p["md"].max())],
            dip_median=float(np.median(p["dip"])),
            az_median=float(np.median(p["az"])),
            hdev_median=float(np.median(p["hdev"])),
            type_counts={kk: int(vv) for kk, vv in
                         zip(*np.unique(np.array(p["ftype"]), return_counts=True))},
            raw_type_counts={kk: int(vv) for kk, vv in
                            zip(*np.unique(np.array(p["raw_type"]), return_counts=True))},
        )
        print(f"[{wid}] {p['surface_label']}")
        print(f"    n={summary[wid]['n']}  MD_ft={summary[wid]['md_ft_range']}")
        print(f"    dip_med={summary[wid]['dip_median']:.1f}  az_med={summary[wid]['az_median']:.1f}  hdev_med={summary[wid]['hdev_median']:.1f}")
        print(f"    type={summary[wid]['type_counts']}")
    nets, widx = build_forge_nets(parsed)
    out_path = os.path.join(_ROOT, "data/external/utah_forge_fmi/forge_fmi_2wells.pt")
    torch.save(dict(nets=nets, widx=widx, summary=summary), out_path)
    with open(os.path.join(_ROOT, "data/external/utah_forge_fmi/forge_fmi_2wells_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"\nSaved -> {out_path}")
    print(f"  total fractures = {sum(len(p['md']) for p in parsed.values())}")
    return nets, summary


if __name__ == "__main__":
    main()
