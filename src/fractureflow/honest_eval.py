# -*- coding: utf-8 -*-
"""诚实评测 Harness —— 全项目唯一评测入口 (铁律 2)。

设计原则 (三次泄漏事故后的工程化防线):
1. 物理隐藏, 不靠约定 (铁律 1): BlindInput.nrm_blind 隐伏点已置 NaN,
   预测器试图消费隐伏法向 → 计算产 NaN → 断言崩溃。
2. 预测器协议签名锁死: DirectionPredictor.predict(BlindInput) -> np.ndarray,
   无法额外传入全量 nrm / set_ids (F4)。
3. 泄漏探测器自动运行: 毒丸测试 + NaN 传播断言, 每次评测必跑。
4. 红旗自动审计: R1 超越检查 / R2 循环检查 / R3 形状检查。

用法:
    from fractureflow.honest_eval import evaluate, make_blind, DirectionPredictor
    result = evaluate(my_predictor, nets, seeds=range(10), obs_frac=0.4)
    # result = {mae_mean, mae_std, p50, p90, n_seeds, flags, per_well, ...}

参照: docs/技术路线_诚实评测与组系表产品化_v1.md §A1。
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Protocol, Optional
from scipy.optimize import linear_sum_assignment


class PoisonRejectedError(RuntimeError):
    """命中毒丸测试的方法被硬拦截: 不得进 leaderboard, 不得被 load_baseline 读取."""

# ---------------------------------------------------------------------------
# 1. 盲化输入构造
# ---------------------------------------------------------------------------

@dataclass
class BlindInput:
    """预测器唯一能看到的输入。

    nrm_blind: 隐伏点法向已物理置 NaN (不是"约定不用", 是"拿不到")。
    occ: True = 观测点, False = 隐伏点。
    """
    wid: str
    pos: np.ndarray          # (L, 3) 全量位置 (位置对双方可见, 合法)
    nrm_blind: np.ndarray    # (L, 3) 隐伏点已置 NaN 的法向
    occ: np.ndarray          # (L,) bool 观测掩码
    meta: dict = field(default_factory=dict)  # seed, obs_frac, 来源数据集名


def make_blind(net: dict, mask_seed: int, obs_frac: float = 0.4) -> BlindInput:
    """全项目唯一掩码生成点。

    掩码规则: rng = np.random.default_rng(mask_seed); occ = rng.random(L) < obs_frac.
    隐伏点法向物理置 NaN 后返回。

    net dict 至少需要: pos (L,3), nrm (L,3)。
    """
    pos = np.asarray(net["pos"], dtype=np.float64)
    pos = pos - pos.mean(0, keepdims=True)  # 中心化 (与 prepare_net 一致)
    nrm = np.asarray(net.get("nrm_full", net["nrm"]), dtype=np.float64)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)

    L = pos.shape[0]
    rng = np.random.default_rng(mask_seed)
    occ = rng.random(L) < obs_frac

    nrm_blind = nrm.copy()
    nrm_blind[~occ] = np.nan  # 物理置 NaN

    wid = str(net.get("wid", ""))
    widx = int(net.get("widx", 0))
    if wid:
        wid_key = f"{wid}#{widx}"
    else:
        wid_key = f"net_{widx}"
    return BlindInput(
        wid=wid_key,
        pos=pos,
        nrm_blind=nrm_blind,
        occ=occ,
        meta={"mask_seed": mask_seed, "obs_frac": obs_frac,
              "src": str(net.get("src", "")),
              "wid": wid, "widx": widx,
              "n_total": L, "n_obs": int(occ.sum())},
    )


# ---------------------------------------------------------------------------
# 2. 预测器协议
# ---------------------------------------------------------------------------

class DirectionPredictor(Protocol):
    """预测器协议 —— 签名锁死, 违反 F4 即编译期可查。

    预测器**只能**访问 BlindInput 的字段。不允许额外传全量 nrm、set_ids、组心。
    """
    name: str

    def predict(self, x: BlindInput) -> np.ndarray:  # (L, 3), 隐伏点各行必须为单位向量
        ...


# ---------------------------------------------------------------------------
# 3. 泄漏探测器
# ---------------------------------------------------------------------------

def _propagation_assert(dirs: np.ndarray, x: BlindInput) -> None:
    """NaN 传播断言: 预测结果含 NaN → 试图消费隐伏法向 → 崩溃。"""
    if np.any(np.isnan(dirs)):
        nan_rows = np.where(np.any(np.isnan(dirs), axis=1))[0]
        raise AssertionError(
            f"[NaN 传播断言] 预测器 {len(nan_rows)} 行输出 NaN "
            f"(indices={nan_rows[:5].tolist()}...), 疑似消费隐伏法向。"
        )


def poison_test(predictor: DirectionPredictor, x: BlindInput,
                truth: np.ndarray, n_trials: int = 3) -> bool:
    """毒丸测试: 构造 x', 把隐伏点位置上的 NaN 替换为随机单位向量 (其余不变),
    若 predictor(x) 与 predictor(x') 的预测不同 → 用了真值 → 判泄漏。

    原理: 泄漏方法内部会检测 NaN 位置并用外部真值填充。毒丸测试把 NaN 换成
    随机向量后, 泄漏方法"以为"是真值填入, 实际用了垃圾 → 预测改变。
    合法方法只用观测法向 (非 NaN 部分), 不受影响。

    返回 True = 泄漏 (危险), False = 干净。
    """
    rng = np.random.default_rng(12345)
    pred_orig = predictor.predict(x)
    _propagation_assert(pred_orig, x)

    for _ in range(n_trials):
        x_prime = BlindInput(
            wid=x.wid,
            pos=x.pos.copy(),
            nrm_blind=x.nrm_blind.copy(),
            occ=x.occ.copy(),
            meta=x.meta.copy(),
        )
        # 把隐伏点的 NaN 替换为随机单位向量
        hid = ~x.occ
        if hid.sum() == 0:
            return False
        random_vecs = rng.normal(size=(int(hid.sum()), 3))
        random_vecs /= np.linalg.norm(random_vecs, axis=1, keepdims=True) + 1e-12
        x_prime.nrm_blind[hid] = random_vecs

        pred_poison = predictor.predict(x_prime)
        _propagation_assert(pred_poison, x_prime)

        # 比较隐伏点预测是否一致
        diff = np.linalg.norm(pred_orig[hid] - pred_poison[hid], axis=1)
        if np.any(diff > 1e-3):
            return True  # 泄漏!

    return False


# ---------------------------------------------------------------------------
# 4. 红旗审计
# ---------------------------------------------------------------------------

def _check_red_flags(result: dict, baseline_mae: Optional[float] = None) -> list:
    """红旗自动审计。命中任一条 → 结果 JSON 打 FLAGGED。"""
    flags = []

    # R1 超越检查: 可部署方法优于 l1_local 基线超过 2°
    if baseline_mae is not None and not result.get("is_oracle", False):
        if result["mae_mean"] < baseline_mae - 2.0:
            flags.append(
                f"R1-SURPASS: MAE {result['mae_mean']:.2f}° 优于基线 {baseline_mae:.2f}° "
                f"超 2° (差 {baseline_mae - result['mae_mean']:.2f}°)"
            )

    # R2 循环检查: 方法与"地板"差距 < 0.3°
    floor = result.get("floor_mae")
    if floor is not None and not result.get("is_oracle", False):
        if abs(result["mae_mean"] - floor) < 0.3:
            flags.append(
                f"R2-CIRCULAR: MAE {result['mae_mean']:.2f}° 与地板 {floor:.2f}° "
                f"差距 < 0.3° (疑似同源)"
            )

    # R3 形状检查: K 网格曲线单调下降无膝点
    k_curve = result.get("k_curve")
    if k_curve is not None and len(k_curve) >= 4:
        maes = [v["mae"] for k, v in sorted(k_curve.items())]
        # 检查是否单调下降 (允许微小波动 < 0.2°)
        decreasing = all(maes[i] - maes[i + 1] > -0.2 for i in range(len(maes) - 1))
        if decreasing and (maes[0] - maes[-1]) > 1.0:
            flags.append(
                f"R3-MONOTONE: K 网格单调下降无膝点 ({maes[0]:.2f}→{maes[-1]:.2f}), "
                f"疑似泄漏指纹"
            )

    return flags


# ---------------------------------------------------------------------------
# 5. 参照预测器 (内置锚点)
# ---------------------------------------------------------------------------

class _L1LocalPredictor:
    """l1_local 基线 (可部署锚, 当前 SOTA)。"""
    name = "l1_local"

    def predict(self, x: BlindInput) -> np.ndarray:
        from .inference import l1_local_dirs
        # 把 NaN 替换为 0 以符合 l1_local_dirs 接口 (它只看 occ, 不用隐伏法向)
        nrm_safe = np.where(np.isnan(x.nrm_blind), 0.0, x.nrm_blind)
        dirs, _ = l1_local_dirs(x.pos, nrm_safe, x.occ)
        return dirs


class _Top2EnsPredictor:
    """top2_ens 几何基线。"""
    name = "top2_ens"

    def predict(self, x: BlindInput) -> np.ndarray:
        from .inference import top2_ens_dirs
        nrm_safe = np.where(np.isnan(x.nrm_blind), 0.0, x.nrm_blind)
        dirs, _ = top2_ens_dirs(x.pos, nrm_safe, x.occ)
        return dirs


class _GlobalMeanPredictor:
    """全局主方向 (下界锚)。"""
    name = "global_mean"

    def predict(self, x: BlindInput) -> np.ndarray:
        nrm_safe = np.where(np.isnan(x.nrm_blind), 0.0, x.nrm_blind)
        T = nrm_safe[x.occ].T @ nrm_safe[x.occ]
        w, v = np.linalg.eigh(T)
        g = v[:, -1]
        g = g / (np.linalg.norm(g) + 1e-12)
        dirs = np.zeros((x.pos.shape[0], 3))
        dirs[x.occ] = nrm_safe[x.occ]
        dirs[~x.occ] = g
        return dirs


class _ObsGlobalFrechetPredictor:
    """观测全局 Fréchet 中位数 (合法方法, 用于毒丸测试零误报验证)。"""
    name = "obs_global_frechet"

    def predict(self, x: BlindInput) -> np.ndarray:
        from .inference import _l1_median_batch
        nrm_safe = np.where(np.isnan(x.nrm_blind), 0.0, x.nrm_blind)
        N = nrm_safe[x.occ]
        if len(N) < 2:
            return np.zeros((x.pos.shape[0], 3))
        S = N.T @ N
        v0 = np.linalg.eigh(S)[1][:, -1][None]
        v = _l1_median_batch(N, np.ones((1, len(N))), v0, iters=40)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        dirs = np.zeros((x.pos.shape[0], 3))
        dirs[x.occ] = N
        dirs[~x.occ] = v[0]
        return dirs


class _TruthAssignOraclePredictor:
    """真值指派 oracle (诊断锚, 强制打 ORACLE-ONLY 标记)。

    仅用于内部误差分解, 禁止出现在任何面向客户的材料。
    """
    name = "truth_assign_oracle"
    is_oracle = True

    def __init__(self, K: int = 6):
        self.K = K

    def predict(self, x: BlindInput) -> np.ndarray:
        """用隐伏点真值法向指派到最近组心 —— 这是泄漏的, 仅作诊断。"""
        # 注意: 这个预测器需要全量真值, 只能在特殊模式下运行
        # 实际实现中, 真值从外部注入 (evaluate 的 truth_map)
        raise RuntimeError("Oracle 预测器只能在 evaluate(..., truth_map=...) 模式下运行")


# ---------------------------------------------------------------------------
# 6. 评测主函数
# ---------------------------------------------------------------------------

def _ang_err(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """逐点角误差 (度)。"""
    cos = np.clip(np.abs((pred * true).sum(-1)), 0, 1)
    return np.degrees(np.arccos(cos))


def evaluate(
    predictor: DirectionPredictor,
    nets: list,
    seeds: range = range(10),
    obs_frac: float = 0.4,
    baseline_mae: Optional[float] = None,
    run_poison: bool = True,
    truth_map: Optional[dict] = None,
) -> dict:
    """评测主函数。

    返回 dict:
        mae_mean, mae_std (跨 seed), p50, p90, n_seeds, n_wells,
        flags: list, per_well: dict, is_oracle: bool
    落盘: results/honest_leaderboard/<predictor.name>__<dataset>.json

    误差口径: 隐伏点 acos(|<pred,true>|) 均值, 对每 seed 算井均值再对 seed 求均值/标准差。
    """
    seed_maes = []
    all_errs = []
    per_well_errs = {}  # wid -> list of MAE per seed
    poison_leaks = []

    for s in seeds:
        rng = np.random.default_rng(999 + 1000 * s)
        seed_well_maes = []

        for net in nets:
            x = make_blind(net, mask_seed=999 + 1000 * s, obs_frac=obs_frac)
            if x.occ.sum() < 2 or (~x.occ).sum() == 0:
                continue

            # 获取真值 (用于评测)
            truth = np.asarray(net.get("nrm_full", net["nrm"]), dtype=np.float64)
            truth = truth / (np.linalg.norm(truth, axis=1, keepdims=True) + 1e-12)

            # Oracle 预测器特殊处理
            if getattr(predictor, "is_oracle", False) and truth_map is not None:
                dirs = _oracle_predict(predictor, x, truth)
            else:
                dirs = predictor.predict(x)

            # NaN 传播断言
            _propagation_assert(dirs, x)

            # 毒丸测试 (仅对非 oracle 预测器)
            if run_poison and not getattr(predictor, "is_oracle", False):
                is_leak = poison_test(predictor, x, truth, n_trials=1)
                if is_leak:
                    poison_leaks.append(x.wid)

            # 计算隐伏点误差
            hid = ~x.occ
            err = _ang_err(dirs[hid], truth[hid])
            seed_well_maes.append(float(err.mean()))
            all_errs.extend(err.tolist())
            per_well_errs.setdefault(x.wid, []).append(float(err.mean()))

        if seed_well_maes:
            seed_maes.append(float(np.mean(seed_well_maes)))

    if not seed_maes:
        return {"error": "no valid seeds", "predictor": predictor.name}

    seed_maes = np.array(seed_maes)
    all_errs_arr = np.array(all_errs)

    result = {
        "predictor": predictor.name,
        "mae_mean": round(float(seed_maes.mean()), 4),
        "mae_std": round(float(seed_maes.std(ddof=1)), 4) if len(seed_maes) > 1 else 0.0,
        "p50": round(float(np.median(all_errs_arr)), 4),
        "p90": round(float(np.percentile(all_errs_arr, 90)), 4),
        "n_seeds": len(seed_maes),
        "n_wells": len(per_well_errs),
        "n_hidden_total": int(all_errs_arr.size),
        "is_oracle": getattr(predictor, "is_oracle", False),
        "poison_leaks_detected": len(poison_leaks),
        "poison_leak_wells": poison_leaks[:10],
        "per_well_mean": {k: round(float(np.mean(v)), 3) for k, v in per_well_errs.items()},
    }

    # 红旗审计
    result["flags"] = _check_red_flags(result, baseline_mae)

    # B3-5 毒丸硬门禁: 命中毒丸测试 -> 标记拒绝, 不得进 leaderboard / 不得被 load_baseline 读取
    if poison_leaks:
        result["poison_rejected"] = True
        result["flags"] = result["flags"] + [(
            f"POISON-REJECTED: 命中毒丸测试 (泄露真值), 拒绝进 leaderboard 且 "
            f"load_baseline 将拒绝读取. 命中井: {poison_leaks[:10]}")]
    else:
        result["poison_rejected"] = False

    return result


def _oracle_predict(predictor: DirectionPredictor, x: BlindInput,
                    truth: np.ndarray) -> np.ndarray:
    """Oracle 预测器: 用真值法向做组指派 (仅诊断)。"""
    from .setlabel import spherical_kmeans
    K = predictor.K
    nrm_occ = truth[x.occ]
    if len(nrm_occ) < K:
        K = len(nrm_occ)
    if K < 2:
        v = truth.mean(0)
        v = v / (np.linalg.norm(v) + 1e-12)
        return np.repeat(v[None], x.pos.shape[0], axis=0)

    cents, _ = spherical_kmeans(nrm_occ, K, seed=42)
    # 用真值法向指派 (泄漏!)
    assign = np.abs(truth @ cents.T).argmax(1)
    dirs = np.zeros((x.pos.shape[0], 3))
    dirs[x.occ] = truth[x.occ]
    dirs[~x.occ] = cents[assign[~x.occ]]
    return dirs


# ---------------------------------------------------------------------------
# 7. 组系表评测 (Phase B 入口) —— 委托 set_table_eval 单一实现
# ---------------------------------------------------------------------------
# 注意: 此处曾有一份重复的 match_tables/set_table_score 实现, 其 coverage
# 未按文档实现 "<30° 才算匹配成功" 阈值 (所有匈牙利配对全计入), 与
# set_table_eval.set_table_score 语义不一致, 属陷阱代码, 已删除并委托.
from .set_table_eval import (  # noqa: E402,F401  (向后兼容再导出)
    match_tables,
    set_table_score,
    evaluate_set_table,
)


# ---------------------------------------------------------------------------
# 8. 便捷入口
# ---------------------------------------------------------------------------

def save_result(result: dict, dataset_name: str, out_dir: str = "results/honest_leaderboard",
                force: bool = False) -> str:
    """保存评测结果到 JSON。

    B3-5 毒丸硬门禁: 命中毒丸的结果 (poison_rejected=True) 默认拒绝落盘 leaderboard,
    除非显式 force=True (仅用于诊断存档, 文件名带 .poison 后缀以示区分)。
    """
    if result.get("poison_rejected") and not force:
        raise PoisonRejectedError(
            f"预测器 {result.get('predictor')} 命中毒丸测试, 拒绝写入 leaderboard "
            f"(如需诊断存档请用 force=True)")
    os.makedirs(out_dir, exist_ok=True)
    predictor_name = result.get("predictor", "unknown")
    suffix = ".poison" if result.get("poison_rejected") else ""
    path = os.path.join(out_dir, f"{predictor_name}__{dataset_name}{suffix}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    return path


def load_baseline(path: str = "results/honest_leaderboard/l1_local__beishan_22.json") -> Optional[float]:
    """加载 l1_local 基线 MAE (用于 R1 红旗)。

    B3-5: 若文件标记为中毒丸泄漏 (poison_rejected / poison_leaks_detected>0),
    拒绝作为合法基线读取, 抛 PoisonRejectedError。
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if data.get("poison_rejected") or data.get("poison_leaks_detected", 0) > 0:
        raise PoisonRejectedError(
            f"基线文件 {path} 被标记为毒丸泄漏, 拒绝作为合法基线读取")
    return float(data.get("mae_mean", 0))
