# -*- coding: utf-8 -*-
"""B3-2 / B3-5 真项修复回归测试 + B3-4 grep 门禁冒烟.

覆盖:
  - R4: beta=inf 必须显式 ValueError (非静默退化)
  - R7: joint_set_ids_grid 极小输入不得 None 解引用崩
  - R8: prepare_net 幂等 (len/log_len 链路, 已被 test_invariants_geometry 覆盖, 此处重复保底)
  - B3-5: 毒丸硬门禁 (命中 -> poison_rejected, save_result 拒绝, load_baseline 拒绝)
  - B3-4: check_geometry_conventions.py 全代码库 0 违规
"""
import os
import sys
import json
import subprocess
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from fractureflow.dfn import SetTable, generate_dfn
from fractureflow import setlabel as sl
from fractureflow import data as dt
from fractureflow.honest_eval import (
    evaluate, make_blind, DirectionPredictor, PoisonRejectedError,
    save_result, load_baseline,
)


# ---------------------------------------------------------------------------
# R4: beta=inf 守卫
# ---------------------------------------------------------------------------
def test_beta_inf_guard():
    st = SetTable(centers=np.array([[0.0, 0.0, 1.0]]),
                  concentrations=np.array([1.0]),
                  proportions=np.array([1.0]))
    for bad in (float("inf"), float("-inf"), float("nan")):
        try:
            generate_dfn(st, p32=0.5, beta=bad, domain=(5, 5, 5), seed=1)
            raise AssertionError(f"beta={bad} 未触发守卫")
        except (ValueError, Exception) as e:
            # 仅接受显式 ValueError; 其他异常也视为未静默通过
            if not isinstance(e, ValueError):
                raise AssertionError(f"beta={bad} 抛非预期异常类型: {type(e).__name__}")
    # 有限 beta 仍正常
    r = generate_dfn(st, p32=0.5, beta=3.0, domain=(5, 5, 5), seed=1)
    assert r.n_fractures > 0


# ---------------------------------------------------------------------------
# R7: joint_set_ids_grid 极小输入
# ---------------------------------------------------------------------------
def test_joint_set_ids_grid_tiny():
    tiny = [np.array([[0.0, 0.0, 1.0]])]   # 单井单点
    try:
        best_K, assign, centers, scores = sl.joint_set_ids_grid(tiny, Krange=(2, 7), seed=0)
    except Exception as e:
        raise AssertionError(f"joint_set_ids_grid 极小输入崩: {type(e).__name__}: {e}")
    assert isinstance(scores, dict)
    assert len(assign) == 1
    assert centers.shape[1] == 3


# ---------------------------------------------------------------------------
# R8: prepare_net 幂等 (保底, 主覆盖在 test_invariants_geometry)
# ---------------------------------------------------------------------------
def test_prepare_net_idempotent_b3():
    import torch
    rng = np.random.default_rng(11)
    L = 8
    net = {
        "pos": torch.tensor(rng.normal(size=(L, 3)), dtype=torch.float32),
        "nrm": torch.tensor(rng.normal(size=(L, 3)), dtype=torch.float32),
        "len": torch.tensor(rng.uniform(0.1, 10.0, size=(L, 1)), dtype=torch.float32),
        "lith": torch.tensor(np.zeros(L, dtype=int)),
        "s1": torch.tensor(rng.normal(size=3), dtype=torch.float32),
        "s3": torch.tensor(rng.normal(size=3), dtype=torch.float32),
    }
    a = dt.prepare_net(net, 0.4, np.random.default_rng(1))
    b = dt.prepare_net(a, 0.4, np.random.default_rng(1))
    assert a["lens"].allclose(b["lens"])
    assert a["log_len"].allclose(b["log_len"])


# ---------------------------------------------------------------------------
# B3-5: 毒丸硬门禁
# ---------------------------------------------------------------------------
class _LeakyProbe:
    """模拟泄漏预测器: 隐伏点 NaN 时用 obs 均值填充, 但毒丸替换后透传随机向量 -> 命中测试."""
    name = "leaky_probe"

    def predict(self, x):
        out = np.zeros((x.pos.shape[0], 3))
        out[x.occ] = x.nrm_blind[x.occ]
        hid = ~x.occ
        if np.any(np.isnan(x.nrm_blind[hid])):
            out[hid] = 0.0                      # x: 隐伏=NaN -> 填 0
        else:
            out[hid] = x.nrm_blind[hid]         # x_prime: 隐伏=随机 -> 透传
        return out


def _make_nets(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return [{
        "pos": rng.normal(size=(n, 3)).astype(np.float64),
        "nrm": rng.normal(size=(n, 3)).astype(np.float64),
    }]


def test_poison_hard_gate_blocks_leak():
    pred = _LeakyProbe()
    nets = _make_nets()
    res = evaluate(pred, nets, seeds=range(2), obs_frac=0.5, run_poison=True)
    assert res.get("poison_rejected") is True, f"泄漏器未被拦截: {res}"
    assert res["poison_leaks_detected"] > 0

    # save_result 默认拒绝
    tmp = tempfile.mkdtemp()
    try:
        save_result(res, "demo", out_dir=tmp)
        raise AssertionError("save_result 未拒绝毒丸结果")
    except PoisonRejectedError:
        pass
    # force=True 仅诊断存档 (文件名带 .poison)
    p = save_result(res, "demo", out_dir=tmp, force=True)
    assert ".poison" in p
    # load_baseline 拒绝读取毒丸文件
    try:
        load_baseline(p)
        raise AssertionError("load_baseline 未拒绝毒丸文件")
    except PoisonRejectedError:
        pass


def test_poison_gate_allows_clean():
    from fractureflow.honest_eval import _L1LocalPredictor
    pred = _L1LocalPredictor()
    nets = _make_nets()
    res = evaluate(pred, nets, seeds=range(2), obs_frac=0.5, run_poison=True)
    assert res.get("poison_rejected") is False
    assert res["poison_leaks_detected"] == 0
    tmp = tempfile.mkdtemp()
    p = save_result(res, "demo", out_dir=tmp)   # 不应抛
    assert os.path.exists(p)
    mae = load_baseline(p)
    assert isinstance(mae, float)


# ---------------------------------------------------------------------------
# B3-4: grep 门禁冒烟
# ---------------------------------------------------------------------------
def test_geometry_convention_grep_gate():
    script = os.path.join(_ROOT, "scripts", "check_geometry_conventions.py")
    assert os.path.exists(script), "grep 门禁脚本缺失"
    proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"grep 门禁发现违规:\n{proc.stdout}\n{proc.stderr}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
