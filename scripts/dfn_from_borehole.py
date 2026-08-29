# -*- coding: utf-8 -*-
"""端到端管线: 钻孔打标 → SetTable → DFN → 渗流 → 报告.

输入: 含 set_ids 的 net 文件 (auto_label_borehole.py / label_free_dirs 输出)
输出: 连通性筛查报告 JSON (含渗流曲线 + 场景指标 + 诚实边界)

用法:
    python scripts/dfn_from_borehole.py --set-table results/x_routeA.pt \
        --domain 20 20 20 --seeds 10 --out results/dfn_report.json

    # npz 输入 (wells x normals, (n_wells, n_fracs, 3))
    python scripts/dfn_from_borehole.py --set-table your_wells.npz \
        --K 4 --domain 50 50 50 --seeds 10

    # 直接指定 SetTable
    python scripts/dfn_from_borehole.py --from-normals normals.npy --K 3
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))


def _fmt_score(k, v):
    """scores 字典条目安全格式化: float 保留 3 位, int/str 元数据原样 (T91)."""
    if isinstance(v, float):
        return f"{k}:{v:.3f}"
    return f"{k}:{v}"

from fractureflow.dfn import (SetTable, generate_dfn, build_connectivity_graph,
                              set_table_from_net, set_table_from_normals,
                              estimate_p32_interval, auto_select_K,
                              fit_beta_from_tracelength, estimate_p32_from_spacing)
from fractureflow.percolation import (percolation_curve, connectivity_anisotropy,
                                      egs_connectivity_metric, mine_risk_sections,
                                      disposal_escape_priority)


def load_source(args):
    """加载数据源, 返回 (SetTable, net_dict_for_p32).

    支持三种输入:
      1. --set-table pt/npz 文件 (含 set_ids)
      2. --from-normals numpy 法向文件 + --K
      3. --inline-csv 钻孔 CSV
    """
    if args.set_table:
        path = args.set_table
        if path.endswith('.npz'):
            data = np.load(path)
            # wells-npz 格式: (n_wells, n_fracs, 3) 法向
            normals = data['arr_0'] if 'arr_0' in data else list(data.values())[0]
            if normals.ndim == 3:
                # (n_wells, n_frac, 3) → (n_wells*n_frac, 3)
                normals = normals.reshape(-1, 3)
            net = {'nrm_full': normals, 'nrm': normals}
            # K 由 auto_select_K 决定 (如果未指定 --K)
            st, set_ids = set_table_from_normals(normals, K=args.K or 4, seed=args.seed)
            net['set_ids'] = set_ids
            net['_set_ids_from_file'] = False
            return st, net
        else:
            # pt 文件 (auto_label_borehole 输出 / loaded_real_nets_setid)
            nets = torch.load(path, weights_only=False)
            net = nets[0] if isinstance(nets, list) else nets
            if 'set_ids' not in net:
                # 没 set_ids → 用球面 k-means 生成
                nrm = np.asarray(net.get('nrm_full', net.get('nrm')))
                st, set_ids = set_table_from_normals(nrm, K=args.K or 4, seed=args.seed)
                net['set_ids'] = set_ids
                net['_set_ids_from_file'] = False
                return st, net
            # 文件自带产品打标 (如 full_pipeline Step1 的 routeA.pt):
            # 标记来源, 防止下游 auto-K 静默覆盖导致交付物自相矛盾
            net['_set_ids_from_file'] = True
            return set_table_from_net(net), net

    elif args.from_normals:
        nrm = np.load(args.from_normals)
        net = {'nrm_full': nrm, 'nrm': nrm}
        st, set_ids = set_table_from_normals(nrm, K=args.K or 4, seed=args.seed)
        net['set_ids'] = set_ids
        net['_set_ids_from_file'] = False
        return st, net

    else:
        raise ValueError("需要 --set-table 或 --from-normals 输入")


def main():
    ap = argparse.ArgumentParser(description='钻孔→SetTable→DFN→渗流→报告 端到端管线')
    ap.add_argument('--set-table', default=None,
                    help='含 set_ids 的 net 文件 (.pt/.npz)')
    ap.add_argument('--from-normals', default=None,
                    help='法向数组 .npy (N,3)')
    ap.add_argument('--K', type=int, default=None,
                    help='组数 (默认 4)')
    ap.add_argument('--domain', nargs=3, type=float, default=[20, 20, 20],
                    help='域尺寸 Lx Ly Lz (m), 默认 20 20 20')
    ap.add_argument('--beta', type=float, default=3.5,
                    help='幂律指数 β (默认 3.5)')
    ap.add_argument('--beta-range', nargs='+', type=float, default=None,
                    help='β 扫描范围 (如 3.0 3.5 4.0)')
    ap.add_argument('--seeds', type=int, default=20,
                    help='每 P32 点实现数 (默认 20, H6 合规)')
    ap.add_argument('--p32-fixed', type=float, default=None,
                    help='固定 P32 (不扫描)')
    ap.add_argument('--well-axis', nargs=3, type=float, default=[1, 0, 0],
                    help='井对/巷道轴向 (默认 [1,0,0])')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='results/dfn_pipeline_report.json',
                    help='输出 JSON 路径')
    ap.add_argument('--report', action='store_true', default=True,
                    help='同时生成 Markdown 报告 (默认 True)')
    ap.add_argument('--no-report', dest='report', action='store_false',
                    help='不生成 Markdown 报告')
    ap.add_argument('--site-name', default='场地',
                    help='场地名称 (用于报告标题)')
    # Phase D: 数据升档接口
    ap.add_argument('--trace-lengths', default=None,
                    help='迹长文件 .npy (N,) → MLE 拟合 β (替代默认 β=3.5)')
    ap.add_argument('--spacing', default=None,
                    help='间距文件 .npy (N,) → 直接算 P32 (替代计数估计)')
    ap.add_argument('--trace-lengths-col', type=int, default=None,
                    help='迹长在有表头的 CSV 中的列索引')
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("  钻孔 → SetTable → DFN → 渗流 → 报告  (v3 端到端管线)")
    print("=" * 60)

    # 1. 加载数据 → SetTable
    print(f"\n[1/6] 加载数据源...")
    st, net = load_source(args)
    nrm = np.asarray(net.get('nrm_full', net.get('nrm')))

    # 1.5 Auto K (仅当输入未自带打标时; 文件自带 set_ids 必须沿用,
    #     否则 full_pipeline Step1 的产品档位打标会被静默替换成另一套聚类)
    labels_from_file = bool(net.pop('_set_ids_from_file', False))
    if args.K is None and not labels_from_file:
        print(f"\n[1.5/6] 自动选择 K (spherical silhouette)...")
        best_K, scores = auto_select_K(nrm, Krange=(2, 8), seed=args.seed)
        # T91 演练修复: scores 混有 int/str 元数据键 (K_silhouette/fusion_rule 等),
        # 旧写法 f'{v:.3f}' 对 str 值直接 ValueError, 干净数据也会崩 (L5 规则: dict
        # 混合类型值禁止盲用 :.Nf 格式化, 见 tests/test_t91_drill_fixes.py).
        score_strs = ', '.join(_fmt_score(k, v) for k, v in scores.items())
        print(f"  K 选择: {best_K} (scores: {score_strs})")
        # 用 best_K 重新生成 SetTable
        st, set_ids = set_table_from_normals(nrm, K=best_K, seed=args.seed)
        net['set_ids'] = set_ids
    elif args.K is None and labels_from_file:
        print(f"\n[1.5/6] 输入文件自带 set_ids → 沿用 (K={st.K}), 不重新聚类")

    print(f"  SetTable: K={st.K}, N_frac={len(nrm)}")
    for k in range(st.K):
        print(f"    组 {k}: dir={st.centers[k].round(2)}, "
              f"κ={st.concentrations[k]:.1f}, prop={st.proportions[k]:.2f}")

    # 2. P32 区间估计 (两阶段: 粗网格 → 细网格)
    print(f"\n[2/6] P32 区间估计 (编录表数据 → 量级)...")
    domain = tuple(args.domain)
    p32_info = estimate_p32_interval(net, domain)
    print(f"  P32 估计: p10={p32_info['p32_p10']:.3f}, "
          f"p50={p32_info['p32_p50']:.3f}, p90={p32_info['p32_p90']:.3f}")
    print(f"  注: {p32_info['assumptions'][:60]}...")

    # 2.5 Phase D: 数据升档 (迹长→β, 间距→P32)
    beta_fitted = None
    beta_fit = None
    p32_spacing = None
    if args.trace_lengths is not None:
        print(f"\n[2.5/6] Phase D: 迹长 → β 拟合...")
        tl = np.load(args.trace_lengths)
        beta_fit = fit_beta_from_tracelength(tl)
        beta_fitted = beta_fit['beta']
        print(f"  β 拟合: {beta_fitted:.2f} ± {beta_fit['beta_std']:.2f} "
              f"(n={beta_fit['n_samples']}, method={beta_fit['method']})")
        # 更新 β
        if args.beta_range is None:
            args.beta = beta_fitted

    if args.spacing is not None:
        print(f"\n[2.5/6] Phase D: 间距 → P32 直接估计...")
        sp = np.load(args.spacing)
        net['spacing'] = sp
        p32_spacing = estimate_p32_from_spacing(net, domain)
        if p32_spacing.get('p32') is not None:
            print(f"  P32 (spacing): {p32_spacing['p32']:.4f} "
                  f"(method: {p32_spacing['method']})")
            # 用间距估计覆盖计数估计
            p32_info['p32_p50'] = p32_spacing['p32']
            p32_info['p32_p10'] = p32_spacing['p32'] * 0.3
            p32_info['p32_p90'] = p32_spacing['p32'] * 3.0
            p32_info['spacing_info'] = p32_spacing

    # 3. 渗流曲线 (两阶段: 粗网格定位 → 细网格精确, 性能优化)
    print(f"\n[3/6] 渗流曲线...")
    if args.p32_fixed is not None:
        p32_grid = np.array([args.p32_fixed])
    else:
        # 用 P32 估计的区间 + 外延
        p50 = p32_info['p32_p50']
        p32_grid = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5])
        # 如果 p50 远大于默认网格, 调整
        if p50 > 2.0:
            p32_grid = np.linspace(0.1, p50 * 2, 8)
        # 限制最大 P32: 防裂隙数过多 (N ≈ P32 * V / mean_area, 上限 ~15k)
        V = domain[0] * domain[1] * domain[2]
        mean_area = np.pi * (0.5 * 5.0)  # 几何平均近似
        p32_max = min(5.0, 15000 * mean_area / V)  # 提高上限到 5.0 (大域适用)
        p32_grid = p32_grid[p32_grid <= p32_max]
        if len(p32_grid) < 3:
            p32_grid = np.linspace(0.05, p32_max, 6)

    beta_list = args.beta_range if args.beta_range else [args.beta]
    perc_results = {}
    grid_used = {}   # 每个 β 实际使用的 P32 网格 (两阶段时=细网格)
    for beta in beta_list:
        # 两阶段: 先用粗网格 + 少实现定位, 再在阈值附近加密
        if args.p32_fixed is None and len(p32_grid) > 4:
            # 阶段 A: 粗网格 (每点 5 实现, 快速定位)
            perc_coarse = percolation_curve(st, p32_grid, beta=beta, domain=domain,
                                            seeds=range(5), pbc=False)
            crit_coarse = perc_coarse['p32_crit']
            # 阶段 B: 在粗阈值附近加密 (每点全 seeds)
            fine_grid = np.array([crit_coarse * 0.7, crit_coarse * 0.85,
                                  crit_coarse, crit_coarse * 1.15, crit_coarse * 1.3])
            fine_grid = fine_grid[fine_grid > 0]
            fine_grid = np.unique(np.round(fine_grid, 3))
            perc = percolation_curve(st, fine_grid, beta=beta, domain=domain,
                                     seeds=range(args.seeds), pbc=False)
            grid_used[beta] = fine_grid
        else:
            perc = percolation_curve(st, p32_grid, beta=beta, domain=domain,
                                     seeds=range(args.seeds), pbc=False)
            grid_used[beta] = p32_grid
        perc_results[beta] = perc
        print(f"  β={beta:.1f}: p32_crit={perc['p32_crit']:.2f} "
              f"[{perc['p32_crit_lower']:.2f}, {perc['p32_crit_upper']:.2f}]")

    # 选取主结果 (第一个 β)
    beta_main = beta_list[0]
    perc_main = perc_results[beta_main]

    # 4. 场景指标 (在 p32_crit 处生成一个实现计算)
    print(f"\n[4/6] 场景指标...")
    dfn = generate_dfn(st, p32=perc_main['p32_crit'], beta=beta_main,
                       domain=domain, seed=args.seed)
    well_axis = np.array(args.well_axis, float)
    well_axis /= np.linalg.norm(well_axis) + 1e-12
    egs = egs_connectivity_metric(dfn, well_axis)
    mine = mine_risk_sections(dfn, well_axis)
    disp = disposal_escape_priority(dfn)
    print(f"  EGS 连通分数: {egs['connectivity_fraction']:.3f}")
    print(f"  矿井风险: {mine['risk_level']}")
    print(f"  处置逃逸优先级: {disp['escape_priority']}")

    # 5. 汇总输出
    print(f"\n[5/6] 落盘报告...")
    out = {
        'meta': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline': 'dfn_from_borehole v1.0',
            'elapsed_sec': round(time.time() - t0, 1),
        },
        'input': {
            'N_fractures': int(len(nrm)),
            'K': st.K,
            'domain_m': list(domain),
        },
        'set_table': {
            'K': st.K,
            'centers': st.centers.tolist(),
            'concentrations': st.concentrations.tolist(),
            'proportions': st.proportions.tolist(),
        },
        'p32_estimate': p32_info,
        'phase_d_upgrade': {
            'beta_fitted': beta_fitted,
            'beta_fit_info': beta_fit if args.trace_lengths else None,
            'spacing_p32': p32_spacing.get('p32') if args.spacing else None,
            'spacing_info': p32_spacing if args.spacing else None,
        },
        'percolation': {
            'beta': beta_main,
            'beta_sweep': {f'β={b:.1f}': {
                'p32_crit': r['p32_crit'],
                'p32_crit_lower': r['p32_crit_lower'],
                'p32_crit_upper': r['p32_crit_upper'],
            } for b, r in perc_results.items()},
            'p32_grid': grid_used[beta_main].tolist(),   # 与 p_conn 同网格 (修复: 曾存粗网格导致横纵错位)
            'p32_grid_coarse': p32_grid.tolist(),        # 两阶段定位用的粗网格 (仅参考)
            'p_conn': perc_main['p_conn'].tolist() if hasattr(perc_main['p_conn'], 'tolist') else list(perc_main['p_conn']),
            'p32_crit': perc_main['p32_crit'],
            'p32_crit_lower': perc_main['p32_crit_lower'],
            'p32_crit_upper': perc_main['p32_crit_upper'],
            'assumptions': perc_main['assumptions'],
            'n_realizations_per_point': args.seeds,
        },
        'scenario_metrics': {
            'egs': {k: v for k, v in egs.items() if k != 'dominant_direction'},
            'mine': {k: v for k, v in mine.items() if k != 'dominant_direction'},
            'disposal': {k: v for k, v in disp.items() if k != 'dominant_direction'},
        },
        'honest_boundary': {
            'screening_level': True,
            'not_deterministic': True,
            'no_aperture': '开度未知 → 无绝对渗透量',
            'no_real_connectivity': '筛查级结论, 非真实连通性判定',
            'size_distribution_unknown': f'β={beta_main:.2f} 为假设, 需露头/岩心标定',
            'data_needed_for_tighter_band': '延伸长/迹长 → β 可拟合, P32 带收窄',
        },
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)
    print(f"\n  ✓ JSON 落盘: {args.out}")
    print(f"  耗时: {out['meta']['elapsed_sec']:.1f}s")

    # 生成 Markdown 报告
    if args.report:
        from fractureflow.report import generate_report
        report_path = os.path.splitext(args.out)[0] + "_report.md"
        md = generate_report(out, site_name=args.site_name)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"  ✓ 报告落盘: {report_path}")

    print(f"\n  ⚠ 诚实边界: 本报告是筛查级结论 + 不确定性区间,")
    print(f"    不是连通性判定。尺寸分布未知 → P32_crit 为区间。")

    return out


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError) as e:
        if os.environ.get("FRACTUREFLOW_DEBUG"):
            raise
        print(f"\n[错误] {type(e).__name__}: {e}", file=sys.stderr)
        print("[提示] 请检查数据文件路径/列名/空值; 完整调试栈: 设置环境变量 "
              "FRACTUREFLOW_DEBUG=1 后重跑。", file=sys.stderr)
        sys.exit(2)
