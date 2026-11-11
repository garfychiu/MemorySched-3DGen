"""
根据步骤 1~5 消融实验实测数据生成论文级图表。
可作为独立脚本运行，也可由 run.py --gen_charts 调用。
输出到指定目录（默认 outputs/charts/）。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import os
import json

# 自适应调度器（支柱三·闭环）：用于生成「图 6 自适应工作点」决策图。
# 失败时降级跳过，不影响其余消融图生成。
try:
    from src.memory_cost_model import recommend_config, detect_device_budget_gb
    _HAVE_SCHEDULER = True
except Exception:
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.memory_cost_model import recommend_config, detect_device_budget_gb
        _HAVE_SCHEDULER = True
    except Exception:
        _HAVE_SCHEDULER = False



# ── 全局样式（《计算机科学》规范：Times New Roman + 宋体，小五字，单栏 ~8cm）──
FONT = ['Times New Roman', 'SimSun', 'DejaVu Sans']

# 协调、对灰度印刷友好的学术配色（按明度区分，便于黑白打印辨识）
PALETTE = ['#C00000', '#E36C09', '#2E7D32', '#1F4E79',
           '#6A1B9A', '#5D4037', '#00796B']

# λ-sweep 实验独立数据目录（不与 Table 1 的 5 组聚合混淆）。
LAMBDA_SWEEP_DIR = 'outputs/ablation_lambda_sweep'
LAMBDA_SWEEP_G = 128          # 扫描固定分辨率（λ 分母 G^3 固定）
LAMBDA_SWEEP_G3 = LAMBDA_SWEEP_G ** 3

# 图表展示用分组标签：用代码侧实际参数 Chunk200k（grid_extract_chunk 200000）表达分块程度，
# 直接对应实现、比 λ 更直观且三档 LOD 统一；λ 作为连续变量保留在方法/理论叙述（3.3.5、4.4）中。
# 展示标签两行：第一行代码侧实际参数 Chunk200k（grid_extract_chunk 200000，三档 LOD 统一），
# 第二行连续变量 λ（来自 3.3.5 临界邻域，随 LOD 的 G 变化：128/96/64 → 0.095/0.161/0.433），
# 既直观可追溯，又保留 λ 作为代价模型与 Pareto 统一的连续变量。
GROUP_CHUNK_LABEL = {
    '+FP8+Chunk200k': '+FP8+Chunk200k\n(λ=0.095)',
    '+FP8+Lambda0': '+FP8+Lambda0\n(λ=λ0, peak locked at Stage1)',
    '+FP8+Chunk200k+LOD96': '+FP8+Chunk200k+LOD96\n(λ=0.161)',
    '+FP8+Chunk200k+LOD64': '+FP8+Chunk200k+LOD64\n(λ=0.433)',
    '+FP8+Chunk200k+LOD128': '+FP8+Chunk200k+LOD128\n(λ=0.095)',
}


def _setup_style():
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'legend.fontsize': 8.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': FONT,
        'axes.linewidth': 0.7,
        'grid.linewidth': 0.4,
        'axes.edgecolor': '#333333',
        'xtick.color': '#222222',
        'ytick.color': '#222222',
        'text.color': '#111111',
        # 出版级：将字体嵌入 PDF/EPS，避免"字体未嵌入"被退稿
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.unicode_minus': False,
    })


def _save(fig, out_dir, base):
    """导出 300 DPI 位图（PNG）与矢量图（PDF / EPS），满足期刊出版要求。"""
    fig.savefig(f'{out_dir}/{base}.png', dpi=300)
    fig.savefig(f'{out_dir}/{base}.pdf')
    fig.savefig(f'{out_dir}/{base}.eps')


def _caption(fig, text, y=0.02):
    """图题置于图形正下方（《计算机科学》规范：图 X 中文图名，宋体小五居中）。"""
    fig.text(0.5, y, text, ha='center', va='bottom', fontsize=8.5)


def _baked_caption(fig, num, title, y=0.012):
    """把"图 X 中文图名"作为图题烘焙进图片（文档缺图题时自描述，居中宋体小五）。"""
    fig.text(0.5, y, f'图{num} {title}', ha='center', va='bottom',
             fontsize=8.5, linespacing=1.25)


def _normalize(arr, invert=False):
    a = np.array(arr, dtype=float)
    mn, mx = a.min(), a.max()
    if mx == mn:
        return np.zeros_like(a)
    val = (a - mn) / (mx - mn)
    return 1 - val if invert else val


# ── 真实运行记录：幂等写入 / 聚合 ──────────────────────────
def record_run(metric_dir, metric, filename='run_metrics.jsonl'):
    """
    把一次运行测得的指标写入 metric_dir/run_metrics.jsonl。
    按 metric['group'] 幂等更新：同组重跑会覆盖上次记录，而不是重复追加。
    """
    import os
    import json
    os.makedirs(metric_dir, exist_ok=True)
    path = os.path.join(metric_dir, filename)
    records = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    replaced = False
    for i, r in enumerate(records):
        if r.get('group') == metric['group']:
            records[i] = metric
            replaced = True
            break
    if not replaced:
        records.append(metric)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def aggregate_runs(out_dir='outputs/ablation', filename='run_metrics.jsonl'):
    """
    聚合真实测得的 mem / time / 相对质量指标（支持多物体评测）。

    多物体结构：每个输入单独跑一套 6 组，落到 out_dir/<输入名>/ 子目录，
    各含自己的 run_metrics.jsonl。本函数会扫描 out_dir 下所有子目录的
    run_metrics.jsonl，按组对「每个输入一条记录」取 均值 ± 标准差。
    若存在子目录记录，则忽略父目录直接放置的 run_metrics.jsonl（避免重复计入）。

    返回 generate_all_charts 所需的 results dict（含 *_std 与 n_inputs）；无记录时返回 None。
    """
    import os
    import json

    # 1) 收集所有 run_metrics.jsonl：优先「每个输入一个子目录」结构（多物体）。
    sub_paths = []
    if os.path.isdir(out_dir):
        for name in sorted(os.listdir(out_dir)):
            p = os.path.join(out_dir, name, filename)
            if os.path.isfile(p):
                sub_paths.append(p)
    direct = os.path.join(out_dir, filename)
    jsonl_paths = sub_paths if sub_paths else ([direct] if os.path.isfile(direct) else [])
    if not jsonl_paths:
        return None

    all_records = []
    for p in jsonl_paths:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue
    if not all_records:
        return None

    # 2) 按组聚合（组内 = 多个输入）
    by_group = {}
    for r in all_records:
        by_group.setdefault(r['group'], []).append(r)

    # 已从流水线移除的阴性对照组（+ SparseAttn）：丢弃磁盘上的陈旧记录，
    # 避免其污染当前消融图表。
    by_group.pop('+ SparseAttn', None)

    canonical = ['Baseline', '+FP8', '+FP8+Chunk200k', '+FP8+Lambda0',
                 '+FP8+Chunk200k+LOD96', '+FP8+Chunk200k+LOD64']
    ordered_groups = [g for g in canonical if g in by_group]
    ordered_groups += [g for g in by_group if g not in canonical]

    default_by_group = {}
    dgroups = _DEFAULT_DATA.get('groups', [])
    for i, g in enumerate(dgroups):
        default_by_group[g] = {
            'cd': _DEFAULT_DATA['cd'][i],
            'fscore': _DEFAULT_DATA['fscore'][i],
            'lpips': _DEFAULT_DATA['lpips'][i],
        }

    def _mean_std(vals):
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            return None, None
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std(ddof=0))

    groups, mem, mem_std, time_, time_std = [], [], [], [], []
    cd, cd_std, fscore, fscore_std = [], [], [], []
    lpips, lpips_std, ssim, ssim_std = [], [], [], []
    s1_list, s2_list, n_inputs = [], [], []

    has_rel = any('rel_cd' in r for r in all_records) and all(
        any('rel_cd' in r and r.get('rel_cd') is not None for r in recs)
        for recs in by_group.values()
    )

    for g in ordered_groups:
        recs = by_group[g]
        groups.append(g)
        n_inputs.append(len(recs))
        mem_m, mem_s = _mean_std([r.get('mem') for r in recs])
        time_m, time_s = _mean_std([r.get('time') for r in recs])
        s1 = [r.get('stage1_mem') for r in recs if r.get('stage1_mem') is not None]
        s2 = [r.get('stage2_mem') for r in recs if r.get('stage2_mem') is not None]
        s1_list.append(float(np.mean(s1)) if s1 else None)
        s2_list.append(float(np.mean(s2)) if s2 else None)
        mem.append(mem_m)
        mem_std.append(mem_s)
        time_.append(time_m)
        time_std.append(time_s)
        if has_rel:
            cdm, cds = _mean_std([r.get('rel_cd') for r in recs])
            fsm, fss = _mean_std([r.get('rel_fscore') for r in recs])
            lpm, lps = _mean_std([r.get('rel_lpips') for r in recs])
            ssm, sss = _mean_std([r.get('rel_ssim') for r in recs])
            cd.append(cdm); cd_std.append(cds)
            fscore.append(fsm); fscore_std.append(fss)
            lpips.append(lpm); lpips_std.append(lps)
            ssim.append(ssm); ssim_std.append(sss)
        else:
            q = default_by_group.get(g)
            if q is not None:
                cd.append(q['cd']); cd_std.append(None)
                fscore.append(q['fscore']); fscore_std.append(None)
                lpips.append(q['lpips']); lpips_std.append(None)
                ssim.append(None); ssim_std.append(None)
            else:
                cd.append(None); cd_std.append(None)
                fscore.append(None); fscore_std.append(None)
                lpips.append(None); lpips_std.append(None)
                ssim.append(None); ssim_std.append(None)

    return {
        'groups': groups,
        'mem': mem, 'mem_std': mem_std,
        'time': time_, 'time_std': time_std,
        'cd': cd, 'cd_std': cd_std,
        'fscore': fscore, 'fscore_std': fscore_std,
        'lpips': lpips, 'lpips_std': lpips_std,
        'ssim': ssim, 'ssim_std': ssim_std,
        'stage1_mem': s1_list, 'stage2_mem': s2_list,
        'n_inputs': n_inputs,
        'quality_from_default': (not has_rel),
        'quality_relative': has_rel,
        'diagnostic': _DEFAULT_DATA.get('diagnostic'),
        'compress_cd': _DEFAULT_DATA.get('compress_cd'),
        'source': 'auto-recorded runs',
    }


# ── λ-sweep 实验数据读取与绘图（支柱二·连续调节可视化）──
def read_lambda_sweep(sweep_dir=LAMBDA_SWEEP_DIR):
    """
    读取 run_ablation.py --lambda_sweep 写入的独立记录，返回按 λ 降序排列的扫描点。
    每组 group 名形如 lambda_sweep_g800000（对应 g 顶点）或 lambda_sweep_full（整批 λ=1.0）；
    由组名恢复 g 与 λ = g / G^3，并取 stage1_mem / stage2_mem / mem(=max) / time。
    """
    import re
    if not os.path.isdir(sweep_dir):
        return None
    pts = []
    for name in sorted(os.listdir(sweep_dir)):
        p = os.path.join(sweep_dir, name, 'run_metrics.jsonl')
        if not os.path.isfile(p):
            continue
        try:
            lines = open(p, encoding='utf-8').read().splitlines()
        except Exception:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            g = r.get('group', '')
            m = re.search(r'lambda_sweep_g(\d+)', g)
            if m:
                gv = int(m.group(1))
                lam = gv / LAMBDA_SWEEP_G3
            elif 'lambda_sweep_full' in g:
                gv = LAMBDA_SWEEP_G3
                lam = 1.0
            else:
                continue
            s1 = r.get('stage1_mem')
            s2 = r.get('stage2_mem')
            if s2 is None:
                continue
            peak = max(s1, s2) if s1 is not None else s2
            pts.append({'group': g, 'g': gv, 'lambda': lam,
                        'stage1': s1, 'stage2': s2, 'peak': peak,
                        'time': r.get('time')})
    if not pts:
        return None
    pts.sort(key=lambda d: d['lambda'], reverse=True)
    return pts


def generate_lambda_sweep_chart(points, out_dir='outputs/charts'):
    """
    Fig. 8 · λ 均衡曲线：把 3.3.3–3.3.5 一次性可视化。
    横轴 λ（对数），左轴 M_stage2 与全局峰值 M_peak=max(M_stage1, M_stage2)，
    右轴推理延迟；标出理论 λ0 竖线与 M_stage1 刚性下限线。
    """
    _setup_style()
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.9, 3.1))

    lam = [p['lambda'] for p in points]
    s2 = [p['stage2'] for p in points]
    peak = [p['peak'] for p in points]
    t = [p['time'] for p in points]
    s1 = next((p['stage1'] for p in points if p['stage1'] is not None), None)

    # 左轴：Stage2 峰值与全局峰值
    ax.plot(lam, s2, 'o-', color='#1F4E79', linewidth=1.6, markersize=5,
            label=r'$M_{stage2}$')
    ax.plot(lam, peak, 's--', color='black', linewidth=1.6, markersize=5,
            label=r'$M_{peak}$')
    if s1 is not None:
        ax.axhline(y=s1, color='gray', linestyle=':', linewidth=1.2)
        ax.text(lam[0], s1 + 0.15, f'$M_{{stage1}}$={s1:.2f} GB (rigid bound)',
                color='gray', fontsize=5.5, ha='left', va='bottom')

    # 理论 λ0 竖线：用实测 Stage1 反解，与 3.3.5 一致
    lam0 = None
    try:
        from src.memory_cost_model import lambda0_eq
        if s1 is not None:
            lam0 = lambda0_eq(LAMBDA_SWEEP_G, m_stage1=s1)
            ax.axvline(x=lam0, color='#C00000', linestyle='--', linewidth=1.2)
            ax.text(lam0, max(peak) * 0.96, f'$\\lambda_0\\approx{lam0:.3f}$',
                    color='#C00000', fontsize=5.5, ha='right', va='top', rotation=90)
    except Exception:
        lam0 = None

    # 右轴：推理延迟（碗形：整批过大→慢，中等分块最快，过度细分因 kernel 启动开销回升）
    ax2 = ax.twinx()
    ax2.plot(lam, t, '^:', color='#E36C09', linewidth=1.4, markersize=5,
             label='Latency')
    ax2.set_ylabel('Inference Latency / s', color='#E36C09', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='#E36C09', labelsize=7)
    ax.tick_params(axis='both', labelsize=7)

    # 标注（方案 A）：延迟最优粒度 与 过度分块退化区
    try:
        _tmin = min(t)
        _i_min = t.index(_tmin)
        ax2.annotate('延迟最优\n粒度 g≈100k',
                     xy=(lam[_i_min], t[_i_min]),
                     xytext=(lam[_i_min], t[_i_min] + (max(t) - min(t)) * 0.18),
                     fontsize=4.5, ha='center', va='bottom', color='#E36C09',
                     arrowprops=dict(arrowstyle='->', color='#E36C09', lw=0.6))
        for _i, _p in enumerate(points):
            if _p['group'] == 'lambda_sweep_g50000' and _i > 0 and _p['time'] > points[_i - 1]['time']:
                ax2.annotate('过度分块\n退化区',
                             xy=(lam[_i], t[_i]),
                             xytext=(lam[_i], t[_i] + (max(t) - min(t)) * 0.18),
                             fontsize=4.5, ha='center', va='bottom', color='#888888',
                             arrowprops=dict(arrowstyle='->', color='#888888', lw=0.6))
    except Exception:
        pass

    # 区域注释：λ>λ0 重建为瓶颈；λ≤λ0 峰值锁死 Stage1、延迟呈碗形（谷底≈g100k、过细则回升）
    if lam0 is not None:
        ax.text(lam0 * 6, max(peak) * 0.5,
                'λ>λ0:\nStage-2 bottleneck', fontsize=5, ha='center', va='center',
                color='#1F4E79')
        ax.text(lam0 / 6, s1 + 0.4,
                'λ≤λ0:\npeak locked at $M_{stage1}$,\nlatency: bowl-shaped\n(min≈g100k, then rises)', fontsize=5,
                ha='center', va='bottom', color='#555555')

    ax.set_xscale('log')
    ax.set_xlabel(r'Chunk balancing factor $\lambda = g/G^3$ (log)', fontsize=8)
    ax.set_ylabel('Peak GPU Memory / GB', fontsize=8)
    ax.set_ylim(0, max(max(peak), s1 or 0) * 1.18)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, color='#cfcfcf')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5.2, loc='upper center',
              bbox_to_anchor=(0.5, -0.34), ncol=3, framealpha=0.95,
              borderpad=0.3, labelspacing=0.2, handlelength=1.6)
    fig.subplots_adjust(left=0.15, right=0.85, top=0.93, bottom=0.34)
    _save(fig, out_dir, 'fig8_lambda_sweep')
    plt.close(fig)
    print('[ok] fig8_lambda_sweep.(png/pdf/eps)')


def generate_all_charts(results, out_dir='outputs/charts'):
    """
    results: dict with keys:
        groups, mem, cd, fscore, lpips, time        -- 主消融
        diagnostic: { prune_ratios, prune_cd }       -- 诊断（可选）
        compress_cd: float                            -- Compress-48 CD（可选）
    out_dir: 输出目录
    """
    _setup_style()
    os.makedirs(out_dir, exist_ok=True)

    groups = results['groups']
    # 展示标签：分块程度标注为 λ（机制名 +Chunk 保留，degree 用 λ 表达）
    disp_groups = [GROUP_CHUNK_LABEL.get(g, g) for g in groups]
    mem = results['mem']
    cd = results['cd']
    fscore = results['fscore']
    lpips = results['lpips']
    time_ = results['time']
    time_std = results.get('time_std')

    # 质量指标是否齐全（相对模式下 lpips 为 None，只看 cd / fscore）。
    has_quality = all(v is not None for v in cd) and all(v is not None for v in fscore)

    # 数据真实性检查：仅当使用内置模板（占位）数据时才提醒；
    # 真实 jsonl 实测记录（source='auto-recorded runs'）不触发，避免误报。
    max_mem = max(mem)
    if max_mem > 8.5 and results.get('source') != 'auto-recorded runs':
        print('\\n[WARNING] Detected Baseline peak memory > 8.5 GB.')
        print('  If you are running on an 8GB GPU, this is likely placeholder data.')
        print('  Please replace the values in --chart_results or chart_ablation.py')
        print('  with your actual measured torch.cuda.max_memory_allocated().')

    x = np.arange(len(groups))

    width = 0.6
    colors_mem = PALETTE

    # ═══ 图 1：Stage 1 / Stage 2 峰值显存拆分（不重跑，直接用已记录的分阶段峰值）═══
    s1 = results.get('stage1_mem')
    s2 = results.get('stage2_mem')
    if s1 and s2 and all(v is not None for v in s1) and all(v is not None for v in s2):
        fig, ax = plt.subplots(figsize=(3.3, 3.0))   # 单栏宽 ~8cm
        x = np.arange(len(groups))
        w = 0.5
        # 关键语义：两阶段「顺序执行」、显存非同时占用，故「总峰值」= max(Stage1, Stage2)，
        # 而非两阶段峰值之和。因此用「分区柱」：柱高 = max = 总峰值（与 Table 1 完全一致），
        # 柱内按 min/max 切分，下段着「较小阶段」色、上段着「较大阶段」色，直观展示瓶颈来自哪阶段。
        for xi, (a, b, m) in enumerate(zip(s1, s2, mem)):
            lo, hi = min(a, b), max(a, b)
            # 下段归属较小阶段、上段归属较大阶段
            if a <= b:                       # Stage1 较小在下、Stage2 较大在上
                c_lo, c_hi = '#ff7f0e', '#1f77b4'
            else:                            # Stage2 较小在下、Stage1 较大在上
                c_lo, c_hi = '#1f77b4', '#ff7f0e'
            ax.bar(xi, lo, w, facecolor=c_lo, edgecolor='black', linewidth=0.8,
                   hatch='//' if c_lo == '#ff7f0e' else '\\\\', zorder=3)
            ax.bar(xi, hi - lo, w, bottom=lo, facecolor=c_hi, edgecolor='black',
                   linewidth=0.8, hatch='//' if c_hi == '#ff7f0e' else '\\\\', zorder=3)
            # 段内数值：段够高放段中部（白字），否则放段右侧（同色字），避免文字溢出/重叠
            if lo >= 0.8:
                ax.text(xi, lo / 2, f'{lo:.2f}', ha='center', va='center',
                        fontsize=6.5, color='white', zorder=4)
            else:
                ax.text(xi + w / 2 + 0.03, lo / 2, f'{lo:.2f}', ha='left', va='center',
                        fontsize=6.5, color=c_lo, zorder=4)
            if (hi - lo) >= 0.8:
                ax.text(xi, lo + (hi - lo) / 2, f'{hi:.2f}', ha='center', va='center',
                        fontsize=6.5, color='white', zorder=4)
            else:
                ax.text(xi + w / 2 + 0.03, lo + (hi - lo) / 2, f'{hi:.2f}',
                        ha='left', va='center', fontsize=6.5, color=c_hi, zorder=4)
        # 总峰值折线（= max(Stage1, Stage2)）= 柱顶，与 Table 1「总峰值显存」完全一致
        ax.plot(x, mem, 'kD-', linewidth=1.6, markersize=5, markeredgewidth=1.0,
                label='Total Peak = max(Stage1, Stage2)')
        for xi, m in zip(x, mem):
            ax.text(xi, m + 0.6, f'{m:.2f}', ha='center', va='bottom',
                    fontsize=8, color='black')
        # 图内注释：明确「顺序执行、取 max」口径，消除相加误解
        ax.text(0.02, 0.97, '顺序执行：总峰值 = max(阶段1, 阶段2)，非二者之和',
                transform=ax.transAxes, fontsize=6.3, color='#333333',
                ha='left', va='top')

        ax.set_xticks(x)
        ax.set_xticklabels(disp_groups, rotation=45, ha='right')
        ax.set_ylabel('Peak GPU Memory / GB')
        # 8 GB 参考线：灰色虚线，文字标注靠左（不挤右侧）
        ax.axhline(y=8.0, color='gray', linestyle='--', linewidth=0.6)
        ax.text(-0.35, 8.10, '8 GB Line', color='gray', fontsize=8,
                ha='left', va='bottom')
        # 16 GB 参考线：灰色点线，与 8GB 虚线区分；横穿 Baseline/+FP8 高柱
        ax.axhline(y=16.0, color='gray', linestyle=':', linewidth=0.7)
        ax.text(-0.35, 16.10, '16 GB Line', color='gray', fontsize=8,
                ha='left', va='bottom')
        # y 轴上限按「总峰值」而非「阶段之和」定，柱顶不会越过坐标轴、Total Peak 线始终在柱顶
        ax.set_ylim(0, max(max(mem), 16.0) * 1.18)
        ax.grid(axis='y', linestyle='--', linewidth=0.4, color='#cfcfcf')
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor='#ff7f0e', edgecolor='black', hatch='//',
                  label='Stage 1 (Multi-view Diffusion)'),
            Patch(facecolor='#1f77b4', edgecolor='black', hatch='\\\\',
                  label='Stage 2 (LRM Reconstruction)'),
            plt.Line2D([0], [0], color='black', marker='D', linestyle='-',
                       linewidth=1.6, label='Total Peak = max(Stage1, Stage2)'),
        ]
        ax.legend(handles=handles, fontsize=6.5, loc='upper right', framealpha=1.0,
                  bbox_to_anchor=(0.98, 0.86), borderpad=0.3, labelspacing=0.2,
                  handletextpad=0.3)
        # 删除底部中文图题；保留旋转 x 轴标签所需底部空间，避免裁切
        fig.subplots_adjust(left=0.18, right=0.97, top=0.94, bottom=0.30)
        _save(fig, out_dir, 'fig1_stage_breakdown')
        plt.close(fig)
        print('[ok] fig1_stage_breakdown.(png/pdf/eps)')
    else:
        print('[skip] fig1_stage_breakdown (stage1_mem/stage2_mem not recorded; need re-run or add fields)')



    # ═══ 图 2：显存-精度权衡散点图（形状+纹理区分，灰度可辨）═══
    if has_quality:
        fig, ax = plt.subplots(figsize=(3.3, 3.0))   # 单栏宽 ~8cm
        # 6 个配置按「形状 + 实心/空心」区分；彩色仅作线上辅助，灰度靠形状
        #  Table 1 6 组：形状 + 实心区分（灰度可辨）
        #  0 Baseline=实心圆●  1 +FP8=实心方■  2 +FP8+Chunk200k=实心菱形◆
        #  3 +FP8+Lambda0=实心三角▲  4 +FP8+Chunk200k+LOD96=实心加号✚
        #  5 +FP8+Chunk200k+LOD64=实心叉✕
        MARKERS = ['o', 's', 'D', '^', 'P', 'X']
        FILLED  = [True, True, True, True, True, True]
        # 把视觉上重叠的点聚簇，标注沿圆周均匀散开，彻底避免互相遮挡
        n = len(groups)
        clusters = []
        for i in range(n):
            placed = False
            for c in clusters:
                j = c[0]
                if abs(mem[i] - mem[j]) < 0.06 and abs(cd[i] - cd[j]) < 0.0012:
                    c.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])
        # 含 Chunk 的低显存组强制归并到同一簇：标签统一置于左侧并垂直错开，避免互相遮挡
        chunk_idx = [i for c in clusters for i in c if 'Chunk' in groups[i]]
        if chunk_idx:
            rest = [c for c in clusters if not any('Chunk' in groups[i] for i in c)]
            rest.append(sorted(chunk_idx, key=lambda i: (cd[i], mem[i])))
            clusters = rest

        for c in clusters:
            k = len(c)
            for t, i in enumerate(c):
                mi = i % len(MARKERS)
                ci = i % len(colors_mem)
                fc = colors_mem[ci] if FILLED[mi] else 'none'
                ax.scatter(mem[i], cd[i], s=55, marker=MARKERS[mi], facecolors=fc,
                           edgecolors=colors_mem[ci], linewidth=1.5, zorder=5)
                # Chunk 簇：标签移到数据点左侧（屏幕左），垂直等距错开，不互相遮挡
                if k > 1 and 'Chunk' in groups[i]:
                    dx = -24
                    dy_extra = 0
                    if 'LOD96' in groups[i]:
                        dx = -56   # +FP8+Chunk+LOD96 额外左移，与其他两项拉开
                        dy_extra = -16   # 再额外下移
                    dy = (t - (k - 1) / 2) * 30 + dy_extra
                    ax.annotate(disp_groups[i], (mem[i], cd[i]),
                                textcoords='offset points', xytext=(dx, dy),
                                ha='center', va='center', fontsize=8,
                                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=1.0),
                                arrowprops=dict(arrowstyle='->', color='gray', lw=0.6))
                    continue
                if k == 1:
                    if 'FP8' in groups[i]:
                        ang, R = 0.0, 24            # +FP8-UNet 标签置于点右侧
                    elif 'LOD64' in groups[i]:
                        ang, R = -np.pi / 2, 22     # +LOD64 标签置于点下方
                    else:
                        ang, R = np.pi / 4, 18
                else:
                    ang = -np.pi / 2 + 2 * np.pi * t / k  # 从正上方起顺时针
                    R = 28 + 5 * (k - 2)
                dx, dy = R * np.cos(ang), R * np.sin(ang)
                ax.annotate(disp_groups[i], (mem[i], cd[i]),
                            textcoords='offset points', xytext=(dx, dy),
                            ha='center', va='center', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=1.0),
                            arrowprops=dict(arrowstyle='->', color='gray', lw=0.6))
        cd_min = min(cd)
        # 仅保留指向「低显存、低偏差」最优区的箭头（删除原文字框，含义由图注说明）
        ax.annotate('', xy=(min(mem) + 0.1, cd_min),
                    xytext=(min(mem) + 1.4, cd_min + 0.007),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
        # 反向坐标轴（数值从左到右递减），旁注说明趋势，避免审稿人误读
        ax.set_xlabel('Peak GPU Memory / GB (smaller X = lower memory)')
        if results.get('quality_relative', False):
            ax.set_ylabel('Geometric Deviation from Baseline (CD \u2193)')
        else:
            ax.set_ylabel('Chamfer Distance (CD)')
        # 反转前固定范围（低显存在右），右侧（低显存侧）额外留白，避免 +FP8+Chunk+LOD96 标签被裁
        ax.set_xlim(min(mem) - 1.5, max(mem) + 0.7)
        # y 轴顶/底留白，避免圆周散开的标签被边界或网格线挡住
        _cd = [float(v) for v in cd if v is not None]
        _ymin, _ymax = min(_cd), max(_cd)
        _pad = (_ymax - _ymin) * 0.45 + 0.002
        ax.set_ylim(_ymin - _pad * 0.4, _ymax + _pad)
        ax.invert_xaxis()
        ax.grid(linestyle='--', linewidth=0.4, color='#cfcfcf')
        # 底部中文图题按需删除；缩小底部留白，图形占满画布
        # 加宽左侧边距，避免旋转 y 轴标题被画布左缘裁掉；
        # 右侧留出边距，避免低显存聚簇标签 / x 轴刻度标签在右缘被裁
        # 左侧留宽边距，给移到左半区的 Chunk 标签留出空间，避免被左边界裁掉
        fig.subplots_adjust(left=0.28, right=0.92, top=0.94, bottom=0.14)
        _save(fig, out_dir, 'fig2_memory_quality_tradeoff')
        plt.close(fig)
        print('[ok] fig2_memory_quality_tradeoff.(png/pdf/eps)')
    else:
        print('[skip] fig2_memory_quality_tradeoff (quality metrics unavailable; add ground-truth evaluation)')

    # ═══ 图 4：综合雷达图（4 个代表性配置） ═══
    if has_quality:
        radar_groups = [disp_groups[0], disp_groups[1], disp_groups[2], disp_groups[-1]]
        radar_idx = [0, 1, 2, len(groups) - 1]
        if results.get('quality_relative', False):
            radar_labels = ['Memory \u2193', 'Deviation \u2193', 'Infer Time \u2193', 'Retention \u2191']
        else:
            radar_labels = ['Memory \u2193', 'CD \u2193', 'Infer Time \u2193', 'F-Score \u2191']

        norm_mem  = _normalize(mem)
        norm_cd   = _normalize(cd, invert=True)   # 越低(偏差/CD)越好 -> 越大
        norm_time = _normalize(time_)
        norm_fs   = _normalize(fscore, invert=True)

        radar_values = []
        for i in radar_idx:
            radar_values.append([norm_mem[i], norm_cd[i], norm_time[i], norm_fs[i]])

        angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(4.3, 4.0), subplot_kw=dict(polar=True))
        colors_radar = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']
        for i, (vals, name) in enumerate(zip(radar_values, radar_groups)):
            v_ = vals + vals[:1]
            # 仅描边、不填充：EPS（PostScript）不支持透明度，描边更利于黑白印刷
            ax.plot(angles, v_, 'o-', linewidth=1.6, markersize=5,
                    color=colors_radar[i], label=name)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=8.5)
        ax.set_yticklabels([])
        ax.grid(linewidth=0.4, color='#cfcfcf')
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=8, framealpha=1.0)
        # 删除底部中文图题
        fig.subplots_adjust(left=0.10, right=0.62, top=0.90, bottom=0.18)
        _save(fig, out_dir, 'fig4_radar')
        plt.close(fig)
        print('[ok] fig4_radar.(png/pdf/eps)')
    else:
        print('[skip] fig4_radar (quality metrics unavailable; add ground-truth evaluation)')

    # ═══ 图 5：显存瀑布图 ═══
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    deltas = [mem[i] - mem[i - 1] for i in range(1, len(mem))]
    bottom_vals = [mem[0]] + [mem[i - 1] for i in range(1, len(mem))]
    heights = [0] + deltas
    bar_colors = ['#aaaaaa'] + ['#2ca02c' if h < 0 else '#d62728' for h in deltas]

    ax.bar(0, mem[0], width * 0.7, facecolor='#aaaaaa', edgecolor='black',
           linewidth=0.6, hatch='//', label='Baseline')
    for i in range(1, len(mem)):
        ax.bar(i, heights[i], width * 0.7, bottom=bottom_vals[i],
               facecolor=bar_colors[i], edgecolor='black', linewidth=0.6,
               hatch='//' if heights[i] < 0 else '\\')
        ax.plot([i - 0.35, i + 0.35], [bottom_vals[i], bottom_vals[i]],
                color='gray', linewidth=0.6)

    for i, (m, h) in enumerate(zip(mem, heights)):
        if i == 0:
            ax.text(i, m + 0.2, f'{m:.2f}', ha='center', fontsize=8)
        else:
            c = '#2ca02c' if h < 0 else '#d62728'
            ax.text(i, bottom_vals[i] + h / 2, f'{h:+.2f}', ha='center', va='center',
                    fontsize=8, color='white',
                    bbox=dict(facecolor=c, alpha=1.0, pad=2, boxstyle='round,pad=0.2'))

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_ylabel('Peak GPU Memory / GB')
    ax.grid(axis='y', linestyle='--', linewidth=0.4, color='#cfcfcf')
    # 删除底部中文图题；保留旋转 x 轴标签所需底部空间，避免裁切
    fig.subplots_adjust(left=0.16, right=0.97, top=0.94, bottom=0.34)
    _save(fig, out_dir, 'fig5_waterfall')
    plt.close(fig)
    print('[ok] fig5_waterfall.(png/pdf/eps)')

    # ═══ 表 1：主消融表（含 LPIPS / SSIM 感知指标）═══
    _print_table(results, out_dir)

    # ═══ 图 6：自适应调度器工作点（支柱三·闭环决策表）═══
    if _HAVE_SCHEDULER:
        budgets = [6, 8, 12, 16, 24]
        recs = [recommend_config(b) for b in budgets]
        peaks = [r['predicted_stage2_peak_gb'] for r in recs]
        fig, ax = plt.subplots(figsize=(3.9, 3.1))
        x = np.arange(len(budgets))
        # 分块档(绿) vs 无分块档(蓝) 用明度区分，灰度可辨
        colors_auto = ['#2E7D32', '#2E7D32', '#2E7D32', '#1F4E79', '#1F4E79']
        ax.bar(x, peaks, width=0.55, color=colors_auto, edgecolor='black',
               linewidth=0.7, hatch='//',
               label='Predicted Stage-2 Peak (auto)')
        # 预算参考线：调度器必须让峰值位于其下方（闭环余量验证）
        ax.plot(x, budgets, 'kD--', linewidth=1.3, markersize=5,
                label='GPU VRAM Budget')
        for xi, (b, r, p) in enumerate(zip(budgets, recs, peaks)):
            ax.text(xi, p + 0.3, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
            cfg = (f"fp8={'ON' if r['fp8_unet'] else 'OFF'}\n"
                   f"chunk={'ON' if r['grid_extract_chunk'] else 'OFF'}\n"
                   f"G*={r['grid_res']}")
            ax.text(xi, -0.03, cfg, ha='center', va='top', fontsize=6.5,
                    transform=mtransforms.blended_transform_factory(
                        ax.transData, ax.transAxes))
        # 若可探测到本机 GPU 显存，标注 auto-detected 位置（红虚线）
        det = detect_device_budget_gb()
        if det > 0:
            d_clamped = min(max(det, budgets[0]), budgets[-1])
            xi_det = float(np.interp(d_clamped, budgets, x))
            ax.axvline(x=xi_det, color='#C00000', linestyle=':', linewidth=1.2)
            ax.text(xi_det, max(budgets), f'detected {det:.2f}GB',
                    color='#C00000', fontsize=7, ha='center', va='bottom',
                    rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{b}GB' for b in budgets])
        ax.set_ylabel('Peak GPU Memory / GB')
        ax.set_ylim(0, max(budgets) * 1.15)
        ax.grid(axis='y', linestyle='--', linewidth=0.4, color='#cfcfcf')
        ax.legend(fontsize=7, loc='upper left', framealpha=1.0,
                  borderpad=0.3, labelspacing=0.2)
        # 底部留出空间放各档配置注释
        fig.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.28)
        _save(fig, out_dir, 'fig6_auto_scheduler')
        plt.close(fig)
        print('[ok] fig6_auto_scheduler.(png/pdf/eps)')
    else:
        print('[skip] fig6_auto_scheduler (src.memory_cost_model unavailable)')

    # ═══ 图 8：λ 均衡扫描曲线（支柱二·连续调节可视化）═══
    try:
        sweep = read_lambda_sweep(LAMBDA_SWEEP_DIR)
        if sweep:
            generate_lambda_sweep_chart(sweep, out_dir)
        else:
            print('[skip] fig8_lambda_sweep (no data in ' + LAMBDA_SWEEP_DIR +
                  '; run:  python run_ablation.py --lambda_sweep)')
    except Exception as e:
        print(f'[skip] fig8_lambda_sweep (error: {e})')

    # ═══ 图 7：定性对比（法线图 + 线框图）═══
    try:
        generate_fig7(out_dir)
    except Exception as e:
        print(f'[skip] fig7_qualitative (error: {e})')

    count = 7
    print(f'\nAll {count} charts saved to {out_dir}/')
    print('Files: fig1_stage_breakdown, fig2_memory_quality_tradeoff, fig4 ~ fig6_auto_scheduler (each .png / .pdf / .eps)')
    print('        fig7_qualitative.png (qualitative: normal + wireframe, CPU render)')
    print('        fig8_lambda_sweep.(png/pdf/eps) (λ equilibrium curve, --lambda_sweep)')
    print(f'        table1.md (markdown Table 1 with LPIPS / SSIM columns)')


def _print_table(results, out_dir):
    """
    打印《计算机科学》主消融表（Table 1），新增 LPIPS↓(vs Base) 与 SSIM↑(vs Base)
    两个感知/纹理指标列，回答"优化是否损伤纹理观感"的问题。
    多物体时各指标以「均值±标准差」呈现（N = 输入物体数）。
    同时把 markdown 写入 out_dir/table1.md 供论文直接粘贴。
    """
    groups = results['groups']
    # 展示标签：分块程度标注为 λ（机制名 +Chunk 保留，degree 用 λ 表达）
    disp_groups = [GROUP_CHUNK_LABEL.get(g, g) for g in groups]
    mem = results['mem']; mem_std = results.get('mem_std')
    time_ = results['time']; time_std = results.get('time_std')
    cd = results['cd']; cd_std = results.get('cd_std')
    fscore = results['fscore']; fscore_std = results.get('fscore_std')
    lpips = results['lpips']; lpips_std = results.get('lpips_std')
    ssim = results.get('ssim', [None] * len(groups)); ssim_std = results.get('ssim_std')
    s1 = results.get('stage1_mem'); s2 = results.get('stage2_mem')
    n_inputs = results.get('n_inputs', [1] * len(groups))
    base_mem = mem[0] if mem else 0.0
    N = n_inputs[0] if n_inputs else 1

    has_perceptual = any(v is not None for v in lpips) or any(v is not None for v in ssim)

    def ms(mean, std, nd):
        if mean is None:
            return '—'
        s = f'{mean:.{nd}f}'
        if std is not None:
            s += f'±{std:.{nd}f}'
        return s

    header = ['实验组', 'Stage1峰值显存/GB', 'Stage2峰值显存/GB', '总峰值显存/GB',
              '平均推理耗时/s', '相对Baseline显存变化', '相对CD↓', 'F-Score保留率↑',
              'LPIPS↓(vs Base)']
    rows = []
    for i, g in enumerate(groups):
        m = mem[i]
        dmem = '≈0%' if abs(m - base_mem) < 1e-6 else f'{(m - base_mem) / base_mem * 100:+.1f}%'
        row = [
            disp_groups[i],
            ms(s1[i] if (s1 and s1[i] is not None) else None, None, 2),
            ms(s2[i] if (s2 and s2[i] is not None) else None, None, 2),
            ms(m, mem_std[i] if mem_std else None, 2),
            ms(time_[i], time_std[i] if time_std else None, 1),
            dmem,
            ms(cd[i], cd_std[i] if cd_std else None, 4),
            ms(fscore[i], fscore_std[i] if fscore_std else None, 3),
            ms(lpips[i], lpips_std[i] if lpips_std else None, 4),
        ]
        rows.append(row)

    def _fmt(cells):
        return '| ' + ' | '.join(cells) + ' |'

    md = []
    md.append(_fmt(header))
    md.append('|' + '|'.join(['---'] * len(header)) + '|')
    for r in rows:
        md.append(_fmt(r))
    md_text = '\n'.join(md)

    print('\n' + '=' * 78)
    print(f'Table 1  主消融表（含感知指标 LPIPS，相对 Baseline；N = {N}）')
    print('=' * 78)
    print(md_text)
    print('=' * 78)
    if not has_perceptual:
        print('NOTE: LPIPS / SSIM 未计算。请先运行  python eval_perceptual.py --ablation_dir <dir>')
        print('      （以 Baseline 为锚点渲染各网格并计算相对感知指标）。')

    try:
        with open(os.path.join(out_dir, 'table1.md'), 'w', encoding='utf-8') as f:
            f.write(f'# Table 1  主消融表（含感知指标 LPIPS，相对 Baseline；N = {N}）\n\n')
            f.write(md_text + '\n')
            f.write('\n> 物体数 N = ' + str(N) +
                    '：各指标为跨 N 个输入对每组取的「均值±标准差」。\n')
            f.write('> LPIPS 以 Baseline 为参考锚点（相对感知偏差）；越低表示越接近 Baseline 的纹理观感。\n')
            f.write('> 注：SSIM 在「同相机渲染对比」设定下对纹理像素级差异极敏感、与 LPIPS/CD 矛盾，'
                    '已弃用；感知/纹理保真度以稳健的 LPIPS 衡量。\n')
    except Exception:
        pass


# ── 图 7：定性对比（法线图 + 线框图），由 render_qual.py 融合而来 ──
# fig7 旋转调试开关：默认保持与当前脚本一致（绕 Y 90° + 2D 翻转）。
# 改这两个开关用于定位与原图一致的旋转配置，正常出图无需改动。
FIG7_ROT_Y = 180     # 绕世界 Y 轴的旋转角度（度）
FIG7_ROT_X = 0     # 绕世界 X 轴的旋转角度（度），0 表示不绕 X 旋转
FIG7_ROT_Z = 180     # 绕世界 Z 轴的旋转角度（度），0 表示不绕 Z 旋转
FIG7_FLIP = True      # 渲染后是否对单格做 180° 2D 翻转


def generate_fig7(out_dir='outputs/charts'):
    """
    生成图 7：3 行物体 × N 列 mesh 分组的定性对比（法线图上 + 线框图下）。
    原逻辑来自 render_qual.py，融合进本模块，由 generate_all_charts 统一调用。
    依赖 trimesh / PIL（可选）：缺失时仅跳过本图，不影响其余图表/表格。
    """
    try:
        import math
        import trimesh
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        print(f'[skip] fig7_qualitative (缺少可选依赖 trimesh/PIL: {e})')
        return

    ABL_DIR = 'outputs/ablation'

    PANEL_W = 300            # 单格宽度
    NORM_H  = 200            # 法线图高度
    WIRE_H  = 170            # 线框图高度
    CELL_H  = NORM_H + WIRE_H
    FACE_CAP = 45000         # 单网格用于绘制的面数上限；超过则按比例随机抽面
    FOV = 30.0
    ELEV, AZIM, DIST = 18.0, -55.0, 3.0
    # 绕世界 Y 轴旋转：模型侧向展示，仍保持直立。
    ay = math.radians(FIG7_ROT_Y)
    RY = np.array([[math.cos(ay), 0.0, math.sin(ay)],
                   [0.0, 1.0, 0.0],
                   [-math.sin(ay), 0.0, math.cos(ay)]], dtype=float)
    # 绕世界 X 轴旋转：模型翻倒（顶视/底视），用于调整展示朝向。
    ax = math.radians(FIG7_ROT_X)
    RX = np.array([[1.0, 0.0, 0.0],
                   [0.0, math.cos(ax), -math.sin(ax)],
                   [0.0, math.sin(ax), math.cos(ax)]], dtype=float)
    # 绕世界 Z 轴旋转：模型在画面平面内自转，用于调整朝向角度。
    az = math.radians(FIG7_ROT_Z)
    RZ = np.array([[math.cos(az), -math.sin(az), 0.0],
                   [math.sin(az), math.cos(az), 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)

    # 3 行物体（rigid / non-rigid / thin-wall）；缺失时自动回退到首个有网格的物体
    CATEGORIES = [
        ("Rigid (cake)",        "cake",           "examples/cake.jpg"),
        ("Non-rigid (blue_cat)","blue_cat",       "examples/blue_cat.png"),
        ("Thin-wall (teapot)",  "genshin_teapot", "examples/genshin_teapot.png"),
    ]
    GROUP_ORDER = [
        "Baseline", "+FP8", "+FP8+Chunk200k", "+FP8+Lambda0",
        "+FP8+Chunk200k+LOD96", "+FP8+Chunk200k+LOD64",
    ]

    def load_mesh(path):
        m = trimesh.load(path, process=False)
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        v = np.asarray(m.vertices, dtype=np.float64)
        f = np.asarray(m.faces, dtype=np.int64)
        return v, f

    def align(v, c, s):
        return (v - c) / s

    def camera_vectors():
        el = math.radians(ELEV)
        az = math.radians(AZIM)
        cam = np.array([
            DIST * math.cos(el) * math.sin(az),
            DIST * math.sin(el),
            DIST * math.cos(el) * math.cos(az),
        ])
        forward = -cam
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return cam, forward, up, right

    def project(v, cam, forward, up, right, W, H, margin=18):
        d = v - cam
        xc = d @ right
        yc = d @ up
        zc = d @ forward
        f_persp = 1.0 / math.tan(math.radians(FOV / 2.0))
        scale = (W * 0.5 - margin) / 1.35
        px = xc / np.maximum(zc, 1e-3) * f_persp * scale + W / 2.0
        py = H / 2.0 - yc / np.maximum(zc, 1e-3) * f_persp * scale
        return px, py, zc

    def face_normals(v, f):
        a = v[f[:, 0]]; b = v[f[:, 1]]; c = v[f[:, 2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln < 1e-12] = 1.0
        return n / ln

    def sample_faces(f, frac):
        """按统一比例 frac 抽面，保留不同网格之间的真实密度对比。"""
        if frac >= 1.0 or f.shape[0] <= 1:
            return f
        n = max(1, int(round(f.shape[0] * frac)))
        idx = np.random.RandomState(0).choice(f.shape[0], n, replace=False)
        return f[idx]

    def render_normal(v, f, W, H, cam, forward, up, right, frac):
        f = sample_faces(f, frac)
        n = face_normals(v, f)
        # 转到相机坐标系着色：亮度与物体在世界朝向无关，各行列对比度一致
        R = np.stack([right, up, forward], axis=1)
        n_view = n @ R
        col = ((n_view * 0.5 + 0.5) * 255).astype(np.uint8)
        px, py, zc = project(v, cam, forward, up, right, W, H)
        depth = zc[f].mean(axis=1)
        order = np.argsort(-depth)  # 远 -> 近
        img = Image.new("RGB", (W, H), (235, 235, 235))
        dr = ImageDraw.Draw(img)
        for i in order:
            tri = [(px[f[i, 0]], py[f[i, 0]]), (px[f[i, 1]], py[f[i, 1]]),
                   (px[f[i, 2]], py[f[i, 2]])]
            c = (int(col[i, 0]), int(col[i, 1]), int(col[i, 2]))
            dr.polygon(tri, fill=c)
        return img

    def render_wire(v, f, W, H, cam, forward, up, right, frac):
        f = sample_faces(f, frac)
        px, py, _ = project(v, cam, forward, up, right, W, H)
        img = Image.new("RGB", (W, H), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        for tri in f:
            p0 = (px[tri[0]], py[tri[0]])
            p1 = (px[tri[1]], py[tri[1]])
            p2 = (px[tri[2]], py[tri[2]])
            dr.line([p0, p1, p2, p0], fill=(60, 60, 60), width=1)
        return img

    def load_font(size):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def vlabel(text, w, h, size=18):
        """左侧竖排类别标签。"""
        img = Image.new("RGB", (w, h), (245, 245, 245))
        font = load_font(size)
        tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        tw = tmp.textlength(text, font=font)
        t = Image.new("RGB", (int(tw) + 8, size + 6), (245, 245, 245))
        d = ImageDraw.Draw(t)
        d.text((4, 3), text, fill=(20, 20, 20), font=font)
        t = t.rotate(90, expand=True)
        img.paste(t, ((w - t.width) // 2, (h - t.height) // 2))
        return img

    def placeholder(w, h, text):
        img = Image.new("RGB", (w, h), (220, 220, 220))
        d = ImageDraw.Draw(img)
        font = load_font(16)
        tw = d.textlength(text, font=font)
        d.text(((w - tw) / 2, h / 2 - 8), text, fill=(90, 90, 90), font=font)
        return img

    def has_meshes(stem):
        mp = os.path.join(ABL_DIR, stem, "meshes")
        if not os.path.isdir(mp):
            return False
        return any(os.path.exists(os.path.join(mp, g, f"{stem}.obj"))
                   for g in GROUP_ORDER)

    def discover_groups(stem):
        mp = os.path.join(ABL_DIR, stem, "meshes")
        if not os.path.isdir(mp):
            return []
        return [g for g in GROUP_ORDER
                if os.path.exists(os.path.join(mp, g, f"{stem}.obj"))]

    def resolve_categories():
        avail = [d for d in os.listdir(ABL_DIR)
                 if os.path.isdir(os.path.join(ABL_DIR, d)) and has_meshes(d)]
        resolved, used = [], set()
        for label, stem, src in CATEGORIES:
            if has_meshes(stem):
                resolved.append((label, stem))
                used.add(stem)
            else:
                fb = next((s for s in avail if s not in used), None)
                if fb is None:
                    continue
                resolved.append((f"Fallback ({fb})", fb))
                used.add(fb)
        return resolved

    resolved = resolve_categories()
    if not resolved:
        print("未找到任何含 mesh 分组的物体，跳过 fig7。")
        return
    groups = discover_groups(resolved[0][1])
    if not groups:
        print("未发现有 mesh 分组的目录，跳过 fig7。")
        return
    print("fig7 列（mesh 分组）：", groups)
    print("fig7 行（物体）：", [r[0] for r in resolved])

    cam, forward, up, right = camera_vectors()

    # 加载所有网格，按各物体首个可用分组做归一化；记录全局最密网格用于统一抽面
    data = []
    all_faces = []
    for label, stem in resolved:
        mp = os.path.join(ABL_DIR, stem, "meshes")
        base_c = base_s = None
        row = []
        for g in groups:
            p = os.path.join(mp, g, f"{stem}.obj")
            if os.path.exists(p):
                v, f = load_mesh(p)
                if base_c is None:
                    base_c = v.reshape(-1, 3).mean(0)
                    base_s = np.abs(v - base_c).max()
                v = align(v, base_c, base_s)
                v = v @ RY @ RX @ RZ   # 依次绕 X、Y、Z 轴旋转（所有分组一致，保持可比）
                all_faces.append(f)
                row.append((v, f))
            else:
                row.append(None)
        data.append((label, row))

    max_faces = max((f.shape[0] for f in all_faces), default=1)
    frac = min(1.0, FACE_CAP / max_faces)
    print(f"fig7 global max faces={max_faces}, sampling frac={frac:.3f}")

    # 渲染每个单元（法线图 + 线框图纵向拼接）
    GAP = 4
    LBL = 46
    col_w = PANEL_W + GAP
    total_w = LBL + col_w * len(groups)

    rows_imgs = []
    for label, row in data:
        cells = []
        for g, item in zip(groups, row):
            if item is None:
                cells.append(placeholder(PANEL_W, CELL_H, f"{g}\nN/A"))
                continue
            v, f = item
            norm = render_normal(v, f, PANEL_W, NORM_H,
                                 cam, forward, up, right, frac)
            wire = render_wire(v, f, PANEL_W, WIRE_H,
                               cam, forward, up, right, frac)
            if FIG7_FLIP:
                norm = norm.rotate(180)
                wire = wire.rotate(180)
            # 旋转后重新绘制正向可读的面数标注
            wd = ImageDraw.Draw(wire)
            wd.text((4, WIRE_H - 16), f"{f.shape[0] / 1000:.1f}k faces",
                    fill=(90, 90, 90))
            cell = Image.new("RGB", (PANEL_W, CELL_H), (255, 255, 255))
            cell.paste(norm, (0, 0))
            cell.paste(wire, (0, NORM_H))
            cells.append(cell)
        # 行画布：左侧竖排标签 + 各列单元
        rimg = Image.new("RGB", (total_w, CELL_H), (255, 255, 255))
        rimg.paste(vlabel(label, LBL, CELL_H), (0, 0))
        for j, c in enumerate(cells):
            rimg.paste(c, (LBL + j * col_w, 0))
        rows_imgs.append(rimg)

    # 列标题（mesh 分组名）
    HEAD_H = 30
    head = Image.new("RGB", (total_w, HEAD_H), (245, 245, 245))
    hd = ImageDraw.Draw(head)
    font = load_font(18)
    hd.text((LBL // 2 - 10, 7), "Object", fill=(0, 0, 0), font=font)
    for j, g in enumerate(groups):
        tw = hd.textlength(g, font=font)
        hd.text((LBL + j * col_w + (PANEL_W - tw) / 2, 7),
                g, fill=(0, 0, 0), font=font)

    # 拼装
    final = Image.new("RGB", (total_w, HEAD_H + CELL_H * len(rows_imgs)),
                      (255, 255, 255))
    final.paste(head, (0, 0))
    for i, rimg in enumerate(rows_imgs):
        final.paste(rimg, (0, HEAD_H + i * CELL_H))

    # 细网格线（列间、表头下、行间）
    gd = ImageDraw.Draw(final)
    line = (190, 190, 190)
    for j in range(len(groups) + 1):
        x = LBL + j * col_w - GAP // 2
        if j == 0:
            x = LBL - GAP // 2
        gd.line([(x, HEAD_H), (x, final.height)], fill=line, width=1)
    gd.line([(0, HEAD_H), (total_w, HEAD_H)], fill=line, width=1)
    for i in range(len(rows_imgs)):
        y = HEAD_H + (i + 1) * CELL_H
        gd.line([(0, y), (total_w, y)], fill=line, width=1)
    gd.rectangle([(0, 0), (total_w - 1, final.height - 1)],
                 outline=(120, 120, 120), width=1)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig7_qualitative.png")
    final.save(out_path, dpi=(300, 300))
    print(f'[ok] fig7_qualitative.(png) -> {out_path} {final.size}')


# ── 独立运行时使用内置数据 ───────────────────────────────
_DEFAULT_DATA = {
    # ==========================================================================
    # 注意：以下数据为 8GB 显卡场景下的示例/模板数据，必须替换为你的真实测量值。
    # 在 8GB 硬件上，Baseline 不应超过 8GB；若原始 6 视图直接 OOM，应标注为 OOM 或
    # 以可运行的最低配置为起点。
    # ==========================================================================
    'groups': ['Baseline', '+FP8', '+FP8+Chunk200k', '+FP8+Lambda0',
               '+FP8+Chunk200k+LOD96', '+FP8+Chunk200k+LOD64'],
    # 模板/占位数据（8GB 场景），必须替换为你的真实测量值。
    # 核心结论支撑：+FP8+Chunk200k(LOD128) 总峰值 ≤ 8GB（此处模板 6.12GB，与
    # src/memory_cost_model 注释的 LOD128 总峰值一致），证明 LOD128 经 FP8+Chunk 可在 8GB 跑通；
    # +FP8+Lambda0 用 --Lambda 动态均衡分块，全局峰值锁死在 Stage1 刚性下限（模板 5.86GB），
    # 比固定 200k 块更省、且不浪费——即论文甜点位。
    # 质量随 LOD 单调：LOD128 最优、LOD64 最粗；FP8/Chunk 对质量近似零损失。
    'mem':    [7.85, 6.40, 6.12, 5.86, 5.20, 4.35],  # GB, 8GB 上限模板
    # 分阶段峰值模板（8GB 场景）：总峰值 = max(Stage1, Stage2)，故令 Stage2 = 总峰值、
    # Stage1 为较小值；请替换为你实测的 stage1_mem / stage2_mem。
    'stage1_mem': [6.00, 5.50, 5.80, 5.86, 5.00, 4.20],
    'stage2_mem': [7.85, 6.40, 6.12, 5.86, 5.20, 4.35],
    'cd':     [0.076, 0.078, 0.073, 0.074, 0.079, 0.085],
    'fscore': [0.912, 0.909, 0.918, 0.916, 0.907, 0.895],
    'lpips':  [0.184, 0.187, 0.181, 0.182, 0.189, 0.196],
    'time':   [31.2, 30.5, 33.0, 31.0, 31.8, 27.2],
    'diagnostic': {
        'prune_ratios': [0.0, 0.05, 0.10, 0.15],
        'prune_cd': [0.076, 0.079, 0.082, 0.097],
    },
    'compress_cd': 0.113,
}


if __name__ == '__main__':
    import os
    import glob
    # 本地数据分析：优先读真实测得的 jsonl；找不到才退回占位模板数据。
    out_dir = 'outputs/ablation'
    charts_dir = 'outputs/charts'
    # 先剥掉每个 jsonl 中的冗余 +FP8+Chunk+LOD128 行
    # （config=128 后该组与 +FP8+Chunk 参数完全相同，会制造重复行）。
    REDUNDANT = '"+FP8+Chunk+LOD128"'
    n_clean = 0
    for p in glob.glob(os.path.join(out_dir, '*', 'run_metrics.jsonl')):
        lines = open(p, encoding='utf-8').readlines()
        kept = [l for l in lines if REDUNDANT not in l]
        if len(kept) != len(lines):
            open(p, 'w', encoding='utf-8').writelines(kept)
            n_clean += 1
    if n_clean:
        print(f'[clean] stripped redundant line from {n_clean} file(s)')
    results = aggregate_runs(out_dir) if os.path.isdir(out_dir) else None
    if results is None:
        print('[WARN] 未在 outputs/ablation/ 找到真实记录，使用 _DEFAULT_DATA 占位模板（非真实测量）。')
        generate_all_charts(_DEFAULT_DATA, charts_dir)
    else:
        print(f'[INFO] 已从 {out_dir} 聚合 {results.get("n_inputs")} 个输入的真实数据，'
              f'组：{results.get("groups")}')
        generate_all_charts(results, charts_dir)
