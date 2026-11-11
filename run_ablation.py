"""
一键跑通 N 个输入 × 6 组主消融（论文 Table 1：+FP8 / +Chunk 逐级叠加 + LOD128/96/64 对照）
+ 相对几何/感知评测 + 出图。

多物体支持：每个输入单独跑一套 6 组，落到 outputs/<ablation_dir>/<输入名>/，
各子目录内含自己的 run_metrics.jsonl 与 meshes/<group>/；chart_ablation 会
跨子目录对每组取「均值±标准差」，从而得到统计意义上的消融表（N>1）。

用法：
    python run_ablation.py --inputs examples --max_inputs 8
    python run_ablation.py --inputs examples/blue_cat.png,examples/bird.jpg
    python run_ablation.py --inputs examples            # 跑 examples 下全部图片
    python run_ablation.py --input examples/chair_comfort.jpg   # 单输入（旧用法，仍支持）
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable  # 用它来跑子进程，保证和当前 venv 一致

DEFAULT_CONFIG = 'configs/instant-mesh-large.yaml'
DEFAULT_INPUT = 'examples/blue_cat.png'
DEFAULT_ABLATION_DIR = 'outputs/ablation'
DEFAULT_CHARTS_DIR = 'outputs/charts'
DEFAULT_TAU = 0.1

# Table 1 主消融配置：(组名, run.py 额外参数)，全部固定 --view 6（View 维度已证明无关）。
# 设计：先「+FP8 → +Chunk200k」逐级叠加（Baseline→+FP8→+FP8+Chunk200k），
#       再在「FP8+Chunk200k 轻量化全栈」上加「+FP8+Lambda0」甜点位（动态均衡分块，
#       全局峰值锁死在 Stage1 刚性下限），最后做 LOD128/96/64 三组对照，
#       核心结论：LOD128 经 FP8+Chunk200k 也能在 8 GB 跑通（总峰值 ≤ 8GB），
#       而 +FP8+Lambda0 进一步把峰值压到 Stage1 下限、更省且不浪费。
# 注：config 默认 grid_res 已于 2026-07-13 改为 128，故叠加链（Baseline/+FP8/+FP8+Chunk200k）
#     三组现在均为 LOD128（即「在 LOD128 上逐级叠加 FP8/Chunk 优化」）。
#     +FP8+Lambda0 用 --Lambda 动态反解分块（默认 LOD128），不写死块大小；
#     LOD96/64 对照仍基于 FP8+Chunk200k 下探（显式 --LOD），呈现分辨率对显存/质量的 Pareto 权衡。
#     最终 6 行：Baseline / +FP8 / +FP8+Chunk200k / +FP8+Lambda0 /
#                +FP8+Chunk200k+LOD96 / +FP8+Chunk200k+LOD64。
GROUPS = [
    ('Baseline',            ['--view', '6']),
    ('+FP8',                ['--view', '6', '--FP8Unet']),
    ('+FP8+Chunk200k',      ['--view', '6', '--FP8Unet', '--Chunk', '200000']),
    ('+FP8+Lambda0',        ['--view', '6', '--FP8Unet', '--Lambda']),
    ('+FP8+Chunk200k+LOD96',['--view', '6', '--FP8Unet', '--Chunk', '200000', '--LOD', '96']),
    ('+FP8+Chunk200k+LOD64',['--view', '6', '--FP8Unet', '--Chunk', '200000', '--LOD', '64']),
]

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')

# λ-sweep（支柱二·连续调节可视化）：固定 G=128，扫 Stage-2 分块顶点数 g，
# 对应均衡因子 λ = g / 128^3。仅取首个输入（显存为确定性量、与输入无关，无需多物体平均）。
# 注："full" 整批点（λ=1.0）在 G=128 下需 ~20 GB，故本实验要求 ≥16 GB 显存（RTX 4090）；
# 8 GB 卡会在该点 OOM，因此默认关闭，仅用 --lambda_sweep 显式开启。
# 数据独立写入 LAMBDA_SWEEP_DIR，不污染 Table 1 的 5 组聚合（由 chart_ablation.py 出 fig8_lambda_sweep）。
LAMBDA_SWEEP = [
    ('lambda_sweep_full',     0),        # 整批（λ=1.0）
    ('lambda_sweep_g800000', 800000),    # λ ≈ 0.381
    ('lambda_sweep_g400000', 400000),    # λ ≈ 0.191
    ('lambda_sweep_g200000', 200000),    # λ ≈ 0.095（≈ λ0 均衡点）
    ('lambda_sweep_g100000', 100000),    # λ ≈ 0.048
    ('lambda_sweep_g50000',   50000),    # λ ≈ 0.024
]
LAMBDA_SWEEP_DIR = 'outputs/ablation_lambda_sweep'


def parse_inputs(args):
    """把 --inputs（逗号分隔文件列表 / 目录）解析为输入图片路径列表；否则退回 --input。"""
    if args.inputs.strip():
        raw = [x.strip() for x in args.inputs.split(',') if x.strip()]
        inputs = []
        for x in raw:
            if os.path.isdir(x):
                for f in sorted(os.listdir(x)):
                    if f.lower().endswith(IMG_EXTS):
                        inputs.append(os.path.join(x, f))
            elif os.path.isfile(x):
                inputs.append(x)
            else:
                print(f'[WARN] input not found: {x}')
        return inputs
    return [args.input]


def recorded_groups(ablation_dir):
    """从某个 ablation_dir 下的 run_metrics.jsonl 读取已记录的组名集合。"""
    path = os.path.join(ablation_dir, 'run_metrics.jsonl')
    groups = set()
    if not os.path.exists(path):
        return groups
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                g = rec.get('group')
                if g:
                    groups.add(g)
            except json.JSONDecodeError:
                continue
    return groups


def run_group(name, extra, ablation_dir, input_path, args, done):
    if name in done and not args.force:
        print(f'\n[skip] {name:<14s} 已在记录中存在，跳过（用 --force 强制重跑）')
        return True

    cmd = [PY, 'run.py', args.config, input_path,
           '--ablation_name', name,
           '--ablation_dir', ablation_dir] + extra
    if args.save_video:
        cmd.append('--save_video')

    print(f'\n===== [{os.path.basename(input_path)}] [{name}] =====')
    print(' '.join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        print(f'[ERROR] 组 "{name}" 失败，返回码 {rc}')
        if not args.continue_on_error:
            sys.exit(rc)
        return False
    return True


def run_eval_relative(ablation_dir, tau):
    cmd = [PY, 'eval_relative.py', '--ablation_dir', ablation_dir, '--tau', str(tau)]
    print('\n===== 相对几何评测（Baseline 为锚点）=====')
    print(' '.join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        print(f'[WARN] eval_relative.py 返回码 {rc}，图表可能缺少相对质量。')
    return rc == 0


def run_eval_perceptual(ablation_dir):
    cmd = [PY, 'eval_perceptual.py', '--ablation_dir', ablation_dir]
    print('\n===== 相对感知评测（Baseline 为锚点）=====')
    print(' '.join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        print(f'[WARN] eval_perceptual.py 返回码 {rc}，LPIPS/SSIM 可能缺失。')
    return rc == 0


def run_charts(args):
    print('\n===== 跨输入聚合并生成图表 =====')
    try:
        from chart_ablation import aggregate_runs, generate_all_charts
    except Exception as e:
        print(f'[WARN] 无法导入 chart_ablation：{e}')
        return
    os.makedirs(args.charts_dir, exist_ok=True)
    results = aggregate_runs(out_dir=args.ablation_dir)
    if results is None:
        print('[WARN] aggregate_runs 返回 None，跳过出图。')
        return
    generate_all_charts(results, out_dir=args.charts_dir)
    print(f'[ok] 图表已写入 {args.charts_dir}')


def run_lambda_sweep(inputs, args):
    """支柱二·λ-sweep：固定 G=128，扫分块顶点数 g ∈ {full, 800k, 400k, 200k, 100k, 50k}，
    对应 λ = g/128^3。仅取首个输入（显存为确定性量）。结果写入独立目录 LAMBDA_SWEEP_DIR，
    不污染 Table 1 聚合；由 chart_ablation.py 读这些记录画 fig8_lambda_sweep。"""
    if not inputs:
        return
    inp = inputs[0]
    stem = os.path.splitext(os.path.basename(inp))[0]
    sub_dir = os.path.join(LAMBDA_SWEEP_DIR, stem)
    done = recorded_groups(sub_dir)
    print(f'\n===== λ-sweep（固定 G=128，输入 {stem}）=====')
    for name, verts in LAMBDA_SWEEP:
        if name in done and not args.force:
            print(f'\n[skip] {name} 已在记录中存在，跳过（--force 强制重跑）')
            continue
        # 固定 FP8（使 Stage1 恒为 5.86 GB，提供刚性下限线）+ LOD128（λ 分母 G^3 固定）
        extra = ['--view', '6', '--FP8Unet', '--LOD', '128', '--ablation_name', name]
        if verts > 0:
            extra += ['--grid_extract_chunk', str(verts)]
        else:
            extra += ['--Chunk', '0']   # 整批（λ=1.0）
        run_group(name, extra, sub_dir, inp, args, done)
        done.add(name)
    print(f'\n[ok] λ-sweep 数据已写入 {sub_dir}（由 chart_ablation.py 生成 fig8_lambda_sweep）')


def main():
    p = argparse.ArgumentParser(description='Run N-input × 8-group ablation (含 Auto 自适应档) + relative eval + charts.')
    p.add_argument('--config', default=DEFAULT_CONFIG, help='run.py 用的 config 路径')
    p.add_argument('--input', default=DEFAULT_INPUT, help='单输入（旧用法；用 --inputs 做多物体）')
    p.add_argument('--inputs', default='', help='多输入：逗号分隔的文件列表，或某个图片目录')
    p.add_argument('--ablation_dir', default=DEFAULT_ABLATION_DIR, help='指标/mesh 记录父目录')
    p.add_argument('--charts_dir', default=DEFAULT_CHARTS_DIR, help='图表输出目录')
    p.add_argument('--tau', type=float, default=DEFAULT_TAU, help='F-Score 阈值 (归一化空间)')
    p.add_argument('--groups', default='', help='只跑这些组，逗号分隔；留空=全部 6 组')
    p.add_argument('--max_inputs', type=int, default=0, help='最多取前 N 个输入（0=全部）')
    p.add_argument('--force', action='store_true', help='忽略已记录，强制重跑')
    p.add_argument('--save_video', action='store_true', help='每组额外保存环绕视频')
    p.add_argument('--continue_on_error', action='store_true', help='某组失败也继续后续组')
    p.add_argument('--skip_eval', action='store_true', help='跑完推理后不调用评估脚本')
    p.add_argument('--skip_charts', action='store_true', help='最后不生成图表')
    p.add_argument('--lambda_sweep', action='store_true',
                   help='额外跑 λ-sweep 实验（固定 G=128 扫分块顶点数 g，对应 λ=g/128^3），'
                        '需 ≥16 GB 显存（RTX 4090）。数据写入 outputs/ablation_lambda_sweep/，'
                        '由 chart_ablation.py 生成 fig8_lambda_sweep 均衡曲线。')
    args = p.parse_args()

    inputs = parse_inputs(args)
    if not inputs:
        print('没有可用输入，退出。')
        return
    if args.max_inputs and args.max_inputs > 0:
        inputs = inputs[:args.max_inputs]
    print(f'共 {len(inputs)} 个输入：')
    for x in inputs:
        print('  -', x)

    if args.groups.strip():
        wanted = [g.strip() for g in args.groups.split(',') if g.strip()]
        selected = [(n, e) for (n, e) in GROUPS if n in wanted]
        missing = [w for w in wanted if w not in {n for n, _ in GROUPS}]
        if missing:
            print(f'[WARN] 以下组名无效（忽略）：{missing}')
    else:
        selected = list(GROUPS)
    if not selected:
        print('没有要跑的组，退出。')
        return

    # 每个输入单独进一个子目录，避免 jsonl 按组覆盖、并支持跨输入取均值±标准差。
    for inp in inputs:
        stem = os.path.splitext(os.path.basename(inp))[0]
        sub_dir = os.path.join(args.ablation_dir, stem)
        done = recorded_groups(sub_dir)
        if done and not args.force:
            print(f'\n=== 输入 {stem}：已记录组（将跳过）{sorted(done)} ===')
        for name, extra in selected:
            run_group(name, extra, sub_dir, inp, args, done)
            done.add(name)
        if not args.skip_eval:
            run_eval_relative(sub_dir, args.tau)
            run_eval_perceptual(sub_dir)

    # ── λ-sweep（支柱二·连续调节）：在 LOD64 主消融之后追加；仅取首个输入 ──
    if args.lambda_sweep:
        run_lambda_sweep(inputs, args)

    if not args.skip_charts:
        run_charts(args)

    print('\n===== 全部完成 =====')


if __name__ == '__main__':
    main()
