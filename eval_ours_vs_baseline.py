"""
评测「Ours」与「Baseline」两组 mesh，并输出逐项对比表。

前提：pred_root（默认 outputs/ablation/meshes）下需同时存在 Ours/ 与 Baseline/ 两个子目录，
      分别由 run_ours_gso.py（本地 8G）与 run_baseline_gso.py（服务器 24G）生成。
      两组内部都是 <oid>.obj，eval_absolute.py 会按 stem 对齐 <gt_dir>/<oid>/meshes/model.obj（gso 布局）。

      注意：Ours 在本地、Baseline 在服务器，评测前需要把两边 mesh 合并到同一台机器的
      outputs/ablation/meshes/ 下（建议放到 24G 服务器上跑，显存/速度都更稳）。

用法：
    # 默认（ICP 配准开、LPIPS 开）
    python eval_ours_vs_baseline.py --gt_dir D:\AR_3DGeneres\InstantMesh\inputs\GSO_300

    # 指定预测根目录 / 输出
    python eval_ours_vs_baseline.py --gt_dir <GT> --pred_root outputs/ablation/meshes

    # 关 ICP 配准 / 关 LPIPS（离线无 VGG 时）
    python eval_ours_vs_baseline.py --gt_dir <GT> --no_icp
    python eval_ours_vs_baseline.py --gt_dir <GT> --no_lpips

输出：
    outputs/absolute/metrics.json      聚合指标（含 Ours / Baseline 两组）
    outputs/absolute/per_object.csv    逐物体明细
    控制台额外打印 Ours vs Baseline 的 CD / FS@0.2 / FS@0.1 / LPIPS 对比与差值
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

DEFAULT_PRED_ROOT = os.path.join('outputs', 'ablation', 'meshes')
DEFAULT_OUT = os.path.join('outputs', 'absolute', 'metrics.json')
DEFAULT_CSV = os.path.join('outputs', 'absolute', 'per_object.csv')
DEFAULT_GT_DIR = r'D:\AR_3DGeneres\InstantMesh\inputs\GSO_300'
EXPECTED_GROUPS = ['Ours', 'Baseline']


def build_cmd(args):
    cmd = [PY, 'eval_absolute.py',
           '--gt_dir', args.gt_dir,
           '--dataset', args.dataset,
           '--pred_root', args.pred_root,
           '--out', args.out,
           '--csv', args.csv,
           '--tau', str(args.tau),
           '--tau_extra', str(args.tau_extra),
           '--n_points', str(args.n_points)]
    if args.icp:
        cmd.append('--icp')
    if args.no_lpips:
        cmd.append('--no_lpips')
    return cmd


def print_comparison(metrics_json):
    """读取 metrics.json，打印 Ours vs Baseline 对比表。"""
    if not os.path.exists(metrics_json):
        print('[WARN] 未找到 metrics.json，跳过对比表。')
        return
    with open(metrics_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    groups = {g['group']: g for g in data.get('groups', [])}
    if not groups:
        print('[WARN] metrics.json 中没有组结果，跳过对比表。')
        return

    tau = data.get('protocol', {}).get('tau', 0.2)
    tau_e = data.get('protocol', {}).get('tau_extra', 0.1)

    def gv(group, key, default='-'):
        return groups.get(group, {}).get(key, default)

    def fmt_mean_std(group, base):
        m = gv(group, f'{base}_mean', None)
        s = gv(group, f'{base}_std', None)
        if m is None:
            return '-'
        return f'{m:.4f}' + (f'±{s:.4f}' if s is not None else '')

    rows = [
        ('CD×1e3 (↓)', 'cd_x1e3'),
        (f'F-Score@{tau} (↑)', f'fscore@{tau}'),
        (f'F-Score@{tau_e} (↑)', f'fscore@{tau_e}'),
        ('LPIPS vs GT (↓)', 'lpips_vs_gt'),
    ]

    present = [g for g in EXPECTED_GROUPS if g in groups]
    print('\n================ Ours vs Baseline 对比 ================')
    header = f'{"指标":<18}' + ''.join(f'{g:>26}' for g in present)
    if len(present) == 2:
        header += f'{"Δ(Ours-Base)":>18}'
    print(header)
    print('-' * len(header))
    for label, base in rows:
        line = f'{label:<18}'
        vals = {}
        for g in present:
            v = gv(g, f'{base}_mean', None)
            vals[g] = v
            line += f'{fmt_mean_std(g, base):>26}'
        if len(present) == 2 and vals.get('Ours') is not None and vals.get('Baseline') is not None:
            delta = vals['Ours'] - vals['Baseline']
            arrow = '↓好' if '↓' in label else '↑好'
            line += f'{delta:+.4f} ({arrow})'.rjust(18)
        print(line)
    n_ours = gv('Ours', 'n', '-')
    n_base = gv('Baseline', 'n', '-')
    print('-' * len(header))
    print(f'样本数: Ours={n_ours}  Baseline={n_base}')
    print('说明: CD×1e3 / LPIPS 越低越好；F-Score 越高越好。Δ 为 Ours 减 Baseline。')


def main():
    p = argparse.ArgumentParser(description='评测并对比 Ours vs Baseline。')
    p.add_argument('--gt_dir', default=DEFAULT_GT_DIR, help='GT 根目录（gso 布局：<gt_dir>/<oid>/meshes/model.obj）')
    p.add_argument('--pred_root', default=DEFAULT_PRED_ROOT, help='预测根目录（含 Ours/ 与 Baseline/ 子目录）')
    p.add_argument('--dataset', default='gso', choices=['gso', 'omni3d', 'flat'])
    p.add_argument('--out', default=DEFAULT_OUT, help='聚合指标输出 json')
    p.add_argument('--csv', default=DEFAULT_CSV, help='逐物体明细 csv')
    p.add_argument('--tau', type=float, default=0.2, help='F-Score 主阈值')
    p.add_argument('--tau_extra', type=float, default=0.1, help='F-Score 附加阈值')
    p.add_argument('--n_points', type=int, default=100000, help='表面采样点数')
    p.add_argument('--icp', action='store_true', default=True, help='评分前刚性 ICP 配准（默认开）')
    p.add_argument('--no_icp', dest='icp', action='store_false', help='关闭 ICP 配准')
    p.add_argument('--no_lpips', action='store_true', help='跳过 LPIPS（离线无 VGG 时）')
    args = p.parse_args()

    if not os.path.isdir(args.pred_root):
        print(f'[ERROR] pred_root 不存在：{args.pred_root}')
        sys.exit(1)

    found = sorted(d for d in os.listdir(args.pred_root)
                   if os.path.isdir(os.path.join(args.pred_root, d)))
    print(f'pred_root 下发现的组：{found}')
    missing = [g for g in EXPECTED_GROUPS if g not in found]
    if missing:
        print(f'[WARN] 缺少对比组 {missing}（只评存在的组）。合并 Ours 与 Baseline 后再跑可出完整对比。')
    if not found:
        print('[ERROR] pred_root 下没有任何组子目录，退出。')
        sys.exit(1)

    cmd = build_cmd(args)
    print('\n执行：', ' '.join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        print(f'[ERROR] eval_absolute.py 返回码 {rc}')
        sys.exit(rc)

    print_comparison(args.out)


if __name__ == '__main__':
    main()
