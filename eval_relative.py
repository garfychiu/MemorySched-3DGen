"""
以 Baseline（最高保真配置）生成的 mesh 为「参考锚点」，计算其余各组
相对 Baseline 的几何偏差：

    rel_cd      : 配置 X 的 mesh 与 Baseline mesh 的双向 Chamfer 距离
                  （归一化坐标空间下，越低 = 越接近 Baseline）
    rel_fscore  : F-Score@tau（配置 X vs Baseline，越高 = 越接近 Baseline）
                  Baseline 自身为 rel_cd=0 / rel_fscore=1.0

⚠ 诚实边界：这衡量的是「优化把几何从 Baseline 上推开了多远」，
   不是「重建相对真实物体的绝对误差」。论文中应表述为
   "geometric deviation from the unoptimized Baseline configuration"。

用法：
    python eval_relative.py --ablation_dir outputs/ablation [--tau 0.1] [--n_points 100000]

结果写回 outputs/ablation/run_metrics.jsonl（按组幂等更新：rel_cd / rel_fscore）。
随后 run.py --gen_charts 会自动聚合这些相对指标绘图。
"""

import argparse
import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def normalize_mesh(mesh):
    """
    将点云所在的 mesh 平移到包围盒中心，并缩放到最大边长为 1。
    这样 tau 在 [0,1] 空间里有明确物理意义（物体尺寸的占比）。
    """
    verts = mesh.vertices
    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)
    center = (bb_min + bb_max) / 2.0
    extent = float((bb_max - bb_min).max())
    if extent <= 0:
        return mesh
    mesh = mesh.copy()
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / extent)
    return mesh


def sample_surface_points(mesh, n_points):
    """在 mesh 表面按面积均匀采样 n_points 个点。"""
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(points, dtype=np.float64)


def chamfer(a, b):
    """对称 Chamfer 距离：mean_a(min_b) + mean_b(min_a)。"""
    ta = cKDTree(a)
    tb = cKDTree(b)
    d_a, _ = tb.query(a)
    d_b, _ = ta.query(b)
    return float(d_a.mean() + d_b.mean()) / 2.0


def fscore(a, b, tau):
    """F-Score@tau：precision = 预测点中有多少落在 GT 的 tau 邻域；
    recall = GT 点中有多少落在预测的 tau 邻域。"""
    ta = cKDTree(a)
    tb = cKDTree(b)
    d_a, _ = tb.query(a)  # 每个 a 到最近 b 的距离
    d_b, _ = ta.query(b)  # 每个 b 到最近 a 的距离
    precision = float((d_a < tau).mean())
    recall = float((d_b < tau).mean())
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def load_group_meshes(mesh_dir, n_points):
    """
    读入某组目录下的所有 *.obj，返回 {filename: (normalized_points, raw_mesh)}。
    只保留能成功加载且面数 > 0 的 mesh。
    """
    result = {}
    if not os.path.isdir(mesh_dir):
        return result
    for fn in sorted(os.listdir(mesh_dir)):
        if not fn.lower().endswith('.obj'):
            continue
        path = os.path.join(mesh_dir, fn)
        try:
            mesh = trimesh.load(path, process=False, force='mesh')
            if getattr(mesh, 'faces', None) is None or len(mesh.faces) == 0:
                print(f'  [warn] skip {path}: empty/invalid mesh')
                continue
            mesh = normalize_mesh(mesh)
            result[fn] = sample_surface_points(mesh, n_points)
        except Exception as e:
            print(f'  [warn] failed to load {path}: {e}')
    return result


def main():
    parser = argparse.ArgumentParser(description='Relative quality vs Baseline (geometric deviation).')
    parser.add_argument('--ablation_dir', type=str, default='outputs/ablation',
                        help='Directory with run_metrics.jsonl + meshes/<group>/')
    parser.add_argument('--ref_group', type=str, default='Baseline',
                        help='Group name used as the reference anchor.')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='F-Score threshold in normalized [0,1] space (user chose 0.1).')
    parser.add_argument('--n_points', type=int, default=100000,
                        help='Surface sample points per mesh for Chamfer/F-Score.')
    args = parser.parse_args()

    jsonl_path = os.path.join(args.ablation_dir, 'run_metrics.jsonl')
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f'No run_metrics.jsonl at {jsonl_path}. Run run.py first.')

    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not records:
        raise RuntimeError('run_metrics.jsonl is empty.')

    by_group = {r['group']: r for r in records}

    if args.ref_group not in by_group:
        raise KeyError(f'Reference group "{args.ref_group}" not found. '
                       f'Available: {list(by_group.keys())}')
    ref = by_group[args.ref_group]
    ref_dir = ref.get('mesh_dir')
    if not ref_dir or not os.path.isdir(ref_dir):
        raise FileNotFoundError(f'Baseline mesh_dir missing: {ref_dir}')

    print(f'Loading Baseline meshes from: {ref_dir}')
    ref_meshes = load_group_meshes(ref_dir, args.n_points)
    if not ref_meshes:
        raise RuntimeError('No valid .obj found for Baseline.')
    print(f'  Baseline inputs: {list(ref_meshes.keys())}')

    # 预加载所有组的 mesh 点云
    group_points = {}
    for g, r in by_group.items():
        d = r.get('mesh_dir')
        if not d or not os.path.isdir(d):
            print(f'  [warn] group "{g}" has no mesh_dir, skipped in relative eval.')
            continue
        group_points[g] = load_group_meshes(d, args.n_points)

    # 计算相对指标（按输入文件名对齐 Baseline）
    new_metrics = {}  # group -> (rel_cd, rel_fscore)
    # Baseline 自身
    new_metrics[args.ref_group] = (0.0, 1.0)

    for g, meshes in group_points.items():
        if g == args.ref_group:
            continue
        cds, fscores = [], []
        for fn, pts in meshes.items():
            if fn not in ref_meshes:
                print(f'  [warn] group "{g}" input {fn} has no Baseline counterpart, skipped.')
                continue
            ref_pts = ref_meshes[fn]
            cds.append(chamfer(pts, ref_pts))
            fscores.append(fscore(pts, ref_pts, args.tau))
        if not cds:
            print(f'  [warn] group "{g}" matched no inputs with Baseline, skipped.')
            continue
        mean_cd = float(np.mean(cds))
        mean_fs = float(np.mean(fscores))
        new_metrics[g] = (mean_cd, mean_fs)
        print(f'  {g:14s} rel_cd={mean_cd:.4f}  rel_fscore@{args.tau}={mean_fs:.4f}')

    # 幂等写回 jsonl
    for r in records:
        if r['group'] in new_metrics:
            cd, fs = new_metrics[r['group']]
            r['rel_cd'] = cd
            r['rel_fscore'] = fs
            r['rel_tau'] = args.tau

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'\nWrote relative metrics (rel_cd / rel_fscore) for '
          f'{len(new_metrics)} groups into {jsonl_path}')
    print('Next: run  python run.py configs/instant-mesh-large.yaml <img> --gen_charts')
    print('       (or any run.py invocation with --gen_charts) to plot the relative charts.')


if __name__ == '__main__':
    main()
