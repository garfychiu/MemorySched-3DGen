"""
绝对几何 + 感知评测（以 GT 真值 mesh 为锚点）。

本脚本替换 eval_relative.py / eval_perceptual.py 的「相对 Baseline」框架。
那两个脚本把未优化的 Baseline 当锚点，算出的"相对 CD / F-Score 保留率 /
LPIPS↓(vs Base)"都不是相对真值的绝对误差——这正是审稿人指出的致命伤。

这里改以真实 mesh 为锚点，所有数字都是**绝对值**，可直接对标
InstantMesh 原论文（arXiv:2404.07191, Table 2）以及任何在
GSO / OmniObject3D 上的方法。

──────────────────────── 协议（与 InstantMesh/Instant3D 对齐） ────────────────────────
  * 预测 mesh 与 GT mesh 均：平移到包围盒中心、缩放到最大边长为 1（unit bbox）。
    此归一化下 InstantMesh 级质量的 CD 约为 0.18，报告为 CD×10³ ≈ 180。
  * Chamfer Distance (CD)：对称最近邻平均距离，报 CD×10³（即原值 × 1000）。
  * F-Score@tau：precision = 预测点落在 GT 的 tau 邻域比例；
                  recall   = GT 点落在预测 tau 邻域比例；F = 2PR/(P+R)。
                  默认 tau=0.2（InstantMesh 论文阈值）；另报 tau=0.1（审稿人要求）。
  * 可选刚性 ICP 对齐（--icp）：评分前把预测点云刚性配准到 GT，提升可比性。
  * 绝对 LPIPS：把预测 mesh 与 GT mesh 从**同一组固定新视角**渲染，前景裁齐后
    用 VGG 特征算感知距离（pred vs GT 渲染）。即审稿人要的"绝对 LPIPS"
    （对应 InstantMesh Table 2 的 LPIPS vs GT 新视角）。

──────────────────────── 目录约定 ────────────────────────
  GT:    --gt_dir 下按数据集布局自动发现 (--dataset gso|omni3d|flat)：
           gso    : <gt_dir>/<object_id>/meshes/model.obj
           omni3d : <gt_dir>/<category>/<object>/mesh/model.obj (或 raw_model.obj)
           flat   : <gt_dir>/<object_id>.obj   （也可直接指定 --gt_manifest）
        也可用 --gt_manifest <csv|json> 显式映射 object_id -> gt 路径（最稳妥）。
  Pred:  --pred_root 下：
           <pred_root>/<group>/<object_id>.obj   多个组（如 Baseline / Ours）
           或 <pred_root>/<object_id>.obj         单组（命名为 "Ours"）
  object_id 在 GT 与 Pred 间按文件名 stem 对齐。

──────────────────────── 用法 ────────────────────────
  # 完整（需先准备 GSO/Omni3D 真值与预测 mesh）
  python eval_absolute.py \\
      --gt_dir /data/GSO --dataset gso \\
      --pred_root outputs/gso_pred --tau 0.2 --tau_extra 0.1 \\
      --out outputs/absolute/metrics.json --csv outputs/absolute/per_object.csv

  # 冒烟自测（无需任何数据集，验证代码路径）
  python eval_absolute.py --demo
"""

import argparse
import csv
import json
import os
import sys

# 让 Windows GBK 控制台也能正常打印 ×/³/↓/↑/τ 等符号（不影响文件输出，文件始终 utf-8）
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree


# ───────────────────────── 几何：归一化 / 采样 / CD / F-Score ─────────────────────────
def normalize_verts(verts):
    """平移到包围盒中心、缩放到最大边长为 1（unit bbox）。"""
    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)
    center = (bb_min + bb_max) / 2.0
    extent = float((bb_max - bb_min).max())
    if extent <= 0:
        return verts
    return (verts - center) / extent


def sample_surface_points(mesh, n_points):
    """在 mesh 表面按面积均匀采样 n_points 个点。"""
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(points, dtype=np.float64)


def chamfer(a, b):
    """对称 Chamfer 距离：0.5 * (mean_a(min_b) + mean_b(min_a))。"""
    ta = cKDTree(a)
    tb = cKDTree(b)
    d_a, _ = tb.query(a)
    d_b, _ = ta.query(b)
    return float(d_a.mean() + d_b.mean()) / 2.0


def fscore(a, b, tau):
    """F-Score@tau。"""
    ta = cKDTree(a)
    tb = cKDTree(b)
    d_a, _ = tb.query(a)
    d_b, _ = ta.query(b)
    precision = float((d_a < tau).mean())
    recall = float((d_b < tau).mean())
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def icp_align(src_pts, tgt_pts, max_iter=50):
    """把 src 刚性配准到 tgt，返回 4x4 变换（src->tgt）。失败返回单位阵。"""
    # 版本兼容说明（针对 trimesh 4.x + scipy 新版）：
    #  - trimesh 4.0.x 的 icp() 返回 (matrix, transformed, cost) 元组，
    #    第 0 个元素才是 4x4 变换；旧版曾返回含 'transformation' 键的 dict。
    #  - 其底层把多余 kwargs 透传给 scipy.procrustes()，而新版 scipy
    #    已移除 tolerance= 等参数，传了会 TypeError。
    # 因此：用 max_iterations（两版都接受），并兼容 dict/tuple 两种返回。
    try:
        res = trimesh.registration.icp(src_pts, tgt_pts, max_iterations=max_iter)
    except Exception as e:
        print(f'    [warn] ICP 失败，回退到无配准: {e}')
        return np.eye(4)

    if isinstance(res, dict):
        T = res.get('transformation')
    else:  # trimesh 4.x 返回 (matrix, transformed, cost)
        T = res[0]
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        print('    [warn] ICP 返回矩阵形状异常，回退到无配准。')
        return np.eye(4)
    return T


# ───────────────────────── mesh 载入（几何 + 可选顶点色） ─────────────────────────
def _parse_obj(path):
    verts, colors, faces = [], [], []
    with open(path, 'r', encoding='utf-8', errors='replace') as fid:
        for line in fid:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                coords = [float(x) for x in parts[1:]]
                verts.append(coords[:3])
                if len(coords) >= 6:
                    colors.append(coords[3:6])
            elif parts[0] == 'f':
                idx = []
                for tok in parts[1:]:
                    vi = int(tok.split('/')[0])
                    idx.append(vi - 1)
                if len(idx) >= 3:
                    faces.append(idx[:3])
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    C = np.asarray(colors, dtype=np.float64) if colors else None
    if C is not None and C.shape[0] == V.shape[0] and C.max() > 1.0:
        C = C / 255.0
    return V, F, (C if (C is not None and C.shape[0] == V.shape[0]) else None)


def load_mesh_full(path):
    """返回 (V, F, C)：V:(V,3) float64, F:(F,3) int64, C:(V,3)∈[0,1] 或 None。
    兼容文本 OBJ（含内联顶点色）与二进制 glb/ply（trimesh 嗅探）。"""
    try:
        V, F, C = _parse_obj(path)
        if len(F) > 0:
            return V, F, C
    except Exception:
        pass
    try:
        mesh = trimesh.load(path, process=False, force='mesh')
    except Exception as e:
        print(f'  [warn] 载入失败 {path}: {e}')
        return None, None, None
    faces = getattr(mesh, 'faces', None)
    if faces is None or len(faces) == 0:
        return None, None, None
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    C = None
    vis = getattr(mesh, 'visual', None)
    if vis is not None and getattr(vis, 'vertex_colors', None) is not None:
        vc = np.asarray(vis.vertex_colors, dtype=np.float64)
        if vc.ndim == 2 and vc.shape[1] >= 3:
            C = vc[:, :3]
            if C.max() > 1.0:
                C = C / 255.0
    return V, F, C


def load_gt_points_and_mesh(gt_path, n_points):
    """载入 GT：返回 (points_unitbbox, render_mesh_dict)，失败返回 (None, None)。"""
    V, F, C = load_mesh_full(gt_path)
    if V is None or F is None or len(F) == 0:
        return None, None
    Vn = normalize_verts(V)
    C = C if C is not None else np.ones_like(Vn)
    tm = trimesh.Trimesh(vertices=Vn, faces=F, process=False)
    pts = sample_surface_points(tm, n_points)
    return pts, {'v': Vn, 'f': F, 'c': C}


def load_pred_points_and_mesh(pred_path, n_points):
    """载入预测 mesh：返回归一化点云 + 渲染字典（同 GT）。"""
    V, F, C = load_mesh_full(pred_path)
    if V is None or F is None or len(F) == 0:
        return None, None
    Vn = normalize_verts(V)
    C = C if C is not None else np.ones_like(Vn)
    tm = trimesh.Trimesh(vertices=Vn, faces=F, process=False)
    pts = sample_surface_points(tm, n_points)
    return pts, {'v': Vn, 'f': F, 'c': C}


# ───────────────────────── 渲染（用于绝对 LPIPS） ─────────────────────────
def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * near * far) / (near - far)],
        [0, 0, -1, 0],
    ], dtype=np.float32)


def circular_cameras(n_views=12, radius=2.2, elevation=20.0):
    elev = np.deg2rad(elevation)
    c2ws = []
    for i in range(n_views):
        az = 2 * np.pi * i / n_views
        pos = np.array([
            radius * np.cos(elev) * np.cos(az),
            radius * np.cos(elev) * np.sin(az),
            radius * np.sin(elev),
        ], dtype=np.float32)
        z_axis = pos / np.linalg.norm(pos)
        up = np.array([0, 0, 1], dtype=np.float32)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        extr = np.eye(4, dtype=np.float32)
        extr[:3, 0] = x_axis
        extr[:3, 1] = y_axis
        extr[:3, 2] = z_axis
        extr[:3, 3] = pos
        c2ws.append(extr)
    return np.linalg.inv(np.stack(c2ws, axis=0))


def render_views(mesh, cam_mvs, res=256, ctx=None, use_cuda=True):
    if use_cuda:
        return _render_views_cuda(mesh, cam_mvs, res=res, ctx=ctx)
    return _render_views_cpu(mesh, cam_mvs, res=res)


def _render_views_cuda(mesh, cam_mvs, res=256, ctx=None):
    import torch
    import torch.nn.functional as F
    import nvdiffrast.torch as dr
    if ctx is None:
        ctx = dr.RasterizeCudaContext()
    v = torch.from_numpy(mesh['v']).float().cuda().unsqueeze(0)
    c = torch.from_numpy(mesh['c']).float().cuda().unsqueeze(0)
    f = torch.from_numpy(mesh['f']).int().cuda().contiguous()
    proj = perspective(30.0, 1.0, 0.1, 100.0)
    proj_t = torch.from_numpy(proj).float().cuda()
    cam_mvs_t = torch.from_numpy(cam_mvs).float().cuda()
    imgs = []
    with torch.inference_mode():
        for i in range(cam_mvs_t.shape[0]):
            mvp = proj_t @ cam_mvs_t[i]
            vh = F.pad(v, (0, 1), value=1.0)
            v_clip = (mvp @ vh.transpose(1, 2)).transpose(1, 2).contiguous()
            rast = dr.rasterize(ctx, v_clip, f, resolution=[res, res])[0]
            color, _ = dr.interpolate(c, rast, f.contiguous())
            alpha = (rast[..., 3:4] > 0).float()
            img = color * alpha + (1.0 - alpha)
            imgs.append(img.permute(0, 3, 1, 2).clamp(0, 1).cpu())
    return torch.cat(imgs, dim=0)


def _subsample_faces(v, f, c, max_faces=15000):
    if f.shape[0] <= max_faces:
        return v, f, c
    rng = np.random.default_rng(0)
    idx = rng.choice(f.shape[0], max_faces, replace=False)
    return v, f[idx], c


def _render_views_cpu(mesh, cam_mvs, res=256):
    import torch
    v0, f0, c0 = _subsample_faces(mesh['v'], mesh['f'], mesh['c'], max_faces=15000)
    v = np.asarray(v0, dtype=np.float64)
    f = np.asarray(f0, dtype=np.int64)
    c = np.asarray(c0, dtype=np.float64)
    P = perspective(30.0, 1.0, 0.1, 100.0)
    imgs = []
    for mv in cam_mvs:
        mv = np.asarray(mv, dtype=np.float64)
        vh = np.concatenate([v, np.ones((v.shape[0], 1))], axis=1)
        view = (mv @ vh.T).T
        clip = (P @ view.T).T
        w = clip[:, 3]
        ndc = clip[:, :3] / w[:, None]
        px = (ndc[:, 0] * 0.5 + 0.5) * (res - 1)
        py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (res - 1)
        pz = -view[:, 2]
        img = np.ones((res, res, 3), dtype=np.float64)
        zbuf = np.full((res, res), -1.0, dtype=np.float64)
        xs = px[f]; ys = py[f]; zs = pz[f]; ws = w[f]; cs = c[f]
        denom = (ys[:, 1] - ys[:, 2]) * (xs[:, 0] - xs[:, 2]) + \
                (xs[:, 2] - xs[:, 1]) * (ys[:, 0] - ys[:, 2])
        valid = np.abs(denom) > 1e-12
        for tri in range(f.shape[0]):
            if not valid[tri]:
                continue
            xa, xb_, xc = xs[tri]; ya, yb_, yc = ys[tri]
            minx = max(int(np.floor(min(xa, xb_, xc))), 0)
            maxx = min(int(np.ceil(max(xa, xb_, xc))), res - 1)
            miny = max(int(np.floor(min(ya, yb_, yc))), 0)
            maxy = min(int(np.ceil(max(ya, yb_, yc))), res - 1)
            if maxx < minx or maxy < miny:
                continue
            gx, gy = np.meshgrid(np.arange(minx, maxx + 1),
                                 np.arange(miny, maxy + 1), indexing='xy')
            d = denom[tri]
            l0 = ((yb_ - yc) * (gx - xc) + (xc - xb_) * (gy - yc)) / d
            l1 = ((yc - ya) * (gx - xc) + (xa - xc) * (gy - yc)) / d
            l2 = 1.0 - l0 - l1
            m = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not m.any():
                continue
            iw = 1.0 / ws[tri]
            iz = l0 * iw[0] + l1 * iw[1] + l2 * iw[2]
            zp = l0 * zs[tri, 0] + l1 * zs[tri, 1] + l2 * zs[tri, 2]
            col_r = (l0 * cs[tri, 0, 0] / ws[tri, 0] + l1 * cs[tri, 1, 0] / ws[tri, 1] +
                     l2 * cs[tri, 2, 0] / ws[tri, 2]) / iz
            col_g = (l0 * cs[tri, 0, 1] / ws[tri, 0] + l1 * cs[tri, 1, 1] / ws[tri, 1] +
                     l2 * cs[tri, 2, 1] / ws[tri, 2]) / iz
            col_b = (l0 * cs[tri, 0, 2] / ws[tri, 0] + l1 * cs[tri, 1, 2] / ws[tri, 1] +
                     l2 * cs[tri, 2, 2] / ws[tri, 2]) / iz
            col = np.stack([col_r, col_g, col_b], axis=-1)
            gxf = gx[m]; gyf = gy[m]
            zpf = zp[m]; colf = col[m]
            cur = zbuf[gyf, gxf]
            up = zpf > cur
            if up.any():
                zbuf[gyf[up], gxf[up]] = zpf[up]
                img[gyf[up], gxf[up]] = colf[up]
        imgs.append(img.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(imgs, axis=0).astype(np.float32))


def crop_aligned(ref, cand, size=224, margin=0.08):
    import torch
    import torch.nn.functional as F
    def _crop_one(x):
        H, W = x.shape[-2], x.shape[-1]
        fg = (x < 0.98).any(dim=0)
        if not fg.any():
            return x.unsqueeze(0)
        ys, xs = torch.where(fg)
        y0, y1 = ys.min().item(), ys.max().item()
        x0, x1 = xs.min().item(), xs.max().item()
        mh = max(int((y1 - y0) * margin), 1)
        mw = max(int((x1 - x0) * margin), 1)
        y0 = max(0, y0 - mh); y1 = min(H - 1, y1 + mh)
        x0 = max(0, x0 - mw); x1 = min(W - 1, x1 + mw)
        return x[:, y0:y1 + 1, x0:x1 + 1].unsqueeze(0)
    cr = F.interpolate(_crop_one(ref), (size, size), mode='bilinear', align_corners=False)
    cc = F.interpolate(_crop_one(cand), (size, size), mode='bilinear', align_corners=False)
    return cr, cc


def get_vgg(device):
    try:
        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        layers = [3, 8, 15, 22, 29]
        print('  [ok] VGG16 已加载 -> 计算绝对 LPIPS。')
        return vgg, layers
    except Exception as e:
        print(f'  [skip] VGG16 不可用（离线？）LPIPS 省略: {e}')
        return None, None


@torch.no_grad()
def lpips_views(imgs1, imgs2, vgg, layers, device):
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def features(x):
        x = (x - 0.5) / 0.5
        x = (x - mean) / std
        out = []
        for i, layer in enumerate(vgg.children()):
            x = layer(x)
            if i in layers:
                out.append(x)
        return out

    f1 = features(imgs1.to(device))
    f2 = features(imgs2.to(device))
    dist = torch.zeros(imgs1.shape[0], device=device)
    for a, b in zip(f1, f2):
        a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
        b = b / (b.norm(dim=1, keepdim=True) + 1e-8)
        d = (a - b).pow(2).mean(dim=(2, 3)).mean(dim=1)
        dist += d
    return float(dist.mean().item())


# ───────────────────────── GT 发现 ─────────────────────────
def _first_file(obj_dir, candidates):
    """在 obj_dir 下按候选相对路径或递归查找第一个匹配文件。"""
    for rel in candidates:
        p = os.path.join(obj_dir, rel)
        if os.path.exists(p):
            return p
    for root, _, files in os.walk(obj_dir):
        for fl in files:
            if fl.lower() in ('model.obj', 'model_mesh.obj'):
                return os.path.join(root, fl)
    return None


def discover_gt(gt_dir, dataset, manifest=None):
    if manifest:
        gt = {}
        if manifest.endswith('.csv'):
            with open(manifest, 'r', encoding='utf-8') as f:
                for row in csv.reader(f):
                    if len(row) >= 2 and not row[0].startswith('#'):
                        gt[row[0].strip()] = row[1].strip()
        else:
            with open(manifest, 'r', encoding='utf-8') as f:
                gt = {k: v for k, v in json.load(f).items()}
        return gt
    gt = {}
    if dataset == 'gso':
        # 兼容两种官方解压布局：<id>/meshes/model.obj 与 <id>/1/meshes/model.obj
        for d in sorted(os.listdir(gt_dir)):
            obj_dir = os.path.join(gt_dir, d)
            if not os.path.isdir(obj_dir):
                continue
            p = _first_file(obj_dir, ['meshes/model.obj', '1/meshes/model.obj', 'model.obj'])
            if p:
                gt[d] = p
    elif dataset == 'omni3d':
        for cat in sorted(os.listdir(gt_dir)):
            cpath = os.path.join(gt_dir, cat)
            if not os.path.isdir(cpath):
                continue
            for obj in sorted(os.listdir(cpath)):
                opath = os.path.join(cpath, obj)
                if not os.path.isdir(opath):
                    continue
                cand = None
                for root, _, files in os.walk(opath):
                    for fl in files:
                        if fl.lower() in ('model.obj', 'raw_model.obj', 'mesh.obj'):
                            cand = os.path.join(root, fl)
                            break
                    if cand:
                        break
                if cand:
                    gt[f'{cat}_{obj}'] = cand
    else:  # flat / auto
        for fl in sorted(os.listdir(gt_dir)):
            if fl.lower().endswith(('.obj', '.glb', '.ply')):
                gt[os.path.splitext(fl)[0]] = os.path.join(gt_dir, fl)
    if not gt:
        raise RuntimeError(f'在 {gt_dir} (dataset={dataset}) 未发现任何 GT mesh。')
    return gt


def discover_pred_groups(pred_root):
    groups = {}
    entries = [e for e in sorted(os.listdir(pred_root))
               if os.path.isdir(os.path.join(pred_root, e))]
    if entries:
        for g in entries:
            files = {}
            for fl in sorted(os.listdir(os.path.join(pred_root, g))):
                if fl.lower().endswith(('.obj', '.glb', '.ply')):
                    files[os.path.splitext(fl)[0]] = os.path.join(pred_root, g, fl)
            if files:
                groups[g] = files
    else:
        files = {}
        for fl in sorted(os.listdir(pred_root)):
            if fl.lower().endswith(('.obj', '.glb', '.ply')):
                files[os.path.splitext(fl)[0]] = os.path.join(pred_root, fl)
        if files:
            groups['Ours'] = files
    if not groups:
        raise RuntimeError(f'在 {pred_root} 未发现任何预测 mesh。')
    return groups


# ───────────────────────── 主流程 ─────────────────────────
def evaluate_group(group, pred_files, gt_map, args, device, cam_mvs, ctx, use_cuda, vgg, vgg_layers):
    per_obj = []
    for oid, pred_path in pred_files.items():
        if oid not in gt_map:
            print(f'    [warn] {group}/{oid}: 无对应 GT，跳过。')
            continue
        gt_pts, gt_render = load_gt_points_and_mesh(gt_map[oid], args.n_points)
        pred_pts, pred_render = load_pred_points_and_mesh(pred_path, args.n_points)
        if gt_pts is None or pred_pts is None:
            print(f'    [warn] {group}/{oid}: mesh 载入失败，跳过。')
            continue

        if args.icp:
            T = icp_align(pred_pts, gt_pts)
            pred_pts = (T[:3, :3] @ pred_pts.T + T[:3, 3:4]).T
            v = pred_render['v']
            pred_render['v'] = (T[:3, :3] @ v.T + T[:3, 3:4]).T

        cd = chamfer(pred_pts, gt_pts)
        fs_tau = fscore(pred_pts, gt_pts, args.tau)
        fs_extra = fscore(pred_pts, gt_pts, args.tau_extra)

        lp = None
        if vgg is not None and gt_render is not None:
            gr = render_views(gt_render, cam_mvs, res=args.res, ctx=ctx, use_cuda=use_cuda)
            cr = render_views(pred_render, cam_mvs, res=args.res, ctx=ctx, use_cuda=use_cuda)
            gr = gr.to(device); cr = cr.to(device)
            cr_list, cc_list = [], []
            for i in range(gr.shape[0]):
                a, b = crop_aligned(gr[i], cr[i], size=args.crop_size, margin=0.08)
                cr_list.append(a); cc_list.append(b)
            lp = lpips_views(torch.cat(cr_list, 0), torch.cat(cc_list, 0), vgg, vgg_layers, device)

        rec = {'object_id': oid, 'cd_x1e3': cd * 1000.0,
               f'fscore@{args.tau}': fs_tau, f'fscore@{args.tau_extra}': fs_extra,
               'lpips_vs_gt': lp}
        per_obj.append(rec)
        print(f'    {oid:16s} CD×10³={cd*1000:7.2f}  '
              f'FS@{args.tau}={fs_tau:.3f}  FS@{args.tau_extra}={fs_extra:.3f}  '
              f'LPIPS={("%.4f" % lp) if lp is not None else " n/a"}')

    if not per_obj:
        return None
    cds = np.array([r['cd_x1e3'] for r in per_obj])
    fs1 = np.array([r[f'fscore@{args.tau}'] for r in per_obj])
    fs2 = np.array([r[f'fscore@{args.tau_extra}'] for r in per_obj])
    lps = np.array([r['lpips_vs_gt'] for r in per_obj if r['lpips_vs_gt'] is not None])
    agg = {
        'group': group,
        'n': len(per_obj),
        'cd_x1e3_mean': float(cds.mean()), 'cd_x1e3_std': float(cds.std()),
        f'fscore@{args.tau}_mean': float(fs1.mean()), f'fscore@{args.tau}_std': float(fs1.std()),
        f'fscore@{args.tau_extra}_mean': float(fs2.mean()), f'fscore@{args.tau_extra}_std': float(fs2.std()),
        'lpips_vs_gt_mean': float(lps.mean()) if len(lps) else None,
        'lpips_vs_gt_std': float(lps.std()) if len(lps) else None,
    }
    return {'agg': agg, 'per_object': per_obj}


def run_demo(args):
    """冒烟自测：合成 GT（单位球）+ 轻微扰动的预测，验证整条代码路径。
    数字仅供验证，非真实评测结果。"""
    print('=' * 60)
    print('DEMO 模式（合成数据，仅验证代码路径，非真实结果）')
    print('=' * 60)
    gt_mesh = trimesh.creation.icosphere(subdivisions=3)
    gt_pts = sample_surface_points(gt_mesh, args.n_points)
    # 预测：球放大 1.02 + 轻微噪声
    pred_pts = gt_pts * 1.02 + np.random.default_rng(0).normal(0, 0.01, gt_pts.shape)
    cd = chamfer(pred_pts, gt_pts) * 1000.0
    fs = fscore(pred_pts, gt_pts, args.tau)
    fs2 = fscore(pred_pts, gt_pts, args.tau_extra)
    print(f'  CD×10³={cd:.2f}  FS@{args.tau}={fs:.3f}  FS@{args.tau_extra}={fs2:.3f}')
    print('  [ok] 几何指标路径正常。')

    # 渲染 + LPIPS 路径自测：同一带色 mesh 自比应 ≈ 0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vgg, vgg_layers = get_vgg(device)
    if vgg is not None:
        V = np.asarray(gt_mesh.vertices, dtype=np.float64)
        F = np.asarray(gt_mesh.faces, dtype=np.int64)
        C = np.ones_like(V) * 0.6
        cam_mvs = circular_cameras(n_views=args.n_views)
        use_cuda, ctx = False, None
        if torch.cuda.is_available():
            try:
                import nvdiffrast.torch as dr
                ctx = dr.RasterizeCudaContext()
                use_cuda = True
            except Exception:
                pass
        rmesh = {'v': V, 'f': F, 'c': C}
        a = render_views(rmesh, cam_mvs, res=args.res, ctx=ctx, use_cuda=use_cuda).to(device)
        b = render_views(rmesh, cam_mvs, res=args.res, ctx=ctx, use_cuda=use_cuda).to(device)
        cr_list, cc_list = [], []
        for i in range(a.shape[0]):
            x, y = crop_aligned(a[i], b[i], size=args.crop_size, margin=0.08)
            cr_list.append(x); cc_list.append(y)
        lp = lpips_views(torch.cat(cr_list, 0), torch.cat(cc_list, 0), vgg, vgg_layers, device)
        print(f'  [ok] 渲染+LPIPS 路径正常（同 mesh 自比 LPIPS={lp:.6f}，应≈0）。')
    else:
        print('  [demo] VGG 离线，跳过渲染+LPIPS 路径（几何路径已验证）。')

    print('=' * 60)
    print('DEMO 完成。真实评测请用 --gt_dir / --pred_root。')
    print('=' * 60)


def main():
    ap = argparse.ArgumentParser(description='Absolute geometry + perceptual eval vs GT mesh.')
    ap.add_argument('--gt_dir', type=str, default=None, help='GT mesh 根目录（按 --dataset 布局）。')
    ap.add_argument('--dataset', type=str, default='flat', choices=['gso', 'omni3d', 'flat'])
    ap.add_argument('--gt_manifest', type=str, default=None, help='显式 object_id->gt_path 映射 (csv/json)。')
    ap.add_argument('--pred_root', type=str, default=None, help='预测 mesh 根目录（按组分子目录）。')
    ap.add_argument('--tau', type=float, default=0.2, help='F-Score 主阈值（InstantMesh 用 0.2）。')
    ap.add_argument('--tau_extra', type=float, default=0.1, help='F-Score 附加阈值（审稿人要求）。')
    ap.add_argument('--n_points', type=int, default=100000, help='表面采样点数。')
    ap.add_argument('--n_views', type=int, default=12)
    ap.add_argument('--res', type=int, default=256)
    ap.add_argument('--crop_size', type=int, default=224)
    ap.add_argument('--icp', action='store_true', help='评分前刚性 ICP 配准预测到 GT。')
    ap.add_argument('--no_lpips', action='store_true', help='跳过 LPIPS（离线无 VGG 时）。')
    ap.add_argument('--out', type=str, default='outputs/absolute/metrics.json')
    ap.add_argument('--csv', type=str, default='outputs/absolute/per_object.csv')
    ap.add_argument('--demo', action='store_true', help='冒烟自测（合成数据）。')
    args = ap.parse_args()

    import torch
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.demo:
        run_demo(args)
        return

    if not args.gt_dir or not args.pred_root:
        raise SystemExit('完整评测需要 --gt_dir 与 --pred_root（或用 --demo 自测）。')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    gt_map = discover_gt(args.gt_dir, args.dataset, args.gt_manifest)
    pred_groups = discover_pred_groups(args.pred_root)
    print(f'GT 物体数: {len(gt_map)}；预测组: {list(pred_groups.keys())}')

    cam_mvs = circular_cameras(n_views=args.n_views)
    use_cuda, ctx = False, None
    if torch.cuda.is_available():
        try:
            import nvdiffrast.torch as dr
            ctx = dr.RasterizeCudaContext()
            use_cuda = True
            print('[render] nvdiffrast (CUDA)。')
        except Exception as e:
            print(f'[render] nvdiffrast 不可用，回退 CPU: {e}')
    else:
        print('[render] CUDA 不可用；使用 CPU 软件光栅（较慢）。')

    vgg, vgg_layers = (None, None)
    if not args.no_lpips:
        vgg, vgg_layers = get_vgg(device)

    results = []
    for group, pred_files in pred_groups.items():
        print(f'\n>>> 组 {group}  ({len(pred_files)} 个预测)')
        r = evaluate_group(group, pred_files, gt_map, args, device, cam_mvs, ctx, use_cuda, vgg, vgg_layers)
        if r:
            results.append(r)
            a = r['agg']
            print(f'  == {group}: CD×10³={a["cd_x1e3_mean"]:.2f}±{a["cd_x1e3_std"]:.2f}  '
                  f'FS@{args.tau}={a[f"fscore@{args.tau}_mean"]:.3f}±{a[f"fscore@{args.tau}_std"]:.3f}  '
                  f'FS@{args.tau_extra}={a[f"fscore@{args.tau_extra}_mean"]:.3f}±{a[f"fscore@{args.tau_extra}_std"]:.3f}  '
                  f'N={a["n"]}')

    out = {
        'protocol': {
            'normalization': 'center + scale to max bbox extent = 1 (unit bbox)',
            'cd_scale': 1e3,
            'tau': args.tau, 'tau_extra': args.tau_extra,
            'n_points': args.n_points, 'n_views': args.n_views, 'res': args.res,
            'icp': args.icp, 'dataset': args.dataset,
            'lpips_anchor': 'GT novel-view render (absolute)',
        },
        'groups': [r['agg'] for r in results],
        'per_object': {r['agg']['group']: r['per_object'] for r in results},
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n已写入聚合指标: {args.out}')

    # 逐物体 CSV
    with open(args.csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['group', 'object_id', 'cd_x1e3', f'fscore@{args.tau}',
                    f'fscore@{args.tau_extra}', 'lpips_vs_gt'])
        for r in results:
            for rec in r['per_object']:
                w.writerow([rec['object_id'] and r['agg']['group'], rec['object_id'],
                            f"{rec['cd_x1e3']:.4f}",
                            f"{rec[f'fscore@{args.tau}']:.4f}",
                            f"{rec[f'fscore@{args.tau_extra}']:.4f}",
                            (f"{rec['lpips_vs_gt']:.4f}" if rec['lpips_vs_gt'] is not None else '')])
    print(f'已写入逐物体明细: {args.csv}')


if __name__ == '__main__':
    main()
