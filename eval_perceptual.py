"""
感知 / 纹理指标评测：以 Baseline（最高保真配置）生成的 mesh 为「参考锚点」，
把其余各组从固定视角渲染出的图像与 Baseline 同视角渲染对比，计算：

    rel_lpips : LPIPS（感知距离，越低 = 越接近 Baseline 的纹理/外观）
    rel_ssim  : SSIM（结构相似度，越高 = 越接近 Baseline）

⚠ 诚实边界（与 eval_relative.py 一致）：
   这衡量的是「优化把纹理/外观从 Baseline 上推开了多远」，
   不是「相对真实物体的绝对感知误差」。论文中应表述为
   "perceptual deviation from the unoptimized Baseline configuration"。

   若要"看起来是否真实"的绝对感知指标，需要真值多视图或输入原图作锚点；
   本脚本默认给出相对 Baseline 的 LPIPS/SSIM（与 rel_cd 同一框架），
   足以在消融表中回答"各优化是否损伤了纹理/感知质量"。

用法：
    python eval_perceptual.py --ablation_dir outputs/ablation [--n_views 12] [--res 256]
结果写回 outputs/ablation/run_metrics.jsonl（按组幂等更新：rel_lpips / rel_ssim）。
随后 run.py --gen_charts 会自动聚合这些指标并生成含 LPIPS/SSIM 列的表 1。
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

try:
    import nvdiffrast.torch as dr
except Exception as e:  # pragma: no cover
    raise SystemExit(
        'nvdiffrast is required for mesh rendering (pip install nvdiffrast, needs CUDA). '
        f'Import failed: {e}'
    )


# ───────────────────────── 网格加载 / 归一化 ─────────────────────────
def normalize_mesh(verts):
    """平移到包围盒中心，缩放到最大边长为 1，使各组的相机视角完全一致。"""
    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)
    center = (bb_min + bb_max) / 2.0
    extent = float((bb_max - bb_min).max())
    if extent <= 0:
        return verts
    return (verts - center) / extent


def load_group_mesh(mesh_dir):
    """读入某组目录下首个 .obj，返回归一化顶点、面、顶点色（[0,1]）。

    直接解析 OBJ 文本（v x y z r g b），避免 trimesh 在某些版本不回读顶点色的问题。
    """
    if not os.path.isdir(mesh_dir):
        return None
    for fn in sorted(os.listdir(mesh_dir)):
        if not fn.lower().endswith('.obj'):
            continue
        path = os.path.join(mesh_dir, fn)
        verts, faces, colors = _load_mesh(path)
        if verts is None or faces is None or len(faces) == 0:
            print(f'  [warn] skip {path}: empty/invalid mesh')
            continue
        if colors is None:
            print(f'  [warn] {path}: no vertex colors, using white (silhouette-only LPIPS).')
            colors = np.ones_like(verts)
        # 注意：此处返回"原始"顶点（不各自归一化）；统一对齐到 Baseline 包围盒
        # 由 main() 完成后处理，保证所有组在同一相机下严格对齐，SSIM 才有意义。
        return {'v': verts, 'f': faces, 'c': colors, 'file': fn}
    return None


def load_group_meshes(mesh_dir):
    """返回目录下所有 .obj 的列表，每项 {v, f, c, file}（多物体评测时逐物体平均）。"""
    out = []
    if not os.path.isdir(mesh_dir):
        return out
    for fn in sorted(os.listdir(mesh_dir)):
        if not fn.lower().endswith('.obj'):
            continue
        path = os.path.join(mesh_dir, fn)
        verts, faces, colors = _load_mesh(path)
        if verts is None or faces is None or len(faces) == 0:
            continue
        if colors is None:
            print(f'  [warn] {path}: no vertex colors, using white.')
            colors = np.ones_like(verts)
        out.append({'v': verts, 'f': faces, 'c': colors, 'file': fn})
    return out


def _load_mesh(path):
    """载入网格，兼容两种来源：

    1. 文本 OBJ（含内联顶点色 v x y z r g b，如 blue_cat）→ 用 _parse_obj 保留顶点色，
       与历史 rel_lpips 计算方式一致。
    2. .obj 实为 GLB 二进制的情况（新版 save_obj 导出 glb，如 cartoon_panda）→
       文本解析读不出面，回退 trimesh 嗅探格式，避免 nvdiffrast 报
       "tri must have shape [>0,3]"。

    返回 (V, F, C)：V:(V,3) float64, F:(F,3) int64, C:(V,3) float64∈[0,1] 或 None。
    """
    # 1) 先试文本 OBJ 解析（能正确读取内联顶点色）
    try:
        V, F, C = _parse_obj(path)
        if len(F) > 0:
            return V, F, C
    except Exception:
        pass
    # 2) 文本解析无面（GLB 二进制等），回退 trimesh 嗅探格式
    try:
        mesh = trimesh.load(path, process=False, force='mesh')
    except Exception as e:
        print(f'  [warn] failed to load {path}: {e}')
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
            if C.max() > 1.0:  # trimesh 顶点色可能是 0-255
                C = C / 255.0
    return V, F, C


def normalize_to_ref(verts, center, extent):
    """用统一参考（Baseline 的包围盒中心/尺寸）归一化，保证跨组对齐。"""
    if extent <= 0:
        return verts
    return (verts - center) / extent


def _parse_obj(path):
    """解析 OBJ：顶点（含可选 rgb）、面（仅取顶点索引）。返回 (V,3),(F,3),(V,3)|None。"""
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
                    # 支持 1 / 1/1 / 1//1 三种面格式，仅取顶点索引
                    vi = int(tok.split('/')[0])
                    idx.append(vi - 1)  # OBJ 为 1-based
                if len(idx) >= 3:
                    faces.append(idx[:3])
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    C = np.asarray(colors, dtype=np.float64) if colors else None
    if C is not None and C.shape[0] == V.shape[0]:
        if C.max() > 1.0:
            C = C / 255.0
    else:
        C = None
    return V, F, C


# ───────────────────────── 相机 / 渲染 ─────────────────────────
def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) / 2.0)
    # 标准 OpenGL 透视投影矩阵：注意 (2,3) 与 (3,2) 的位置，
    # 写反（转置）会导致 nvdiffrast 的 w/clip 维度错误、三角形被整体剔除 -> 全白图。
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * near * far) / (near - far)],
        [0, 0, -1, 0],
    ], dtype=np.float32)


def circular_cameras(n_views=12, radius=2.2, elevation=20.0):
    """返回 world->view 矩阵 (N,4,4)，与 run.py 一致的环绕相机（白底背景）。"""
    elev = np.deg2rad(elevation)
    c2ws = []
    for i in range(n_views):
        az = 2 * np.pi * i / n_views
        pos = np.array([
            radius * np.cos(elev) * np.cos(az),
            radius * np.cos(elev) * np.sin(az),
            radius * np.sin(elev),
        ], dtype=np.float32)
        # 看向原点、世界上方 +z 的相机（OpenGL 约定）
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
    c2ws = np.stack(c2ws, axis=0)           # (N,4,4) c2w
    w2cs = np.linalg.inv(c2ws)              # (N,4,4) w2c
    return w2cs


def render_views(mesh, cam_mvs, res=256, ctx=None, use_cuda=True):
    """把网格从多视角渲染为 (N,3,res,res) 图像（白底）。
    use_cuda=True 时用 nvdiffrast（快）；否则用纯 numpy 软件光栅化（无需 MSVC/CUDA 插件）。"""
    if use_cuda:
        return _render_views_cuda(mesh, cam_mvs, res=res, ctx=ctx)
    return _render_views_cpu(mesh, cam_mvs, res=res)


def _render_views_cuda(mesh, cam_mvs, res=256, ctx=None):
    if ctx is None:
        ctx = dr.RasterizeCudaContext()
    v = torch.from_numpy(mesh['v']).float().cuda().unsqueeze(0)   # (1,V,3) instanced 模式
    c = torch.from_numpy(mesh['c']).float().cuda().unsqueeze(0)   # (1,V,3)
    # 关键：nvdiffrast 的 tri 必须是 2D [F,3]（instanced 模式下拓扑共享），
    # 不能 unsqueeze 成 3D，否则触发 "tri must have shape [>0, 3]"。
    f = torch.from_numpy(mesh['f']).int().cuda().contiguous()     # (F,3)
    proj = perspective(30.0, 1.0, 0.1, 100.0)
    proj_t = torch.from_numpy(proj).float().cuda()
    cam_mvs_t = torch.from_numpy(cam_mvs).float().cuda()          # (N,4,4)
    imgs = []
    with torch.inference_mode():
        for i in range(cam_mvs_t.shape[0]):
            mvp = proj_t @ cam_mvs_t[i]                            # (4,4)
            vh = F.pad(v, (0, 1), value=1.0)                       # (1,V,4)
            v_clip = (mvp @ vh.transpose(1, 2)).transpose(1, 2).contiguous()  # (1,V,4) 必须连续，否则 nvdiffrast 报错
            rast = dr.rasterize(ctx, v_clip, f, resolution=[res, res])[0]
            color, _ = dr.interpolate(c, rast, f.contiguous())     # (1,res,res,3)
            alpha = (rast[..., 3:4] > 0).float()
            img = color * alpha + (1.0 - alpha)                    # 白底
            imgs.append(img.permute(0, 3, 1, 2).clamp(0, 1).cpu())
    return torch.cat(imgs, dim=0)                                  # (N,3,res,res)


def _subsample_faces(v, f, c, max_faces=15000):
    """随机抽取面片以保证 CPU 光栅化速度（顶点色随顶点保留，仅抽面）。"""
    if f.shape[0] <= max_faces:
        return v, f, c
    rng = np.random.default_rng(0)
    idx = rng.choice(f.shape[0], max_faces, replace=False)
    return v, f[idx], c


def _render_views_cpu(mesh, cam_mvs, res=256):
    """纯 numpy 透视软件光栅化（顶点色，白底）。无需 CUDA / 编译器。"""
    v0, f0, c0 = _subsample_faces(mesh['v'], mesh['f'], mesh['c'], max_faces=15000)
    v = np.asarray(v0, dtype=np.float64)
    f = np.asarray(f0, dtype=np.int64)
    c = np.asarray(c0, dtype=np.float64)
    P = perspective(30.0, 1.0, 0.1, 100.0)                        # (4,4)
    imgs = []
    for mv in cam_mvs:
        mv = np.asarray(mv, dtype=np.float64)
        vh = np.concatenate([v, np.ones((v.shape[0], 1))], axis=1)   # (V,4)
        view = (mv @ vh.T).T                                       # world->view (V,4)
        clip = (P @ view.T).T                                      # (V,4)
        w = clip[:, 3]
        ndc = clip[:, :3] / w[:, None]
        px = (ndc[:, 0] * 0.5 + 0.5) * (res - 1)
        py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (res - 1)
        pz = -view[:, 2]                                           # 视空间深度，越大越近
        img = np.ones((res, res, 3), dtype=np.float64)             # 白底
        zbuf = np.full((res, res), -1.0, dtype=np.float64)
        xs = px[f]; ys = py[f]; zs = pz[f]; ws = w[f]; cs = c[f]    # (F,3)
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
            col = np.stack([col_r, col_g, col_b], axis=-1)              # (H,W,3)
            gxf = gx[m]; gyf = gy[m]
            zpf = zp[m]; colf = col[m]
            cur = zbuf[gyf, gxf]
            up = zpf > cur
            if up.any():
                gyu = gyf[up]; gxu = gxf[up]
                zbuf[gyu, gxu] = zpf[up]
                img[gyu, gxu] = colf[up]
        imgs.append(img.transpose(2, 0, 1))                       # (3,res,res)
    return np.stack(imgs, axis=0).astype(np.float32)               # (N,3,res,res)


# ───────────────────────── 感知指标 ─────────────────────────
def ssim(img1, img2, window_size=11, data_range=1.0, mask=None):
    """标准 SSIM（多通道）。img: (N,3,H,W) in [0,1]。
    mask: 可选 (N,H,W) 布尔，仅在该区域上求均值（排除白底背景，得到更有意义的物体相似度）。
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    # 高斯窗
    g = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(g ** 2) / (2.0 * (window_size / 6.0) ** 2))
    g = (g / g.sum()).unsqueeze(1)
    window = (g @ g.t()).unsqueeze(0).unsqueeze(0)
    window = window.repeat(3, 1, 1, 1).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=3)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if mask is not None:
        m = mask.unsqueeze(1).float()  # (N,1,H,W)
        num = (ssim_map * m).sum()
        den = m.sum().clamp(min=1.0)
        return float((num / den).item())
    return float(ssim_map.mean().item())


def crop_aligned(ref, cand, size=224, margin=0.08):
    """把两张 (3,H,W) 渲染图各自裁到「自己的前景包围盒」并对齐到同一尺寸 (size,size)。

    关键：分别按各自前景框裁剪再 resize，等于把两个物体都归一化到同一画框，
    从而彻底消除平移 / 尺度错位（SSIM 是局部窗口算子，对亚像素错位极敏感，
    这也是此前 View-4 与 Baseline 几何几乎一致却 SSIM≈0.06~0.15 的根因）。
    归一化后 SSIM/LPIPS 只比较「物体形状与纹理本身」，正是感知保真度想要的。
    """
    def _crop_one(x):
        H, W = x.shape[-2], x.shape[-1]
        fg = (x < 0.98).any(dim=0)              # (H,W) 前景（非白底）
        if not fg.any():
            return x.unsqueeze(0)
        ys, xs = torch.where(fg)
        y0, y1 = ys.min().item(), ys.max().item()
        x0, x1 = xs.min().item(), xs.max().item()
        mh = max(int((y1 - y0) * margin), 1)
        mw = max(int((x1 - x0) * margin), 1)
        y0 = max(0, y0 - mh)
        y1 = min(H - 1, y1 + mh)
        x0 = max(0, x0 - mw)
        x1 = min(W - 1, x1 + mw)
        return x[:, y0:y1 + 1, x0:x1 + 1].unsqueeze(0)

    cr = F.interpolate(_crop_one(ref), (size, size), mode='bilinear', align_corners=False)
    cc = F.interpolate(_crop_one(cand), (size, size), mode='bilinear', align_corners=False)
    return cr, cc


def get_vgg(device):
    """尝试加载 VGG16（ImageNet）用于 LPIPS 风格特征距离；无网络则回退 None。"""
    try:
        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        layers = [3, 8, 15, 22, 29]  # relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
        print('  [ok] VGG16 loaded -> computing LPIPS-style perceptual distance.')
        return vgg, layers
    except Exception as e:
        print(f'  [skip] VGG16 unavailable (offline?), LPIPS omitted. ({e})')
        return None, None


@torch.no_grad()
def lpips_views(imgs1, imgs2, vgg, layers, device):
    """imgs: (N,3,H,W) in [0,1]。返回所有视角平均的 LPIPS 风格距离。"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def features(x):
        x = (x - 0.5) / 0.5            # [0,1] -> [-1,1]
        x = (x - mean) / std          # ImageNet 归一化
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
        a = a / (a.norm(dim=1, keepdim=True) + 1e-8)   # LPIPS 通道归一化
        b = b / (b.norm(dim=1, keepdim=True) + 1e-8)
        d = (a - b).pow(2).mean(dim=(2, 3)).mean(dim=1)
        dist += d
    return float(dist.mean().item())


# ───────────────────────── 主流程 ─────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Perceptual (LPIPS/SSIM) vs Baseline.')
    parser.add_argument('--ablation_dir', type=str, default='outputs/ablation')
    parser.add_argument('--ref_group', type=str, default='Baseline')
    parser.add_argument('--n_views', type=int, default=12, help='环绕渲染视角数')
    parser.add_argument('--res', type=int, default=256, help='渲染分辨率')
    parser.add_argument('--crop_size', type=int, default=224, help='前景裁剪对齐后的尺寸')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

    # 同组可能有多条记录（多物体）；这里取每组最后一条用于定位 mesh_dir，
    # 真正的多物体平均在渲染阶段按文件名逐物体完成。
    by_group = {}
    for r in records:
        by_group[r['group']] = r

    canonical = ['Baseline', '+FP8', '+FP8+Chunk',
                 '+FP8+Chunk+LOD128', '+FP8+Chunk+LOD96', '+FP8+Chunk+LOD64']
    ordered = [by_group[g] for g in canonical if g in by_group]
    ordered += [r for r in records if r['group'] not in canonical]

    if args.ref_group not in by_group:
        raise KeyError(f'Reference group "{args.ref_group}" not found: {list(by_group)}')
    ref_dir = by_group[args.ref_group].get('mesh_dir')
    if not ref_dir or not os.path.isdir(ref_dir):
        raise FileNotFoundError(f'Baseline mesh_dir missing: {ref_dir}')

    # 载入每个组目录下【全部】 .obj（支持多物体逐物体评测）。
    print('Loading meshes for all groups ...')
    meshes = {}
    for r in ordered:
        d = r.get('mesh_dir')
        if not d or not os.path.isdir(d):
            continue
        objs = load_group_meshes(d)
        if objs:
            meshes[r['group']] = objs

    # 颜色迁移：简化组丢失顶点色时，从同物体、未简化的兄弟组按最近顶点拷回。
    try:
        from scipy.spatial import cKDTree
        def _transfer_colors(dst, donor):
            tree = cKDTree(donor['v'])
            _, idx = tree.query(dst['v'])
            return donor['c'][idx]
        for g, objs in meshes.items():
            if g == args.ref_group:
                continue
            for ob in objs:
                if float(ob['c'].mean()) > 0.985:
                    donor_objs = meshes.get('+FP8+Chunk+LOD64') or meshes.get('+FP8') or meshes.get('Baseline')
                    if not donor_objs:
                        continue
                    for d in donor_objs:
                        if d['file'] == ob['file'] and float(d['c'].mean()) <= 0.985:
                            ob['c'] = _transfer_colors(ob, d)
                            print(f'  [fix] recovered vertex colors for "{g}/{ob["file"]}" '
                                  f'from sibling by nearest-vertex.')
                            break
    except Exception as e:
        print(f'  [warn] color transfer skipped: {e}')

    # 统一对齐到 Baseline 包围盒（中心/尺寸），保证跨组/跨物体同相机严格对齐。
    ref_objs = meshes[args.ref_group]
    ref_center = ref_objs[0]['v'].min(axis=0) + \
        (ref_objs[0]['v'].max(axis=0) - ref_objs[0]['v'].min(axis=0)) / 2.0
    ref_extent = float((ref_objs[0]['v'].max(axis=0) - ref_objs[0]['v'].min(axis=0)).max())
    for g, objs in meshes.items():
        for ob in objs:
            ob['v'] = normalize_to_ref(ob['v'], ref_center, ref_extent)

    cam_mvs = circular_cameras(n_views=args.n_views)
    use_cuda = False
    ctx = None
    if torch.cuda.is_available():
        try:
            ctx = dr.RasterizeCudaContext()
            use_cuda = True
            print('[render] using nvdiffrast (CUDA).')
        except Exception as e:
            print(f'[render] nvdiffrast unavailable ({e}); '
                  f'falling back to CPU numpy rasterizer (slower, face-subsampled).')
    else:
        print('[render] CUDA not available; using CPU numpy rasterizer (slower, face-subsampled).')

    print(f'Rendering {args.n_views} views @ {args.res}px (white background)...')
    ref_renders = {}
    for ob in ref_objs:
        rimgs = render_views(ob, cam_mvs, res=args.res, ctx=ctx, use_cuda=use_cuda)
        ref_renders[ob['file']] = torch.from_numpy(np.asarray(rimgs)).to(device)

    vgg, vgg_layers = get_vgg(device)

    new_metrics = {args.ref_group: 0.0}  # Baseline: LPIPS=0（锚点，与自身对比）
    print(f'  {args.ref_group:14s} rel_lpips=0.0000')

    for r in ordered:
        g = r['group']
        if g == args.ref_group:
            continue
        objs = meshes.get(g)
        if not objs:
            print(f'  [warn] group "{g}" has no valid .obj, skipped.')
            continue
        lp_sum, cnt = 0.0, 0
        for ob in objs:
            fn = ob['file']
            if fn not in ref_renders:
                print(f'  [warn] {g}/{fn} 无 Baseline 对应物体，跳过。')
                continue
            cimgs_np = render_views(ob, cam_mvs, res=args.res, ctx=ctx, use_cuda=use_cuda)
            cimgs = torch.from_numpy(np.asarray(cimgs_np)).to(device)
            # 逐视角裁到「并集前景包围盒」并对齐到同一尺寸，消除亚像素错位。
            cr_list, cc_list = [], []
            for i in range(cimgs.shape[0]):
                cr, cc = crop_aligned(ref_renders[fn][i], cimgs[i],
                                      size=args.crop_size, margin=0.08)
                cr_list.append(cr)
                cc_list.append(cc)
            CR = torch.cat(cr_list, 0)
            CC = torch.cat(cc_list, 0)
            lp = lpips_views(CR, CC, vgg, vgg_layers, device) if vgg is not None else None
            lp_sum += lp if lp is not None else 0.0
            cnt += 1
        if cnt == 0:
            print(f'  [warn] group "{g}" 无匹配输入，跳过。')
            continue
        mean_lp = lp_sum / cnt if vgg is not None else None
        new_metrics[g] = mean_lp
        lp_str = f'{mean_lp:.4f}' if mean_lp is not None else '  n/a '
        print(f'  {g:14s} rel_lpips={lp_str}  (avg over {cnt} obj)')

    # 幂等写回（LPIPS 仅在成功计算时覆盖，避免离线无 VGG 时清掉已有值）
    for r in records:
        if r['group'] in new_metrics:
            lp = new_metrics[r['group']]
            if lp is not None:
                r['rel_lpips'] = lp

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'\nWrote perceptual metrics (rel_lpips) into {jsonl_path}')
    print('Next: python run.py configs/instant-mesh-large.yaml <img> --gen_charts')


if __name__ == '__main__':
    main()
