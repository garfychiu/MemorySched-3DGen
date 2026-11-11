"""
生成图 7：3 行物体 × N 列 mesh 分组的定性对比。

- 行：3 个示例物体（rigid / non-rigid / thin-wall），行数固定为 3。
- 列：outputs/ablation/{stem}/meshes/ 下自动发现的 mesh 分组，按
      Baseline → +FP8 → +FP8+Chunk → +LOD128 → +LOD96 → +LOD64 排序
      （目前为 6 列：Baseline / +FP8 / +FP8+Chunk / +FP8+Chunk+LOD128 /
       +FP8+Chunk+LOD96 / +FP8+Chunk+LOD64）。

每个单元 = 同一网格的「法线图（上）+ 线框图（下）」纵向拼接。所有分组共用
同一归一化变换与同一相机视角，保证对比公平；按全局最密网格确定统一抽面比例，
保留不同网格之间的真实三角形密度对比。某分组网格缺失时该格渲染为灰色占位。

渲染方式（无需 GPU / 显示）：
  - 法线图：面法线方向映射到 RGB（(n*0.5+0.5)*255）做平面着色，painter 算法
    按深度从远到近填充，得到类似真实法线图的渲染，反映曲面平滑度。
  - 线框图：将每个三角形三条边投影到图像并以细线绘制，反映三角形分布密度。

用法：
  python render_qual.py
输出：outputs/charts/fig7_qualitative.png
"""
import os
import math
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

PANEL_W = 300            # 单格宽度
NORM_H  = 200            # 法线图高度
WIRE_H  = 170            # 线框图高度
CELL_H  = NORM_H + WIRE_H
FACE_CAP = 45000         # 单网格用于绘制的面数上限；超过则按比例随机抽面
FOV = 30.0
ELEV, AZIM, DIST = 18.0, -55.0, 3.0
# 绕世界 Y 轴旋转 90°：模型侧向展示，仍保持直立（顶点 -> (z, y, -x)）
RY90 = np.array([[0.0, 0.0, 1.0],
                 [0.0, 1.0, 0.0],
                 [-1.0, 0.0, 0.0]], dtype=float)
ABL_DIR = "outputs/ablation"
OUT_DIR = "outputs/charts"

# 3 行物体（行数不变，可改 stem）。若某 stem 无跑出的网格，自动回退到第一个
# 有完整网格的物体，并在左侧标签注明 Fallback。
CATEGORIES = [
    ("Rigid (cake)",        "cake",           "examples/cake.jpg"),
    ("Non-rigid (blue_cat)","blue_cat",       "examples/blue_cat.png"),
    ("Thin-wall (teapot)",  "genshin_teapot", "examples/genshin_teapot.png"),
]

GROUP_ORDER = [
    "Baseline",
    "+FP8",
    "+FP8+Chunk",
    "+FP8+Chunk+LOD128",
    "+FP8+Chunk+LOD96",
    "+FP8+Chunk+LOD64",
]


# --------------------------------------------------------------------------- #
# 网格 / 几何工具
# --------------------------------------------------------------------------- #
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
    a = v[f[:, 0]]
    b = v[f[:, 1]]
    c = v[f[:, 2]]
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
    # 转到相机坐标系着色：法线方向按相机基 (right, up, forward) 编码，
    # 使亮度与物体在世界中的朝向无关，正对相机的面稳定偏亮，
    # 各行列的棱面/破洞对比度一致、可比。
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


# --------------------------------------------------------------------------- #
# 文本 / 布局工具
# --------------------------------------------------------------------------- #
def load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def add_label(img, text, size=18, color=(20, 20, 20), bg=(255, 255, 255)):
    font = load_font(size)
    pad = 6
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    tw = tmp.textlength(text, font=font)
    h = size + 2 * pad
    bar = Image.new("RGB", (img.width, h), bg)
    d = ImageDraw.Draw(bar)
    d.text(((img.width - tw) / 2, pad), text, fill=color, font=font)
    out = Image.new("RGB", (img.width, img.height + h), bg)
    out.paste(bar, (0, 0))
    out.paste(img, (0, h))
    return out


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


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
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
    return resolved, avail


def main():
    resolved, avail = resolve_categories()
    if not resolved:
        print("未找到任何含 mesh 分组的物体，退出。")
        return
    groups = discover_groups(resolved[0][1])
    if not groups:
        print("未发现有 mesh 分组的目录，退出。")
        return
    print("列（mesh 分组，按序）：", groups)
    print("行（物体）：", [r[0] for r in resolved])

    cam, forward, up, right = camera_vectors()

    # 加载所有网格，按各物体首个可用分组做归一化；记录全局最密网格用于统一抽面
    data = []          # 每行: (label, [ (v,f) or None for each group ])
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
                v = v @ RY90   # 绕 Y 轴转 90°（所有分组一致，保持可比）
                all_faces.append(f)
                row.append((v, f))
            else:
                row.append(None)
        data.append((label, row))

    max_faces = max((f.shape[0] for f in all_faces), default=1)
    frac = min(1.0, FACE_CAP / max_faces)
    print(f"global max faces={max_faces}, sampling frac={frac:.3f}")

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
                                 cam, forward, up, right, frac).rotate(180)
            wire = render_wire(v, f, PANEL_W, WIRE_H,
                               cam, forward, up, right, frac).rotate(180)
            # 旋转后重新绘制正向可读的面数标注（图已 2D 翻转 180° + 几何绕 Y 轴 180°）
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

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "fig7_qualitative.png")
    final.save(out_path, dpi=(300, 300))
    print("saved", out_path, final.size)


if __name__ == "__main__":
    main()
