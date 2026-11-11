"""
profile_lod.py — 动态 LOD 显存代价模型拟合（回应评审意见第 3 点：动态 LOD 理论化）

目标：为论文 "动态 LOD 网格分辨率与网格简化" 一节提供可复现的定量代价模型，
将 "降低 grid_res 能降显存" 从经验观察提升为带参数的理论关系，并据此推导出
"按显存预算自适应选择网格分辨率 G" 的动态 LOD 准则。

理论假设：FlexiCubes 网格提取阶段的显存由规则四面体栅格的顶点数量主导，
顶点总数 N_G 与网格分辨率 G 呈立方关系 N_G = G^3（实际为 (G+1)^3，大 G 下≈G^3）。
每个顶点需存储 SDF 标量场与 3 维变形场，并在提取过程中伴随 MLP 解码的瞬时激活
缓冲，故 Stage 2 显存 M(G) 可建模为：
        M(G) = β + α · G^3
其中 β 为与 G 无关的固定开销（TriplaneTransformer 重建、forward_planes、planes 张量等），
α 为每单位 G^3 顶点的显存系数（含 SDF/变形场与解码瞬时缓冲）。

脚本做法：固定扩散阶段（仅跑一次），随后对一组 grid_res 分别调用 extract_mesh，
测量每组 Stage 2 峰值显存，对 M vs G^3 做线性最小二乘拟合得到 α、β 与 R^2。

输出：
  * <output>.json ：逐 G 实测峰值、拟合系数、自适应准则验证结果。
  * <output>.png  ：实测峰值 vs 立方拟合曲线 + 8GB 预算线（图 6）。

用法：
  python profile_lod.py --grid_res_list 32 48 64 80 96 --output outputs/charts/lod_model.png
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2
from einops import rearrange
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from omegaconf import OmegaConf

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras
from src.models.lightweight_diffusion.fp8_quantize import apply_fp8_to_unet
from src.memory_cost_model import chunk_level_to_verts


def get_view_indices(num_views):
    assert num_views in [3, 4, 6], f"Unsupported num_views: {num_views}"
    return torch.tensor(
        {6: [0, 1, 2, 3, 4, 5], 4: [0, 2, 4, 5], 3: [0, 2, 4]}[num_views],
        dtype=torch.long,
    )


def config_name(config):
    return os.path.basename(args.config).replace(".yaml", "")


def load_pipeline(config, use_fp8):
    os.environ.setdefault("HF_HUB_CACHE", r"D:\AI_HF_CACHE")
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing"
    )
    pipeline.vae = pipeline.vae.to("cuda")
    pipeline.enable_vae_slicing()
    pipeline.vae = pipeline.vae.to("cuda")
    unet_ckpt = config.infer_config.unet_path
    if not os.path.exists(unet_ckpt):
        from huggingface_hub import hf_hub_download
        unet_ckpt = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename="diffusion_pytorch_model.bin", repo_type="model")
    state_dict = torch.load(unet_ckpt, map_location="cpu")
    pipeline.unet.load_state_dict(state_dict, strict=True)
    pipeline.unet = pipeline.unet.to(dtype=torch.float16)
    pipeline.prepare()
    if use_fp8:
        print("[FP8] applying modular FP8 quantization to UNet ...")
        apply_fp8_to_unet(pipeline.unet.unet)
    return pipeline


def load_recon_model(config, grid_extract_chunk=0):
    if grid_extract_chunk and grid_extract_chunk > 0:
        config.model_config.params.grid_extract_chunk = int(grid_extract_chunk)
    model = instantiate_from_config(config.model_config)
    model_ckpt = config.infer_config.model_path
    if not os.path.exists(model_ckpt):
        from huggingface_hub import hf_hub_download
        model_ckpt = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename=f"{config_name(config)}.ckpt", repo_type="model")
    state_dict = torch.load(model_ckpt, map_location="cpu")["state_dict"]
    state_dict = {k[14:]: v for k, v in state_dict.items()
                  if k.startswith("lrm_generator.")}
    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda").eval()
    try:
        model.init_flexicubes_geometry("cuda", fovy=30.0)
        model._flexicubes_ready = True
    except Exception as e:
        model._flexicubes_ready = False
        print(f"[warn] FlexiCubes init skipped ({e}); LOD sweep 不可用。")
    return model


def _lod_style():
    plt.rcParams.update({
        'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
        'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'figure.dpi': 300, 'savefig.dpi': 300,
        'font.family': ['Times New Roman', 'SimSun', 'DejaVu Sans'],
        'axes.linewidth': 0.7, 'grid.linewidth': 0.4,
        'axes.edgecolor': '#333333', 'xtick.color': '#222222',
        'ytick.color': '#222222', 'text.color': '#111111',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False,
    })


def plot(result, out_path,
         caption='Fig. 6 Dynamic-LOD memory cost model and empirical validation (R²={r2:.4f})'):
    _lod_style()
    beta_fp = result["forward_planes_peak_gb"]
    beta_ext = result["fit"]["beta_ext_gb"]
    alpha = result["fit"]["alpha_gb_per_G3"]
    r2 = result["fit"]["r2"]
    B = result["config"]["budget_gb"]
    G_star = result["adaptive_rule"]["G_star_for_budget"]
    recs = result["records"]
    Gs = np.array([r["grid_res"] for r in recs], dtype=float)
    Ms = np.array([r["extract_peak_gb"] for r in recs], dtype=float)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    gfine = np.linspace(min(Gs) * 0.9, max(Gs) * 1.05, 200)
    ax.plot(gfine, beta_ext + alpha * gfine ** 3, "-", color="#2E7D32", lw=2,
            label=f"fit  M_ext(G)={beta_ext:.2f}+{alpha:.2e}·G³ (R²={r2:.4f})")
    ax.scatter(Gs, Ms, color="#C00000", zorder=5, s=45, label="Measured extraction peak")
    total_curve = np.maximum(beta_fp, beta_ext + alpha * gfine ** 3)
    ax.plot(gfine, total_curve, "--", color="#6A1B9A", lw=1.5,
            label=f"Total Stage2=max(beta_fp={beta_fp:.2f}, M_ext(G))")
    ax.axhline(B, color="#1F4E79", ls="--", lw=1.5, label=f"Memory budget B={B}GB")
    ax.axvline(G_star, color="#E36C09", ls=":", lw=1.5, label=f"Adaptive G*={G_star}")
    ax.annotate(f"G*={G_star}", xy=(G_star, beta_ext + alpha * G_star ** 3),
                xytext=(G_star + 3, beta_ext + alpha * G_star ** 3 + 0.5),
                color="#E36C09", fontsize=9)
    ax.set_xlabel("Grid resolution G (LOD level)")
    ax.set_ylabel("Stage 2 memory / GB")
    ax.set_title("Dynamic LOD: cubic memory cost model of mesh extraction", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.92, bottom=0.20)
    cap = caption.format(r2=r2) if "{r2}" in caption else caption
    fig.text(0.5, 0.012, cap, ha='center', va='bottom', fontsize=8.5)
    fig.savefig(out_path, dpi=300)
    print(f"[plot] saved -> {out_path}")


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/instant-mesh-large.yaml")
    parser.add_argument("--input_path", default="examples/blue_cat.png")
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--view", type=int, default=4, choices=[3, 4, 6])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--grid_res_list", type=int, nargs="+",
                        default=[32, 48, 64, 80, 96])
    parser.add_argument("--budget_gb", type=float, default=8.0,
                        help="目标部署显卡物理显存（用于验证自适应准则）。")
    parser.add_argument("--safety_margin_gb", type=float, default=0.5,
                        help="为 CUDA 碎片/非峰值分配预留的安全余量。")
    parser.add_argument("--FP8Unet", dest="fp8_unet", action="store_true",
                        help="若开启，测量的是 FP8 配置下的 Stage2 峰值（含反量化缓冲尖峰）。")
    parser.add_argument("--Chunk", dest="chunk_level", type=int, default=0,
                        help="Stage-2 chunked grid extraction level (0=off; 1=100k; 2=200k default; "
                             "3=400k); lets the LOD sweep fit high grid_res without OOM.")
    parser.add_argument("--output", default="outputs/charts/lod_model.png")
    args = parser.parse_args()
    # --Chunk 级别 → 实际分块顶点数（等价原 --grid_extract_chunk 数值）。
    args.grid_extract_chunk = chunk_level_to_verts(args.chunk_level)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    config = OmegaConf.load(args.config)
    IS_FLEXICUBES = config_name(config).startswith("instant-mesh")
    args.config = args.config  # 供 config_name 使用

    # ---- 加载 ----
    pipeline = load_pipeline(config, args.fp8_unet)
    model = load_recon_model(config, args.grid_extract_chunk)
    if not getattr(model, "_flexicubes_ready", False):
        print("[error] FlexiCubes 不可用，无法执行 LOD sweep。")
        return

    # ---- Stage 1：扩散一次，得到多视图图 ----
    input_image = Image.open(args.input_path).convert("RGB")
    output_image = pipeline(input_image, num_inference_steps=args.diffusion_steps).images[0]
    torch.cuda.synchronize()
    # 释放扩散侧资源（模拟 run.py 的 del pipeline + empty_cache），
    # 使后续 Stage 2 显存测量不被 UNet 常驻权重污染。
    del pipeline
    torch.cuda.empty_cache()

    # ---- Stage 2 预处理（与 grid_res 无关，常量开销）----
    device = "cuda"
    input_cameras = get_zero123plus_input_cameras(
        batch_size=1, radius=4.0 * args.scale).to(device)
    img = np.asarray(output_image, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
    img = rearrange(img, "c (n h) (m w) -> (n m) c h w", n=3, m=2)
    indices = get_view_indices(args.view).to(device)
    img = img.unsqueeze(0).to(device)
    img = v2.functional.resize(img, 320, interpolation=3, antialias=True).clamp(0, 1)
    img = img[:, indices]
    cameras = input_cameras[:, indices]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        planes = model.forward_planes(img, cameras)
        torch.cuda.synchronize()
    beta_fp = torch.cuda.max_memory_allocated() / 1024**3  # forward_planes 峰值（β_fp 常量开销）

    # ---- LOD sweep：逐 grid_res 测量「提取阶段」峰值 ----
    # 提取阶段峰值 = forward_planes 输出(planes)常驻 + 随 G^3 增长的 SDF/变形场与解码缓冲。
    records = []
    for G in args.grid_res_list:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        with torch.no_grad():
            model.extract_mesh(
                planes,
                use_texture_map=False,
                grid_res=G,
                simplify_ratio=0.0,
                **config.infer_config,
            )
            torch.cuda.synchronize()
        extract_peak = torch.cuda.max_memory_allocated() / 1024**3
        # Stage2 总峰值 = max(forward_planes 峰值, 提取阶段峰值)
        total = max(beta_fp, extract_peak)
        records.append({"grid_res": G, "extract_peak_gb": extract_peak,
                        "stage2_peak_gb": total})
        print(f"[measure] G={G:3d}  extract_peak={extract_peak:6.2f}GB  "
              f"stage2_peak={total:6.2f}GB  N_G={(G**3):,}")

    # ---- 拟合立方模型 M_extract(G)=β_ext + α·G^3（对提取阶段峰值）----
    Gs = np.array([r["grid_res"] for r in records], dtype=float)
    Ms = np.array([r["extract_peak_gb"] for r in records], dtype=float)
    X = (Gs ** 3).reshape(-1, 1)
    Xb = np.concatenate([np.ones_like(X), X], axis=1)
    coef, *_ = np.linalg.lstsq(Xb, Ms, rcond=None)
    beta_ext, alpha = float(coef[0]), float(coef[1])
    pred = Xb @ coef
    ss_res = float(np.sum((Ms - pred) ** 2))
    ss_tot = float(np.sum((Ms - Ms.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # ---- 自适应 LOD 选择准则 ----
    # 提取阶段峰值随 G^3 增长；给定目标显卡可用预算 B，需满足
    #   β_ext + α·G^3 + δ_fp8 + margin ≤ B
    # 其中 δ_fp8 为 FP8 反量化缓冲尖峰（近似常量），margin 为 CUDA 碎片安全余量。
    # 解得：G* = floor( ((B - δ_fp8 - β_ext - margin) / α)^(1/3) )，clamp 到 [16,128]。
    # （当 cap < β_fp 时，瓶颈转为 forward_planes，G* 退化为仅受 β_fp 约束；本实验罕见。）
    delta_fp8 = 0.0
    if args.fp8_unet:
        # FP8 反量化缓冲尖峰近似常量，从已有消融表 (G=96: 9.12→9.94) 估算 ≈0.82GB
        delta_fp8 = 0.82
    B = args.budget_gb
    margin = args.safety_margin_gb
    cap = B - delta_fp8 - beta_ext - margin
    if cap > 0:
        G_star = int(np.floor((cap / alpha) ** (1.0 / 3.0)))
    else:
        G_star = 16
    G_star = max(16, min(G_star, 128))

    result = {
        "config": {
            "view": args.view, "fp8_unet": args.fp8_unet,
            "budget_gb": B, "safety_margin_gb": margin, "delta_fp8_gb": delta_fp8,
        },
        "forward_planes_peak_gb": beta_fp,
        "records": records,
        "fit": {"alpha_gb_per_G3": alpha, "beta_ext_gb": beta_ext,
                "beta_fp_gb": beta_fp, "r2": r2,
                "formula": "M_extract(G) = beta_ext + alpha * G^3"},
        "adaptive_rule": {
            "G_star_for_budget": G_star,
            "total_stage2_at_Gstar_gb":
                max(beta_fp, beta_ext + alpha * (G_star ** 3)),
            "note": ("G* = floor(((B - delta_fp8 - beta_ext - margin)/alpha)^(1/3)), "
                     "clamped to [16,128]")
        },
    }
    json_path = os.path.splitext(args.output)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[fit] M_extract(G)={beta_ext:.3f} + {alpha:.3e}·G^3   R^2={r2:.5f}")
    print(f"[rule] B={B}GB -> G*={G_star}  "
          f"stage2_peak@G*={result['adaptive_rule']['total_stage2_at_Gstar_gb']:.2f}GB")

    # ---- 绘图：实测提取峰值 vs 立方拟合 + 预算线 ----
    plot(result, args.output)


if __name__ == "__main__":
    main()
