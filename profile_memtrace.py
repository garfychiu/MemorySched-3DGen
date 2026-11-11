"""
profile_memtrace.py — 逐步骤显存时序插桩工具（回应评审意见第 2 点）

目标：为 "FP8 量化反量化缓冲抬升 Stage2 瞬时显存" 这一论断提供可复现的
显存轨迹（memory trace）证据。脚本沿 InstantMesh 的两阶段推理流水线采样
CUDA allocated / reserved 显存：

  * Stage 1（扩散去噪）：借助 diffusers 的 callback_on_step_end，在每个去噪
    步结束时记录一次 allocated / reserved / 阶段内峰值。
  * Stage 2（LRM 重建）：在 forward_planes、extract_mesh 等关键事件前后采样，
    并在阶段起始 reset 峰值，记录阶段内峰值。

输出：
  * <output>.png  ：Stage1 步级 + Stage2 事件级双栏时序图（图 5）。
  * <output>.json ：原始采样轨迹，便于二次分析或论文绘图复用。

用法示例：
  # 单配置（含 FP8）生成时序图
  python profile_memtrace.py --FP8Unet --view 4 \
      --output outputs/charts/memtrace_fp8.png

  # 对比模式：先跑 FP8-off，再跑 FP8-on，两张轨迹叠加（推荐，最能体现尖峰）
  python profile_memtrace.py --compare --view 4 \
      --output outputs/charts/memtrace_compare.png

说明：复用了 run.py 的模型加载与分阶段逻辑，但不保存网格/视频，仅做显存采样。
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


def get_view_indices(num_views):
    assert num_views in [3, 4, 6], f"Unsupported num_views: {num_views}"
    return torch.tensor(
        {6: [0, 1, 2, 3, 4, 5], 4: [0, 2, 4, 5], 3: [0, 2, 4]}[num_views],
        dtype=torch.long,
    )


def sample(mem_log):
    """记录当前 CUDA 显存状态（GB）。"""
    mem_log.append({
        "alloc_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
    })
    return mem_log[-1]


def load_pipeline(config, use_fp8):
    """加载 zero123plus 扩散 pipeline（可选 FP8 权重量化）。"""
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


def load_recon_model(config, cpu_offload_triplane, grid_extract_chunk=0):
    """加载 LRM 重建模型。"""
    if grid_extract_chunk and grid_extract_chunk > 0:
        config.model_config.params.grid_extract_chunk = int(grid_extract_chunk)
    if cpu_offload_triplane:
        config.model_config.params.triplane_offload = True
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
    model._flexicubes_ready = False
    if IS_FLEXICUBES:
        try:
            model.init_flexicubes_geometry("cuda", fovy=30.0)
            model._flexicubes_ready = True
        except Exception as e:
            # 环境缺少 Ninja / CUDA 编译工具链时无法 JIT nvdiffrast；
            # 容错跳过 FlexiCubes 网格提取，仅保留 forward_planes 的显存采样。
            print(f"[warn] FlexiCubes init skipped ({e}); "
                  f"Stage2 仅采样 forward_planes，extract_mesh 事件标记为 unavailable。")
    return model


def config_name(config):
    return os.path.basename(args.config).replace(".yaml", "")


def run_one(args, use_fp8, label):
    """执行一次完整的两阶段推理并采样显存，返回轨迹字典。"""
    config = OmegaConf.load(args.config)
    IS_FLEXICUBES = config_name(config).startswith("instant-mesh")
    if args.grid_res > 0 and IS_FLEXICUBES:
        config.model_config.params.grid_res = args.grid_res

    # ---- Stage 0: 加载 ----
    pipeline = load_pipeline(config, use_fp8)
    model = load_recon_model(config, args.cpu_offload_triplane, args.grid_extract_chunk)

    # ---- Stage 1: 多视图扩散（步级采样） ----
    # 当前 diffusers 版本的 zero123plus 自定义 pipeline 透传的基类
    # StableDiffusionPipeline.__call__ 不接受 callback_on_step_end，故改为包装
    # scheduler.step：每个去噪步恰好调用一次 scheduler.step，与"推理步骤"一一对应。
    input_image = Image.open(args.input_path).convert("RGB")
    stage1_log = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    orig_step = pipeline.scheduler.step
    _step_idx = {"n": 0}

    def wrapped_step(self, *a, **k):
        res = orig_step(*a, **k)
        torch.cuda.synchronize()
        stage1_log.append({
            "alloc_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        })
        _step_idx["n"] += 1
        return res

    import types
    pipeline.scheduler.step = types.MethodType(wrapped_step, pipeline.scheduler)

    output_image = pipeline(
        input_image,
        num_inference_steps=args.diffusion_steps,
    ).images[0]
    torch.cuda.synchronize()
    stage1_peak = torch.cuda.max_memory_allocated() / 1024**3

    # 释放扩散侧资源，模拟 run.py 行为（del pipeline + empty_cache）
    del pipeline
    torch.cuda.empty_cache()

    # ---- Stage 2: LRM 重建（事件级采样） ----
    stage2_log = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    stage2_log.append({"event": "before_recon", **sample([])})

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

    with torch.no_grad():
        planes = model.forward_planes(img, cameras)
        torch.cuda.synchronize()
        stage2_log.append({"event": "after_forward_planes", **sample([])})

        if getattr(model, "_flexicubes_ready", False):
            mesh_out = model.extract_mesh(
                planes,
                use_texture_map=False,
                grid_res=config.model_config.params.grid_res,
                simplify_ratio=0.0,
                **config.infer_config,
            )
            torch.cuda.synchronize()
            stage2_log.append({"event": "after_extract_mesh", **sample([])})
        else:
            stage2_log.append({"event": "extract_mesh(unavailable)",
                               "alloc_gb": float("nan"),
                               "reserved_gb": float("nan"),
                               "peak_gb": float("nan")})

    stage2_peak = torch.cuda.max_memory_allocated() / 1024**3
    torch.cuda.empty_cache()

    return {
        "label": label,
        "use_fp8": use_fp8,
        "stage1": {
            "steps": list(range(len(stage1_log))),
            "alloc": [r["alloc_gb"] for r in stage1_log],
            "reserved": [r["reserved_gb"] for r in stage1_log],
            "peak": stage1_peak,
        },
        "stage2": {
            "events": [r["event"] for r in stage2_log],
            "alloc": [r["alloc_gb"] for r in stage2_log],
            "reserved": [r["reserved_gb"] for r in stage2_log],
            "peak": stage2_peak,
        },
    }


def _fmt_peak(v):
    return f"{v:.2f}GB" if v == v else "n/a"


def _memtrace_style():
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


def plot(traces, out_path,
         caption='Fig. 5 Two-stage inference memory traces (allocated / reserved, FP8-off vs FP8-on)'):
    _memtrace_style()
    colors = {"FP8-off": "#1F4E79", "FP8-on": "#C00000"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 5.4), sharex=False)

    # ---- Stage 1: 步级 allocated / reserved 显存 ----
    for t in traces:
        c = colors.get(t["label"], None)
        s1 = t["stage1"]
        ax1.plot(s1["steps"], s1["alloc"], marker="o", ms=3,
                 label=f'{t["label"]} (peak {_fmt_peak(s1["peak"])})', color=c)
        ax1.plot(s1["steps"], s1["reserved"], ls="--", lw=1.0, alpha=0.55, color=c)
    ax1.set_title("Stage 1 — Diffusion denoising (per-step memory)", fontsize=9)
    ax1.set_xlabel("Denoising step")
    ax1.set_ylabel("Memory / GB")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # ---- Stage 2: 事件级 allocated 显存 ----
    xpos = np.arange(len(traces[0]["stage2"]["events"]))
    for t in traces:
        c = colors.get(t["label"], None)
        s2 = t["stage2"]
        ax2.plot(xpos, s2["alloc"], marker="s", ms=6,
                 label=f'{t["label"]} (peak {_fmt_peak(s2["peak"])})', color=c)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(traces[0]["stage2"]["events"], rotation=15, fontsize=8)
    ax2.set_title("Stage 2 — LRM reconstruction (event-level memory)", fontsize=9)
    ax2.set_ylabel("Memory / GB")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    fig.subplots_adjust(left=0.18, right=0.97, top=0.93, bottom=0.17, hspace=0.34)
    fig.text(0.5, 0.012, caption, ha='center', va='bottom', fontsize=8.5)
    fig.savefig(out_path, dpi=300)
    print(f"[plot] saved -> {out_path}")


def main():
    global IS_FLEXICUBES, args
    from src.memory_cost_model import chunk_level_to_verts
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/instant-mesh-large.yaml")
    parser.add_argument("--input_path", default="examples/blue_cat.png")
    parser.add_argument("--diffusion_steps", type=int, default=75)
    parser.add_argument("--view", type=int, default=4, choices=[3, 4, 6])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--LOD", dest="grid_res", type=int, default=-1)
    parser.add_argument("--FP8Unet", dest="fp8_unet", action="store_true",
                        help="Enable FP8 UNet quantization for this run.")
    parser.add_argument("--cpu_offload_triplane", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="Run FP8-off then FP8-on and overlay both traces.")
    parser.add_argument("--Chunk", dest="chunk_level", type=int, default=0,
                        help="Stage-2 chunked grid extraction level (0=off; 1=100k; 2=200k default; "
                             "3=400k); bounds peak memory so high-LOD meshes fit on 8GB.")
    parser.add_argument("--output", default="outputs/charts/memtrace.png")
    args = parser.parse_args()
    # --Chunk 级别 → 实际分块顶点数（等价原 --grid_extract_chunk 数值）。
    args.grid_extract_chunk = chunk_level_to_verts(args.chunk_level)

    config = OmegaConf.load(args.config)
    IS_FLEXICUBES = config_name(config).startswith("instant-mesh")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    traces = []
    if args.compare:
        traces.append(run_one(args, use_fp8=False, label="FP8-off"))
        traces.append(run_one(args, use_fp8=True, label="FP8-on"))
    else:
        traces.append(run_one(args, use_fp8=args.fp8_unet,
                              label="FP8-on" if args.fp8_unet else "FP8-off"))

    # 原始数据落盘
    json_path = os.path.splitext(args.output)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)
    print(f"[json] saved -> {json_path}")

    plot(traces, args.output)


if __name__ == "__main__":
    main()
