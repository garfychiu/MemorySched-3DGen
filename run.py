import os
import sys
import shutil
import contextlib
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
def seed_everything(seed=0):
    """本地实现，避免 import pytorch_lightning 牵出 torchmetrics→transformers→huggingface-hub 版本冲突。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import time

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video
from src.models.lightweight_diffusion.sparse_attention import (
    SparseAttnProcessor,
)
from src.models.lightweight_diffusion.structured_prune import (
    apply_unet_channel_pruning,
    count_unet_conv_weights,
    count_unet_nonzero_conv_weights,
)
from src.models.lightweight_diffusion.fp8_quantize import (
    apply_fp8_to_unet,
    count_unet_weight_gb,
)
from src.memory_cost_model import chunk_level_to_verts


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def get_view_indices(num_views):
    """
    Map the number of input views to the indices of the original 6 zero123plus views.
    Original 6 views layout (3x2 grid):
        0: front-up       1: back-down
        2: left-up        3: right-down
        4: right-up       5: back-down-right
    """
    assert num_views in [3, 4, 6], f"Unsupported num_views: {num_views}, must be 3, 4, or 6"
    indices = {
        6: [0, 1, 2, 3, 4, 5],
        4: [0, 2, 4, 5],  # front + left-up + right-up + back-down-right
        3: [0, 2, 4],     # front + left-up + right-up
    }[num_views]
    return torch.tensor(indices, dtype=torch.long)


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    mesh_v = None
    mesh_f = None
    with torch.inference_mode():
        # For FlexiCubes, extract the mesh once and reuse it for every frame.
        # Recomputing the full SDF grid per frame is what causes OOM.
        if is_flexicubes:
            mesh_v, mesh_f, _, _, _, _ = model.get_geometry_prediction(planes)
        
        for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
            if is_flexicubes:
                frame = model.forward_geometry_cached(
                    planes,
                    mesh_v,
                    mesh_f,
                    render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['img']
            else:
                frame = model.forward_synthesizer(
                    planes,
                    render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['images_rgb']
            # Move each frame to CPU immediately to avoid GPU memory accumulation
            frames.append(frame.cpu())
            del frame
            # Free temporary GPU activations between frames
            torch.cuda.empty_cache()
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames


###############################################################################
# Arguments.
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('input_path', type=str, help='Path to input image or directory.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[3, 4, 6], help='Number of input views.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
parser.add_argument('--prune_ratio', type=float, default=0.0, help='UNet Conv2d output-channel pruning ratio (0 disables).')
parser.add_argument('--use_sparse_attn', action='store_true', help='Use foreground-masked sparse self-attention in UNet.')
parser.add_argument('--FP8Unet', dest='fp8_unet', action='store_true', help='[Pillar 1] Store UNet Conv2d/Linear weights in FP8 (FP16 compute) to halve Stage-1 weight memory.')
parser.add_argument('--triplane_quant', action='store_true', help='Per-channel quantize Triplane features before rendering (quality-impact simulation).')
parser.add_argument('--triplane_quant_bits', type=int, default=8, help='Triplane quantization bit-width when --triplane_quant is set.')
parser.add_argument('--triplane_compress_dim', type=int, default=0, help='Compress Triplane channels to this dim via 1x1 conv (0 disables). Needs Step 6 distillation.')
parser.add_argument('--triplane_query_chunk', type=int, default=0, help='Chunk size (M) for chunked Triplane sampling; 0 = no chunking. Limits peak memory of sampled features.')
parser.add_argument('--cpu_offload_triplane', action='store_true', help='Keep Triplane on CPU and stream chunks to GPU on query (Step 4 dynamic offload).')
parser.add_argument('--Chunk', dest='chunk_level', type=int, default=0,
                    help='[Pillar 2] Stage-2 chunked grid-extraction level for SDF/deformation/weight MLPs. '
                         '0=off (whole-batch, needs >=16GB); 1=100k (aggressive); 2=200k (default, LOD128 ~6.12GB); '
                         '3=400k (light). Decouples Stage-2 peak from G^3. Replaces --grid_extract_chunk.')
parser.add_argument('--grid_extract_chunk', dest='grid_extract_chunk', type=int, default=0,
                    help='[Pillar 2] Explicit Stage-2 chunk vertex count override (0 = derive from --Chunk level). '
                         'Takes precedence over --Chunk when >0; used by the λ-sweep experiment to scan '
                         'arbitrary g (e.g. 50000 / 800000) so λ = g/G^3 can be varied continuously.')
parser.add_argument('--LOD', dest='grid_res', type=int, default=-1, help='[Pillar 2] FlexiCubes grid resolution override (<=0 keeps config value, default 128). Lower = faster/coarser mesh (Step 5 dynamic LOD).')
parser.add_argument('--Lambda', dest='Lambda', type=float, nargs='?', const=-1.0, default=None,
                    help='[Pillar 2] Stage-2 chunk sizing by critical factor λ = g/G³ (g=chunk verts, G=grid_res). '
                         'Use with NO value (--Lambda) to auto-pick the equilibrium λ0 '
                         '(g=λ0·G³, Stage2 peak ≈ Stage1 rigid lower bound → global peak locked at Stage1, no OOM). '
                         'Use WITH a value (--Lambda 0.5250) to FORCE λ=0.5250 → g=λ·G³. '
                         'λ0≈0.0776 @LOD128 is the memory-balanced point; larger λ raises Stage2 peak (fewer chunks, '
                         'faster but more VRAM, may OOM); smaller λ lowers Stage2 peak (more chunks, slower, safer). '
                         'λ must be in (0, 1]; g is clamped to [1k, G³] (G³ = whole-mesh batch).')
parser.add_argument('--Simplify', dest='simplify_ratio', type=float, default=0.0, help='Quadric edge-collapse simplification target face ratio (0 disables, e.g. 0.5 keeps half). Step 5.')
parser.add_argument('--attn_impl', type=str, default='default', choices=['default', 'eager'],
                    help='UNet attention backend. "default" keeps diffusers 0.20.2 SDPA/Flash-style '
                         'efficient attention (already active in this stack). "eager" forces the legacy '
                         'manual-attention processor (no SDPA) to ablate the FlashAttention diffusion '
                         'optimization for the SOTA comparison (Table 3).')
parser.add_argument('--preview', action='store_true', help='Quick low-LOD preview: also extract a coarse mesh at --preview_grid_res before the final mesh (Step 5).')
parser.add_argument('--preview_grid_res', type=int, default=64, help='Grid resolution for the --preview coarse mesh (default 64). Step 5.')
parser.add_argument('--gen_charts', action='store_true', help='Generate ablation charts after inference (auto-aggregates recorded runs, falls back to --chart_results JSON or built-in default data).')
parser.add_argument('--chart_results', type=str, default='', help='Path to JSON file with ablation results for chart generation.')
parser.add_argument('--ablation_name', type=str, default='', help='Override the auto-derived ablation group name used when recording runtime metrics.')
parser.add_argument('--ablation_dir', type=str, default='outputs/ablation', help='Directory where per-run runtime metrics are recorded/aggregated for chart generation.')
# 支柱三·自适应调度闭环：给定显存预算 B，自动选 FP8 on/off、栅格分块粒度、G*。
parser.add_argument('--auto', action='store_true',
                    help='[Pillar 3] Auto-schedule FP8/chunk/G* from GPU budget. '
                         'Budget = --budget_gb if given (>0), else auto-detected from GPU VRAM.')
parser.add_argument('--budget_gb', type=float, default=0.0,
                    help='[Pillar 3] Target GPU memory budget (GB) for --auto scheduler. '
                         '0 = auto-detect from installed GPU total VRAM.')
parser.add_argument('--cost_model_json', type=str, default='',
                    help='[Pillar 3] Optional profile_lod.py fit JSON to override default cost-model coefficients.')
args = parser.parse_args()

# --Chunk 级别 → Stage-2 分块顶点数；若显式给定 --grid_extract_chunk 则优先（用于 λ-sweep 任意顶点数扫描）。
if args.grid_extract_chunk <= 0:
    args.grid_extract_chunk = chunk_level_to_verts(args.chunk_level)

# 支柱三·自适应调度闭环：给定显存预算 B，自动选 FP8/分块/G*（覆盖手动 flag）。
# 代价模型与 profile_lod.py 同源，设备预算 B 下框架自动在 Pareto 前沿选工作点。
if args.auto:
    from src.memory_cost_model import (
        recommend_config, load_cost_model, DEFAULT_COST_MODEL,
        detect_device_budget_gb,
    )
    _cm = DEFAULT_COST_MODEL
    if args.cost_model_json:
        try:
            _cm = load_cost_model(args.cost_model_json)
        except Exception as e:
            print(f'[auto] 载入代价模型 JSON 失败 ({e})，回退到默认系数。')
    # 预算 B：显式 --budget_gb 优先；否则自动探测 GPU 总显存
    if args.budget_gb and args.budget_gb > 0:
        budget = args.budget_gb
        budget_src = f"--budget_gb {budget}GB"
    else:
        budget = detect_device_budget_gb()
        if budget <= 0:
            print('[auto] 未检测到 CUDA GPU，回退到默认 8GB 预算。')
            budget = 8.0
            budget_src = "fallback 8GB (no CUDA)"
        else:
            budget_src = f"auto-detected GPU VRAM = {budget}GB"
    rec = recommend_config(budget, cm=_cm)
    if args.fp8_unet or args.grid_extract_chunk or args.grid_res > 0:
        print('[auto] 检测到手动显存 flag，已被 --auto 调度结果覆盖。')
    args.fp8_unet = rec['fp8_unet']
    args.grid_extract_chunk = rec['grid_extract_chunk']
    args.grid_res = rec['grid_res']
    print('[auto-scheduler] 自适应调度结果（支柱三·闭环）:')
    print(f'  预算来源: {budget_src}')
    print('  ' + rec['decision'])
    print(f'  (代价模型来源: {_cm.source}, α={_cm.alpha:.3e}, β_ext={_cm.beta_ext:.2f}, '
          f'β_fp={_cm.beta_fp:.2f}, R²={_cm.r2:.4f})')

seed_everything(args.seed)


def derive_group_name(args):
    """
    Derive a cumulative ablation group label from the enabled flags, matching the
    6 groups in Table 1 (run_ablation.py GROUPS). Falls back to a descriptive
    label built from the active flags so direct `run.py` calls never collide.
    """
    fp8 = args.fp8_unet
    chunk = args.chunk_level > 0 or args.grid_extract_chunk > 0
    lod = args.grid_res if args.grid_res > 0 else 0   # 0 = config default (LOD128)
    simplify = args.simplify_ratio > 0
    offload = args.cpu_offload_triplane
    sparse = args.use_sparse_attn

    if not (sparse or fp8 or chunk or lod or simplify or offload):
        return 'Baseline'

    parts = []
    if sparse:
        parts.append('SparseAttn')
    if fp8:
        parts.append('FP8')
    if chunk:
        parts.append('Chunk')
    if lod:
        parts.append(f'LOD{lod}')
    if simplify:
        parts.append(f'Simplify{simplify}')
    if offload:
        parts.append('Offload')
    return '+'.join(parts) if parts else 'Custom'

###############################################################################
# Stage 0: Configuration.
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')

# 步骤 5：将 --LOD 注入配置，覆盖 configs/instant-mesh-large.yaml 中的 grid_res。
if IS_FLEXICUBES and args.grid_res > 0:
    print(f'Overriding FlexiCubes grid_res from config ({model_config.params.grid_res}) to {args.grid_res}')
    model_config.params.grid_res = args.grid_res

# --Lambda 的动态均衡分块计算“延后”到 Stage 1 跑完之后（见下方 Stage 1 测量处），
# 必须用“实测” Stage1 峰值反解 Chunk，而不是提前用预测值算——否则顺序不合理。

# 指定本地 HF 缓存目录，避免重复下载。
# 优先使用用户环境变量 HF_HUB_CACHE；未设置时按平台给出默认路径（Windows 沿用原 D 盘缓存，Linux 用 ~/.cache/huggingface）。
if 'HF_HUB_CACHE' not in os.environ:
    if sys.platform.startswith('win'):
        os.environ['HF_HUB_CACHE'] = r'D:\AI_HF_CACHE'
    else:
        os.environ['HF_HUB_CACHE'] = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')


def _safe_hf_hub_download(repo_id, filename, repo_type='model', local_fallback=None):
    """带离线回退的权重下载：在线失败（换服务器/无外网）时优先用本地回退文件，
    其次尝试 local_files_only 命中缓存；都失败则给出明确指引而非超时堆栈。"""
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    except Exception as _e:
        if local_fallback and os.path.exists(local_fallback):
            print(f'[warn] 在线下载 {repo_id}/{filename} 失败（{type(_e).__name__}）；'
                  f'回退到本地文件 {local_fallback}')
            return local_fallback
        try:
            os.environ['HF_HUB_OFFLINE'] = '1'
            return hf_hub_download(repo_id=repo_id, filename=filename,
                                   repo_type=repo_type, local_files_only=True)
        except Exception:
            raise RuntimeError(
                f'无法获取 {repo_id}/{filename}：在线下载失败（{type(_e).__name__}），'
                f'且本地缓存/回退均不可用。请将原机器的 HF 缓存拷到 '
                f'{os.environ.get("HF_HUB_CACHE")}，或把权重放到 {local_fallback}'
            )


# load diffusion model
print('Loading diffusion model ...')
# 若项目自带 zero123plus/ 自定义 pipeline 目录，则暴露给 diffusers，
# 使其在无外网时也能从本地加载 custom pipeline（避免离线回退时仍去 hub 拉代码）。
_local_pipeline = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zero123plus")
if os.path.isdir(_local_pipeline) and 'CUSTOM_PIPELINES' not in os.environ:
    os.environ['CUSTOM_PIPELINES'] = _local_pipeline
_repo = "sudo-ai/zero123plus-v1.2"
try:
    pipeline = DiffusionPipeline.from_pretrained(
        _repo,
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
    ).to('cuda')
except Exception as _e:
    # 联网失败（换服务器 / 无外网）时回退到本地缓存，避免 huggingface.co 元数据校验超时
    print(f'[warn] 在线加载 {_repo} 失败（{type(_e).__name__}）；'
          f'回退到本地缓存（HF_HUB_OFFLINE=1, local_files_only=True）...')
    os.environ['HF_HUB_OFFLINE'] = '1'
    pipeline = DiffusionPipeline.from_pretrained(
        _repo,
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to('cuda')
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# 显存优化：VAE分片
pipeline.vae = pipeline.vae.to('cuda')
pipeline.enable_vae_slicing()
pipeline.vae = pipeline.vae.to('cuda')

# load custom white-background UNet
print('Loading custom white-background unet ...')
if os.path.exists(infer_config.unet_path):
    unet_ckpt_path = infer_config.unet_path
else:
    unet_ckpt_path = _safe_hf_hub_download(
        "TencentARC/InstantMesh", "diffusion_pytorch_model.bin",
        local_fallback=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts", "diffusion_pytorch_model.bin"))
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

# Apply structured channel pruning (optional).
if args.prune_ratio > 0:
    print(f'Applying UNet channel pruning (ratio={args.prune_ratio:.2f}) ...')
    print('  Warning: zeroing-based channel pruning without fine-tuning (Step 6) may severely degrade output quality.')
    print('  Use this only for parameter-count ablation, not for production inference.')
    original_weights = count_unet_conv_weights(pipeline.unet)
    apply_unet_channel_pruning(pipeline.unet, args.prune_ratio, make_permanent=True)
    nonzero_weights = count_unet_nonzero_conv_weights(pipeline.unet)
    print(f'  Conv weights: {original_weights/1e6:.2f}M -> {nonzero_weights/1e6:.2f}M ({(1 - nonzero_weights/original_weights)*100:.1f}% pruned)')

# 加载完成后转为半精度，降低推理显存
pipeline.unet = pipeline.unet.to(dtype=torch.float16)
pipeline.vae = pipeline.vae.to(dtype=torch.float16)

pipeline.prepare()

# FlashAttention/SDPA 消融开关：本栈默认即启用 SDPA/Flash 式高效注意力
# （diffusers 0.20.2 的 AttnProcessor2_0）。当 --attn_impl eager 时，在扩散去噪调用处
# 通过 sdp_kernel 关闭 flash/mem-efficient 后端、强制 math 后端，使注意力以完整
# 注意力矩阵计算（无 Flash 显存收益），从而得到"关闭 FlashAttention 扩散优化"的
# 同硬件对照（见论文表 3）。该方式不替换任何 attention processor，避免破坏
# RefOnlyNoisedUNet 的参考特征注入(mode 参数)。
_disable_sdpa = (args.attn_impl == 'eager')
if _disable_sdpa:
    print('[attn] SDPA flash/mem-efficient backends DISABLED (forced math backend) -> '
          'FlashAttention/SDPA diffusion optimization ablated for Table 3.')

# Apply modular FP8 quantization to the diffusion UNet (optional, Step 3).
# 权重以 FP8 存储（显存减半），前向 dequantize 回 FP16 计算；敏感层 conv_in/conv_out/
# time_embedding 不转换。此仿真模式在任意显卡上均可运行。
if args.fp8_unet:
    print('Applying modular FP8 quantization to UNet ...')
    unet_backbone = pipeline.unet.unet  # 实际 UNet2DConditionModel
    before_gb = count_unet_weight_gb(unet_backbone)
    n_layers = apply_fp8_to_unet(unet_backbone)
    after_gb = count_unet_weight_gb(unet_backbone)
    print(f'  UNet weight memory: {before_gb:.2f} GB -> {after_gb:.2f} GB '
          f'({(1 - after_gb / before_gb) * 100:.1f}% saved on {n_layers} layers)')

# Replace self-attention processors with sparse foreground processors (optional).
if args.use_sparse_attn:
    print('Enabling sparse self-attention in UNet ...')
    attn_procs = {}
    for name, proc in pipeline.unet.unet.attn_processors.items():
        if name.endswith('attn1.processor') and hasattr(proc, 'chained_proc'):
            proc.chained_proc = SparseAttnProcessor(
                base_mask=torch.ones(1, 1, 120, 80, device='cuda'),
                base_h=120,
                base_w=80,
                inner_proc=proc.chained_proc,
                mask_mode='latent',
            )
        attn_procs[name] = proc
    pipeline.unet.unet.set_attn_processor(attn_procs)
    print(f'  Replaced {sum(1 for p in pipeline.unet.unet.attn_processors.values() if isinstance(getattr(p, "chained_proc", None), SparseAttnProcessor))} self-attention processors.')

# load reconstruction model
print('Loading reconstruction model ...')

# 将可选的 Triplane 后量化/压缩/分块/卸载开关注入模型配置（Step 3 & 4）。
if args.triplane_quant:
    print(f'Enabling Triplane per-channel quantization ({args.triplane_quant_bits}-bit) ...')
    model_config.params.quantize_triplane = True
    model_config.params.triplane_quant_bits = args.triplane_quant_bits
if args.triplane_compress_dim > 0:
    print(f'Enabling Triplane channel compression ({args.triplane_compress_dim}-dim, down from 80) ...')
    print('  Note: the 1x1 compress conv is randomly initialized; geometry will be degraded')
    print('  until recovered by Step 6 distillation. Use only for memory-ablation studies.')
    model_config.params.compress_triplane_dim = args.triplane_compress_dim
if args.triplane_query_chunk > 0:
    print(f'Enabling chunked Triplane sampling (query_chunk={args.triplane_query_chunk}) ...')
    model_config.params.triplane_query_chunk = args.triplane_query_chunk
if args.cpu_offload_triplane:
    print('Enabling CPU-offloaded Triplane (dynamic chunked GPU query) ...')
    model_config.params.triplane_offload = True
if args.grid_extract_chunk > 0:
    print(f'Enabling chunked Stage-2 grid extraction (grid_extract_chunk={args.grid_extract_chunk}) ...')
    model_config.params.grid_extract_chunk = args.grid_extract_chunk

model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_path = _safe_hf_hub_download(
        "TencentARC/InstantMesh", f"{config_name.replace('-', '_')}.ckpt",
        local_fallback=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts", f"{config_name.replace('-', '_')}.ckpt"))
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
# 通道压缩会引入新的 triplane_compress 权重（不在原 checkpoint 中），需非严格加载。
compress_enabled = args.triplane_compress_dim > 0
missing, unexpected = model.load_state_dict(state_dict, strict=not compress_enabled)
if compress_enabled:
    print(f'  Loaded with strict=False (channel compression on). '
          f'Missing-from-ckpt (random-init) keys: {missing}')
else:
    assert not missing and not unexpected, f'load_state_dict mismatch: missing={missing} unexpected={unexpected}'

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

# make output directories
image_path = os.path.join(args.output_path, config_name, 'images')
mesh_path = os.path.join(args.output_path, config_name, 'meshes')
video_path = os.path.join(args.output_path, config_name, 'videos')
os.makedirs(image_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# 消融实验：把每组 mesh 保存到独立目录，避免各组运行互相覆盖，
# 供 eval_relative.py 以 Baseline 为锚点做「相对质量比较」。
_ablation_group = args.ablation_name if args.ablation_name else derive_group_name(args)
# FlashAttention 消融：关闭 SDPA(eager) 时重命名为独立对比组。
if args.attn_impl == 'eager' and _ablation_group == 'Baseline':
    _ablation_group = 'InstantMesh-noFlashAttn'
_mesh_root = os.path.join(args.ablation_dir, 'meshes', _ablation_group)
os.makedirs(_mesh_root, exist_ok=True)

# process input files
if os.path.isdir(args.input_path):
    input_files = [
        os.path.join(args.input_path, file) 
        for file in os.listdir(args.input_path) 
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp')
    ]
else:
    input_files = [args.input_path]
print(f'Total number of input images: {len(input_files)}')


###############################################################################
# Stage 1: Multiview generation.
###############################################################################

rembg_session = None if args.no_rembg else rembg.new_session()

torch.cuda.reset_peak_memory_stats()
t0_stage1 = time.time()

outputs = []
for idx, image_file in enumerate(input_files):
    name = os.path.basename(image_file).split('.')[0]
    print(f'[{idx+1}/{len(input_files)}] Imagining {name} ...')

    # remove background optionally
    input_image = Image.open(image_file)
    if not args.no_rembg:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)
    
    # sampling
    with (torch.backends.cuda.sdp_kernel(enable_flash=False,
                                          enable_mem_efficient=False,
                                          enable_math=True)
          if _disable_sdpa else contextlib.nullcontext()):
        output_image = pipeline(
            input_image,
            num_inference_steps=args.diffusion_steps,
        ).images[0]

    output_image.save(os.path.join(image_path, f'{name}.png'))
    print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")

    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    outputs.append({'name': name, 'images': images})

stage1_peak_mem = torch.cuda.max_memory_allocated() / 1024**3
stage1_time = time.time() - t0_stage1
# delete pipeline to save memory
del pipeline
torch.cuda.empty_cache()

print(f'[Stage 1] Peak memory: {stage1_peak_mem:.2f} GB, Time: {stage1_time:.2f} s')

# ── 动态均衡分块（--Lambda）：用“实测” Stage1 峰值反解 Chunk，使 Stage2 ≈ Stage1 ──
# 必须在 Stage 1 跑完、拿到真实 stage1_peak_mem 之后才算；提前算用的是预测值，不合理。
# extract_mesh 运行时读 model.grid_extract_chunk（实例属性），故直接 patch 已实例化的模型。
if args.Lambda is not None:
    from src.memory_cost_model import (
        auto_chunk_verts, predict_stage2_extract,
        lambda0_eq, beta_eff, ALPHA_EFF_BY_LOD,
    )
    _g = args.grid_res if args.grid_res > 0 else model_config.params.grid_res
    _G3 = _g ** 3
    _calibrated = _g in ALPHA_EFF_BY_LOD   # 该 LOD 是否已被实测反推校准过
    if args.Lambda < 0:
        # 自动均衡点 λ0：用实测 Stage1 反解 g_eq（绝不写死常量）
        _lambda = lambda0_eq(_g, m_stage1=stage1_peak_mem)
        _verts = auto_chunk_verts(_g, m_stage1=stage1_peak_mem)
        _mode = 'auto-equilibrium λ0'
    else:
        # 用户强制 λ：g = λ·G³（与 LOD 无关，直接按分辨率缩放）
        _lambda = args.Lambda
        if not (0.0 < _lambda <= 1.0):
            print(f'[Lambda] WARNING: λ={_lambda} out of (0, 1]; clamping g to [1k, G³] may be unsafe.')
        _verts = int(round(_lambda * _G3 / 1000.0)) * 1000
        _verts = max(1000, min(_verts, _G3))   # 限制到 [1k, G³]（G³=整批退化）
        _mode = f'forced λ={_lambda}'
    args.grid_extract_chunk = _verts
    model_config.params.grid_extract_chunk = _verts
    model.grid_extract_chunk = _verts      # 直接下发到模型实例（Stage 2 提取时读取）
    args.chunk_level = -1
    _stage2_pred = predict_stage2_extract(_verts, _g)
    print(f'[Lambda] {_mode}: LOD(G)={_g}, λ={_lambda:.4f}, g=λ·G³={_verts} verts')
    print(f'  alpha_eff source: {"measured & calibrated (ALPHA_EFF_BY_LOD)" if _calibrated else "nearest-LOD interpolation fallback (auto-calibrated after run)"}')
    print(f'  Stage1(measured)={stage1_peak_mem:.2f}GB, predicted Stage2={_stage2_pred:.2f}GB')
    if _stage2_pred <= stage1_peak_mem:
        print(f'  -> Stage2<=Stage1, global peak locked under Stage1 ({stage1_peak_mem:.2f}GB), no OOM')
    else:
        print(f'  -> WARNING: λ>λ0 so Stage2 peak ({_stage2_pred:.2f}GB) EXCEEDS Stage1 '
              f'({stage1_peak_mem:.2f}GB); global peak rises to ~{_stage2_pred:.2f}GB '
              f'— ensure GPU has enough free VRAM')

###############################################################################
# Stage 2: Reconstruction.
###############################################################################

torch.cuda.reset_peak_memory_stats()
t0_stage2 = time.time()

input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device)
chunk_size = 20 if IS_FLEXICUBES else 1

for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    # Select subset of views according to --view
    indices = get_view_indices(args.view).to(device)
    images = images[:, indices]
    cameras = input_cameras[:, indices]

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, cameras)

        final_grid_res = model_config.params.grid_res

        # 步骤 5：可选快速预览模式，先用低分辨率 grid_res 提取粗网格供用户快速查看，
        # 确认几何无误后再用最终 grid_res 输出精细网格。
        if args.preview:
            mesh_path_preview = os.path.join(_mesh_root, f'{name}_preview.obj')
            preview_out = model.extract_mesh(
                planes,
                use_texture_map=False,
                grid_res=args.preview_grid_res,
                simplify_ratio=0.0,
                **infer_config,
            )
            pv, pf, pv_colors = preview_out
            save_obj(pv, pf, pv_colors, mesh_path_preview)
            print(f"[preview] Coarse mesh (grid_res={args.preview_grid_res}) saved to {mesh_path_preview}")

        # get mesh (final, at configured grid_res)
        # 主存位置：用户指定的 --output_path/config_name/meshes/（即便携可预期路径）。
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')
        # 兼容消融评估：同时存一份到消融分组目录（eval_relative.py 以 Baseline 为锚点对比）。
        mesh_path_ablation = os.path.join(_mesh_root, f'{name}.obj')

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_texmap,
            grid_res=final_grid_res,
            simplify_ratio=args.simplify_ratio,
            **infer_config,
        )
        if args.export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)
        # 复制到消融分组目录，供 eval_relative.py 使用。
        shutil.copyfile(mesh_path_idx, mesh_path_ablation)
        print(f"Mesh saved to {mesh_path_idx}")
        print(f"Mesh (ablation copy) saved to {mesh_path_ablation}")

stage2_peak_mem = torch.cuda.max_memory_allocated() / 1024**3
stage2_time = time.time() - t0_stage2
print(f'[Stage 2] Peak memory: {stage2_peak_mem:.2f} GB, Time: {stage2_time:.2f} s')

# [Lambda] 实测反推 α_eff，直接写回 ALPHA_EFF_BY_LOD 源码字典（下次运行自动采用，无需手改）
if args.Lambda is not None:
    from src.memory_cost_model import (
        beta_eff, alpha_eff, predict_stage2_extract, persist_alpha,
    )
    _g2 = args.grid_res if args.grid_res > 0 else model_config.params.grid_res
    _model_alpha = alpha_eff(_g2)   # 本次求解时用的旧 α（写回前）
    _realized_alpha = (stage2_peak_mem - beta_eff(_g2)) / max(args.grid_extract_chunk, 1)
    _pred_stage2 = predict_stage2_extract(args.grid_extract_chunk, _g2)
    # 自标定：用真 Stage2 反推实测 α_eff，写回 ALPHA_EFF_BY_LOD[G]，下次即采用真值
    _ok = persist_alpha(_g2, _realized_alpha)
    print(f'[Lambda] calibration check: realized alpha_eff={_realized_alpha:.3e} GB/vert '
          f'(prev alpha={_model_alpha:.3e}); predicted Stage2={_pred_stage2:.2f}GB vs measured {stage2_peak_mem:.2f}GB')
    print(f'[Lambda] realized alpha_eff(LOD{_g2})={_realized_alpha:.3e} '
          f'{"written back to ALPHA_EFF_BY_LOD[%d], auto-used next run" % _g2 if _ok else "write-back failed (no impact on this run)"})')

# Free memory used by mesh extraction before the heavy video rendering starts
torch.cuda.empty_cache()

###############################################################################
# Stage 3: Rendering (optional).
###############################################################################

t0_stage3 = time.time()
for idx, sample in enumerate(outputs):
    name = sample['name']

    # get video
    if args.save_video:
        video_path_idx = os.path.join(video_path, f'{name}.mp4')
        render_size = infer_config.render_resolution
        render_cameras = get_render_cameras(
            batch_size=1, 
            M=120, 
            radius=args.distance, 
            elevation=20.0,
            is_flexicubes=IS_FLEXICUBES,
        ).to(device)
        
        # Video rendering is memory-intensive; use smaller chunk size to avoid OOM on 8GB cards
        video_chunk_size = 1 if IS_FLEXICUBES else chunk_size
        frames = render_frames(
            model, 
            planes, 
            render_cameras=render_cameras, 
            render_size=render_size, 
            chunk_size=video_chunk_size, 
            is_flexicubes=IS_FLEXICUBES,
        )

        save_video(
            frames,
            video_path_idx,
            fps=30,
        )
        print(f"Video saved to {video_path_idx}")
stage3_time = time.time() - t0_stage3

###############################################################################
# Stage 4: Record runtime metrics + Chart generation (optional).
###############################################################################

# Always record the real measured peak memory / inference time so the ablation
# charts can be filled automatically (no manual JSON editing required).
try:
    from chart_ablation import record_run
    group_name = _ablation_group
    total_time = stage1_time + stage2_time + (stage3_time if args.save_video else 0.0)
    metric = {
        'group': group_name,
        'config': config_name,
        'input': os.path.basename(args.input_path),
        'mesh_dir': _mesh_root,
        'mem': max(stage1_peak_mem, stage2_peak_mem),  # overall per-run peak
        'stage1_mem': stage1_peak_mem,
        'stage2_mem': stage2_peak_mem,
        'time': total_time,
        'stage1_time': stage1_time,
        'stage2_time': stage2_time,
        'stage3_time': stage3_time if args.save_video else 0.0,
        'flags': ' '.join([
            f'--view {args.view}',
            '--use_sparse_attn' if args.use_sparse_attn else '',
            '--FP8Unet' if args.fp8_unet else '',
            f'--LOD {args.grid_res}' if args.grid_res > 0 else '',
            f'--Simplify {args.simplify_ratio}' if args.simplify_ratio > 0 else '',
            (f'--Chunk {args.chunk_level}' if args.chunk_level > 0
             else (f'--grid_extract_chunk {args.grid_extract_chunk}' if args.grid_extract_chunk > 0 else '')),
            (f'--Lambda {args.Lambda:.4f}' if args.Lambda is not None and args.Lambda >= 0
             else ('--Lambda' if args.Lambda is not None else '')),
            '--cpu_offload_triplane' if args.cpu_offload_triplane else '',
        ]).strip(),
    }
    record_run(args.ablation_dir, metric)
    print(f'[Metrics] Recorded group "{group_name}" -> peak {metric["mem"]:.2f} GB, '
          f'time {metric["time"]:.2f} s (saved to {args.ablation_dir}/run_metrics.jsonl)')
except Exception as e:
    print(f'[Metrics] WARNING: failed to record runtime metrics: {e}')

if args.gen_charts:
    import json
    from chart_ablation import generate_all_charts, aggregate_runs

    results = None
    if args.chart_results and os.path.exists(args.chart_results):
        with open(args.chart_results, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f'Generating charts from: {args.chart_results}')
    else:
        # Prefer real auto-recorded runs; fall back to built-in template data.
        agg = aggregate_runs(out_dir=args.ablation_dir)
        if agg is not None:
            results = agg
            print(f'Aggregated {len(agg["groups"])} auto-recorded runs from {args.ablation_dir}.')
            if agg.get('quality_relative'):
                print('NOTE: CD / F-Score are RELATIVE-to-Baseline values (geometric deviation from the')
                print('      highest-fidelity Baseline mesh), computed by eval_relative.py. They measure how')
                print('      much each optimization deviates from Baseline, NOT absolute reconstruction error.')
            elif agg.get('quality_from_default'):
                print('NOTE: CD / F-Score / LPIPS use built-in template values (no ground-truth mesh).')
                print('      Run eval_relative.py for Baseline-relative values, or add GT evaluation.')
        elif args.chart_results:
            print(f'Warning: --chart_results "{args.chart_results}" not found, using built-in default data.')
            from chart_ablation import _DEFAULT_DATA
            results = _DEFAULT_DATA
        else:
            print('No auto-recorded runs found and no --chart_results; using built-in default data.')
            print('IMPORTANT: Replace template numbers with your actual measured values before paper use.')
            from chart_ablation import _DEFAULT_DATA
            results = _DEFAULT_DATA

    generate_all_charts(results, out_dir=os.path.join(args.output_path, 'charts'))
