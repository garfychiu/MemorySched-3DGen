"""
Step 3: 分模块 FP8 量化 (modular FP8 quantization).

差异化精度策略：
  * 扩散 UNet 对量化鲁棒 -> 将 Conv2d / Linear 的 weight 转成 FP8 存储。
  * LRM 重建 Transformer 对精度敏感 -> 不量化其权重，只对生成的 Triplane
    特征张量做 per-channel 后量化（模拟低比特存储对几何的影响）。

FP8 存储 / FP16 计算（仿真模式）：
  权重以 torch.float8_e4m3fn 保存（显存减半），前向时先 dequantize 回 FP16
  再做标准 matmul / conv。这样：
    - 在任何显卡上都能运行（无需 Ada/Hopper 的真实 FP8 matmul 硬件）；
    - 同时可验证 "weight 走 FP8" 对扩散输出 latent 漂移的影响。
  在 Ada Lovelace / Hopper 上若需进一步加速，可把 forward 换成真实 FP8 matmul。

敏感层保护：
  UNet: conv_in, conv_out, time_embedding*（含其 Linear）保持 FP16，不转换。
  LRM:  Transformer 权重全不量化；仅对输出 planes 做可选后量化。
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F


# 默认跳过的敏感层（名称精确匹配或前缀匹配）。
_SKIP_EXACT = ("conv_in", "conv_out")
_SKIP_PREFIX = ("time_embedding",)


def is_fp8_supported() -> bool:
    """
    判断当前 GPU 是否支持真实 FP8 matmul（Ada Lovelace 8.9+ / Hopper 9.0+）。
    注意：本模块采用 FP8 仿真（dequant->FP16 计算），即便返回 False 也能运行，
    只是不会用到硬件 FP8 加速。
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)


# ---------------------------------------------------------------------------
# FP8 权重存储 + FP16 计算的自定义 forward
# ---------------------------------------------------------------------------

def _fp8_linear_forward(self, x):
    w = self.weight.to(x.dtype)  # dequantize FP8 -> FP16
    return F.linear(x, w, self.bias)


def _fp8_conv_forward(self, x):
    w = self.weight.to(x.dtype)
    return F.conv2d(
        x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
    )


def _should_skip(name: str) -> bool:
    if name in _SKIP_EXACT:
        return True
    for prefix in _SKIP_PREFIX:
        if name.startswith(prefix):
            return True
    return False


def apply_fp8_to_unet(unet, skip_layers=None, verbose: bool = True) -> int:
    """
    将 UNet 中 Conv2d / Linear 的 weight 转成 FP8（float8_e4m3fn），bias 保留 FP16。
    前向时 weight 先转回 FP16 再与 FP16 输入做计算。敏感层被跳过。

    Args:
        unet: UNet2DConditionModel（即 pipeline.unet.unet，实际的 backbone）。
        skip_layers: 额外的需要跳过的模块名集合（补充内置敏感层）。
        verbose: 是否打印转换统计信息。
    Returns:
        被转换的层数量。
    """
    fp8_dtype = torch.float8_e4m3fn
    skip = set(_SKIP_EXACT) | set(_SKIP_PREFIX)
    if skip_layers:
        skip.update(skip_layers)

    converted = 0
    for name, module in unet.named_modules():
        if _should_skip(name):
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 以 FP8 存储权重（显存减半），删除原 FP16 Parameter。
            w_fp16 = module.weight.data.to(torch.float16)
            w_fp8 = w_fp16.to(fp8_dtype)
            del module.weight
            module.register_parameter(
                "weight", nn.Parameter(w_fp8, requires_grad=False)
            )
            # bias 保留 FP16
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.float16)
            # 替换 forward 为 dequantize -> compute
            if isinstance(module, nn.Linear):
                module.forward = types.MethodType(_fp8_linear_forward, module)
            else:
                module.forward = types.MethodType(_fp8_conv_forward, module)
            converted += 1

    if verbose:
        mode = "real FP8 matmul capable" if is_fp8_supported() else "FP8->FP16 simulation"
        print(f"  [FP8] Converted {converted} UNet layers to FP8 ({mode}).")
        print("  [FP8] Skipped sensitive layers: conv_in, conv_out, time_embedding*")
    return converted


# ---------------------------------------------------------------------------
# Triplane 通道量化（LRM 分支，后量化模拟）
# ---------------------------------------------------------------------------

def quantize_triplane_channels(planes: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """
    对 Triplane 特征做 per-channel 均匀(线性)量化，再反量化回原 dtype。

    每个通道（triplane_dim，即 C）独立统计其空间维度上的 min/max，量化到 n_bits，
    再反量化。返回的张量形状/dtype 不变，仅用于验证低比特存储对重建质量的影响
    （运行期不会真正以 int8 存储，而是以 FP16 喂给渲染器）。

    Args:
        planes: [N, 3, C, H, W] triplane 特征张量。
        n_bits: 量化位宽（如 8）。>=16 或 None 表示不量化（直接返回）。
    Returns:
        量化并反量化后的 planes（与原 dtype 一致，计算用 FP16）。
    """
    if n_bits is None or n_bits >= 16:
        return planes

    orig_dtype = planes.dtype
    x = planes.float()
    N, P, C, H, W = x.shape
    x = x.reshape(N * P, C, H * W)

    xmin = x.amin(dim=2, keepdim=True)
    xmax = x.amax(dim=2, keepdim=True)
    qmin, qmax = 0, (1 << n_bits) - 1
    scale = (xmax - xmin) / (qmax - qmin)
    # 常量通道（scale==0）避免除零
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero = torch.round(-xmin / scale).clamp(qmin, qmax)
    xq = torch.clamp(torch.round(x / scale) + zero, qmin, qmax)
    xdq = (xq - zero) * scale  # 反量化回 float

    xdq = xdq.reshape(N, P, C, H, W)
    return xdq.to(orig_dtype)


# ---------------------------------------------------------------------------
# 显存统计辅助（用于验证权重减半）
# ---------------------------------------------------------------------------

def count_unet_weight_gb(unet) -> float:
    """统计 UNet 全部 Parameter 的显存占用（GB）。"""
    total_bytes = 0
    for p in unet.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 3)
