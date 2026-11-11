# Copyright (c) 2023, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn

from src.models.lightweight_diffusion.fp8_quantize import quantize_triplane_channels


class BasicTransformerBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition and another modulation vector applied to sub-blocks.
    """
    # use attention from torch.nn.MultiHeadAttention
    # Block contains a cross-attention layer, a self-attention layer, and a MLP
    def __init__(
        self, 
        inner_dim: int, 
        cond_dim: int, 
        num_heads: int, 
        eps: float,
        attn_drop: float = 0., 
        attn_bias: bool = False,
        mlp_ratio: float = 4., 
        mlp_drop: float = 0.,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(inner_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        x = x + self.cross_attn(self.norm1(x), cond, cond)[0]
        before_sa = self.norm2(x)
        x = x + self.self_attn(before_sa, before_sa, before_sa)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class TriplaneTransformer(nn.Module):
    """
    Transformer with condition that generates a triplane representation.
    
    Reference:
    Timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L486
    """
    def __init__(
        self, 
        inner_dim: int, 
        image_feat_dim: int,
        triplane_low_res: int, 
        triplane_high_res: int, 
        triplane_dim: int,
        num_layers: int, 
        num_heads: int,
        eps: float = 1e-6,
        quantize_triplane: bool = False,
        triplane_quant_bits: int = 8,
        compress_triplane_dim: int = 0,
    ):
        super().__init__()

        # attributes
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim
        # 目标压缩维度：0 表示不压缩（保持原始 triplane_dim）。
        # 设 48 / 32 等可把 80 通道降到该值（Step 4 通道维度压缩）。
        self.compress_triplane_dim = int(compress_triplane_dim)

        # 可选：对输出 planes 做 per-channel 低比特后量化（模拟存储影响）
        self.quantize_triplane = quantize_triplane
        self.triplane_quant_bits = triplane_quant_bits

        # 实际传给下游 OSGDecoder 的通道数（压缩后）。
        out_dim = self.compress_triplane_dim if self.compress_triplane_dim > 0 else triplane_dim

        # modules
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, inner_dim) * (1. / inner_dim) ** 0.5)
        self.layers = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim=inner_dim, cond_dim=image_feat_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.deconv = nn.ConvTranspose2d(inner_dim, triplane_dim, kernel_size=2, stride=2, padding=0)
        # 可选：deconv 输出后对通道做 1x1 卷积降维（需要 Step 6 蒸馏恢复精度）。
        if self.compress_triplane_dim > 0:
            assert self.compress_triplane_dim < triplane_dim, \
                f'compress_triplane_dim ({self.compress_triplane_dim}) must be < triplane_dim ({triplane_dim})'
            self.triplane_compress = nn.Conv2d(triplane_dim, self.compress_triplane_dim, kernel_size=1)
        else:
            self.triplane_compress = None

    def forward(self, image_feats):
        # image_feats: [N, L_cond, D_cond]

        N = image_feats.shape[0]
        H = W = self.triplane_low_res
        L = 3 * H * W

        x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
        for layer in self.layers:
            x = layer(x, image_feats)
        x = self.norm(x)

        # separate each plane and apply deconv
        x = x.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.deconv(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()

        # 可选：通道维度压缩（1x1 卷积），把 triplane_dim 降到 compress_triplane_dim。
        # 随机初始化，需要 Step 6 蒸馏恢复精度；此处仅改变张量形状。
        if self.triplane_compress is not None:
            n, p, c, h, w = x.shape
            x = x.reshape(n * p, c, h, w)
            x = self.triplane_compress(x)
            x = x.reshape(n, p, *x.shape[-3:]).contiguous()

        # 可选：输出后做 per-channel 低比特后量化（不影响 Transformer 权重）
        if self.quantize_triplane:
            x = quantize_triplane_channels(x, self.triplane_quant_bits)

        return x
