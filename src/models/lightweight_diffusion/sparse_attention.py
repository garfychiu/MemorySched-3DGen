import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers.models.attention_processor import Attention, AttnProcessor2_0


def infer_hw(num_tokens: int, base_h: int, base_w: int):
    """
    Infer spatial height/width from flattened token count under a fixed aspect ratio.
    """
    ratio = base_h / base_w
    w = int(round(math.sqrt(num_tokens / ratio)))
    h = num_tokens // w
    while h * w != num_tokens and w > 0:
        w -= 1
        h = num_tokens // w
    if h * w != num_tokens:
        return None, None
    return h, w


def build_multiview_mask(
    alpha_image: Optional[Union[Image.Image, np.ndarray, torch.Tensor]],
    base_h: int = 120,
    base_w: int = 80,
    view_h: int = 40,
    view_w: int = 40,
    threshold: float = 0.5,
    device: Union[str, torch.device] = "cpu",
):
    """
    Build a (1, 1, base_h, base_w) foreground mask for the 6-view latent grid (3x2).

    If `alpha_image` is None, returns an all-ones mask (no sparsity, dense fallback).
    """
    if alpha_image is None:
        return torch.ones(1, 1, base_h, base_w, device=device)

    if isinstance(alpha_image, Image.Image):
        if alpha_image.mode in ("L", "1", "I", "F"):
            # e.g. input_image.getchannel("A") -> single-channel grayscale alpha.
            alpha = torch.from_numpy(np.array(alpha_image).astype(np.float32))
            if alpha.max() > 1.0:
                alpha = alpha / 255.0
        elif alpha_image.mode == "RGBA":
            a = alpha_image.getchannel("A")
            alpha = torch.from_numpy(np.array(a).astype(np.float32) / 255.0)
        elif alpha_image.mode in ("RGB", "P"):
            # No alpha: use brightness as a foreground proxy (white background -> 0).
            rgb = alpha_image.convert("RGB")
            arr = np.array(rgb).astype(np.float32) / 255.0
            lum = arr.mean(axis=2)
            alpha = torch.from_numpy((1.0 - lum).clip(0, 1))
        else:
            alpha = torch.ones(alpha_image.size[1], alpha_image.size[0])
    elif isinstance(alpha_image, torch.Tensor):
        alpha = alpha_image.detach().cpu()
    elif isinstance(alpha_image, np.ndarray):
        alpha = torch.from_numpy(alpha_image)
    else:
        raise TypeError(f"Unsupported alpha_image type: {type(alpha_image)}")

    if alpha.dim() == 2:
        alpha = alpha.unsqueeze(0).unsqueeze(0)
    elif alpha.dim() == 3:
        alpha = alpha.unsqueeze(0)
    elif alpha.dim() == 4:
        alpha = alpha
    else:
        raise ValueError(f"Unsupported alpha rank: {alpha.dim()}")

    # If alpha is 4-channel, take the last channel as alpha.
    if alpha.shape[1] == 4:
        alpha = alpha[:, 3:4]

    if alpha.max() > 1.0:
        alpha = alpha / 255.0

    # Resize alpha to a single view's latent size, then tile to 3x2 grid.
    alpha = F.interpolate(alpha, size=(view_h, view_w), mode="bilinear", align_corners=False)
    tiled = alpha.repeat(1, 1, 3, 2)  # (1, 1, 3*view_h, 2*view_w)

    # Crop / pad to exact base size.
    if tiled.shape[2] > base_h:
        tiled = tiled[:, :, :base_h, :]
    if tiled.shape[3] > base_w:
        tiled = tiled[:, :, :, :base_w]
    if tiled.shape[2] < base_h or tiled.shape[3] < base_w:
        pad_h = base_h - tiled.shape[2]
        pad_w = base_w - tiled.shape[3]
        tiled = F.pad(tiled, (0, pad_w, 0, pad_h))

    mask = (tiled >= threshold).float().to(device)
    return mask


class SparseAttnProcessor(nn.Module):
    """
    Masked self-attention processor: only attends on foreground tokens.

    Compatible with the diffusers `Attention` API. The `inner_proc` is typically an
    `AttnProcessor2_0` wrapped by `ReferenceOnlyAttnProc.chained_proc`.
    """

    def __init__(
        self,
        base_mask: torch.Tensor = None,
        base_h: int = 120,
        base_w: int = 80,
        inner_proc=None,
        min_sparse_tokens: int = 0,
        mask_mode: str = 'latent',
        keep_frac: float = 0.6,
        noise_cv_thresh: float = 0.1,
    ):
        super().__init__()
        if base_mask is None:
            base_mask = torch.ones(1, 1, base_h, base_w)
        if base_mask.dim() == 2:
            base_mask = base_mask.unsqueeze(0).unsqueeze(0)
        elif base_mask.dim() == 3:
            base_mask = base_mask.unsqueeze(0)
        self.register_buffer("base_mask", base_mask)
        self.base_h = base_h
        self.base_w = base_w
        self.inner_proc = AttnProcessor2_0() if inner_proc is None else inner_proc
        self.min_sparse_tokens = min_sparse_tokens
        # 'latent' : foreground mask derived per-step from hidden_states (default,
        #           correct per view, no input contour needed).
        # 'fixed'  : foreground mask from a pre-registered base_mask (input contour),
        #           kept only as an internal fallback / ablation comparison.
        self.mask_mode = mask_mode
        self.keep_frac = keep_frac
        self.noise_cv_thresh = noise_cv_thresh

    def _get_layer_mask(self, num_tokens: int, device: torch.device, dtype: torch.dtype):
        H, W = infer_hw(num_tokens, self.base_h, self.base_w)
        if H is None or W is None:
            return None
        mask = F.interpolate(self.base_mask.to(device, dtype=dtype), size=(H, W), mode="nearest")
        return mask.view(-1)  # (N,)

    def _get_latent_mask(self, num_tokens: int, hidden_states, device: torch.device):
        """
        Estimate the foreground mask directly from the current 6-view latents.

        Uniform regions (white background) have near-zero local structure energy,
        while object edges/texture have high energy, so a per-token structure score
        `mean_c((x - blur(x))^2)` cleanly separates them -- and it is correct
        *per view* without ever knowing the view layout. An early-step noise gate
        (low coefficient of variation across tokens) falls back to dense attention
        until the signal becomes coherent enough to trust.
        """
        H, W = infer_hw(num_tokens, self.base_h, self.base_w)
        if H is None or W is None:
            return None
        B, N, C = hidden_states.shape
        x = hidden_states.to(torch.float32).permute(0, 2, 1).reshape(B, C, H, W)
        blurred = F.avg_pool2d(x, 3, stride=1, padding=1)
        score = (x - blurred).pow(2).mean(dim=1)              # (B, H, W)
        score = score.reshape(B, N)
        fg = torch.empty(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            s = score[b]
            cv = s.std() / (s.mean() + 1e-8)                  # low -> near pure noise
            if cv < self.noise_cv_thresh:
                fg[b] = True                                  # dense fallback
            else:
                thr = torch.quantile(s, self.keep_frac)
                fg[b] = s >= thr
        return fg

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        B, N, C = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype

        # --- Determine foreground mask (per batch element) ---
        if self.mask_mode == 'latent':
            fg_mask = self._get_latent_mask(N, hidden_states, device)   # (B, N) or None
        else:
            layer_mask = self._get_layer_mask(N, device, dtype)
            if layer_mask is None:
                return self.inner_proc(
                    attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs
                )
            fg_mask = (layer_mask >= 0.5).unsqueeze(0).expand(B, -1)

        if fg_mask is None:
            return self.inner_proc(
                attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs
            )

        # CFG runs cond+uncond on identical latents, so their masks are identical;
        # use the batch-0 foreground set as the representative for every element.
        rep = fg_mask[0]
        n_fg = int(rep.sum())

        # Fallback when mask is all-foreground / all-background / too small.
        if n_fg == 0 or n_fg == N or n_fg < self.min_sparse_tokens:
            return self.inner_proc(
                attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs
            )

        # Keep all query positions; restrict keys/values to foreground tokens only.
        # Every position still receives an attention output, but it can only attend
        # to valid (foreground) tokens, which saves the K/V projection and the
        # attention-matrix memory.
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        K = encoder_hidden_states.shape[1]

        if K == N:
            fg_indices = torch.where(rep)[0]
            # Gather within each batch element independently so offsets never
            # exceed dim-1 (= N); this is correct for B>1 (e.g. CFG cond+uncond).
            chunks = [encoder_hidden_states[b:b+1, fg_indices] for b in range(B)]
            encoder_hidden_states_fg = torch.cat(chunks, dim=0)   # (B, n_fg, C)
        elif K % N == 0:
            # e.g. ReferenceOnlyAttnProc mode='r' concatenates [current, ref].
            m = K // N
            idx = torch.where(rep)[0]
            chunks = []
            for b in range(B):
                enc_b = encoder_hidden_states[b]             # (K, C)
                gathered = torch.cat(
                    [enc_b[i * N : (i + 1) * N][idx] for i in range(m)], dim=0
                ).unsqueeze(0)                               # (1, m*n_fg, C)
                chunks.append(gathered)
            encoder_hidden_states_fg = torch.cat(chunks, dim=0)
        else:
            # Unknown key layout (e.g. cross-attention text embeddings) -> dense.
            return self.inner_proc(
                attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs
            )

        # Compute attention for ALL queries against foreground keys/values only.
        output = self.inner_proc(
            attn, hidden_states, encoder_hidden_states_fg, attention_mask, **kwargs
        )
        return output


def set_sparse_attention_mask(pipeline, mask: torch.Tensor):
    """
    Update the shared foreground mask on every SparseAttnProcessor instance.

    `pipeline` must already have been wrapped by `RefOnlyNoisedUNet` (i.e. after
    `pipeline.prepare()`), and `replace_self_attention_with_sparse()` must have
    been called.
    """
    unet = pipeline.unet.unet if hasattr(pipeline.unet, "unet") else pipeline.unet
    updated = 0
    for proc in unet.attn_processors.values():
        chained = getattr(proc, "chained_proc", None)
        if isinstance(chained, SparseAttnProcessor):
            # Move mask to the same device as the registered buffer.
            chained.base_mask = mask.to(chained.base_mask.device)
            updated += 1
    return updated
