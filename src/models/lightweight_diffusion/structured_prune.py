import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def _iter_conv2d_modules(unet):
    """Iterate over Conv2d modules inside the UNet backbone."""
    yield unet.conv_in
    yield unet.conv_out
    for block in unet.down_blocks:
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                yield module
    for module in unet.mid_block.modules():
        if isinstance(module, nn.Conv2d):
            yield module
    for block in unet.up_blocks:
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                yield module


def apply_unet_channel_pruning(unet, prune_ratio: float, make_permanent: bool = True):
    """
    Apply structured L1 channel pruning to UNet Conv2d layers.

    Args:
        unet: A `UNet2DConditionModel` (not the RefOnlyNoisedUNet wrapper).
        prune_ratio: Ratio of output channels to prune, e.g. 0.3.
        make_permanent: If True, removes the pruning mask and keeps zeroed weights.
    """
    if prune_ratio <= 0:
        return 0

    pruned_count = 0
    for module in _iter_conv2d_modules(unet):
        # Skip small / grouped convolutions that are fragile to prune.
        if module.out_channels <= 8:
            continue
        if module.groups > 1 and module.groups != module.in_channels:
            continue
        try:
            prune.ln_structured(module, name="weight", amount=prune_ratio, n=1, dim=0)
            pruned_count += 1
        except Exception as e:
            print(f"Warning: failed to prune {module}: {e}")

    if make_permanent:
        make_pruning_permanent(unet)

    return pruned_count


def make_pruning_permanent(unet):
    """Remove pruning masks and keep zeroed weights."""
    for module in _iter_conv2d_modules(unet):
        if hasattr(module, "weight_mask"):
            try:
                prune.remove(module, "weight")
            except Exception:
                pass


def count_unet_conv_weights(unet):
    """Count total Conv2d weight parameters in the UNet."""
    total = 0
    for module in _iter_conv2d_modules(unet):
        total += module.weight.numel()
    return total


def count_unet_nonzero_conv_weights(unet):
    """Count non-zero Conv2d weight parameters (after pruning)."""
    nonzero = 0
    for module in _iter_conv2d_modules(unet):
        nonzero += (module.weight != 0).sum().item()
    return nonzero


def count_unet_parameters(unet):
    """Return (total_params, nonzero_conv_params)."""
    total = sum(p.numel() for p in unet.parameters())
    nonzero = count_unet_nonzero_conv_weights(unet)
    return total, nonzero
