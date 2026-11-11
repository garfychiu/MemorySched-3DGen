# Copyright (c) 2023, Tencent Inc
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

"""
Dynamic chunked CPU-GPU hybrid access for Triplane feature tensors (Step 4).

Motivation (optimization_plan.md, Stage 2 peak memory):
    The Triplane tensor `planes: [N, 3, D, H, W]` plus all intermediate sampled
    features for the full mesh vertices are a major Stage-2 peak contributor.
    `FlexiCubes` grid extraction and texture rendering both resample the same
    planes many times. Keeping the *entire* planes tensor pinned on the GPU while
    also materializing every vertex query at once doubles the footprint.

This helper keeps `planes` resident on **CPU** and exposes `query_planes`, a
chunked resampler that streams only the currently-needed coordinate block to the
GPU, samples via `sample_from_planes`, and immediately releases the GPU copy.
The planes tensor itself is never uploaded as a whole, so it does not add to the
GPU peak during the query.
"""

import torch

from .utils.renderer import sample_from_planes


class CpuOffloadedTriplane(torch.nn.Module):
    """
    Wraps a Triplane tensor so the full tensor lives on CPU.

    All sampling happens in chunks of `query_chunk` coordinates: each chunk is
    moved to the GPU, sampled against the (GPU-uploaded-per-chunk) planes, and the
    block of GPU memory is released before the next chunk. This bounds the peak
    GPU memory to roughly one chunk's worth of sampled features instead of the
    whole mesh.

    Note: when `planes` is already tiny (the typical InstantMesh case), the
    dominant memory is the *decoder MLP* activations and the FlexiCubes grid, not
    the planes themselves. CPU offload still helps when the planes are enlarged
    (e.g. after channel decompression, or high-res triplanes) or when several
    blocks must coexist. It is safe to use unconditionally; it simply adds
    chunked H2D copies that PyTorch overlaps with compute.
    """

    def __init__(self, planes: torch.Tensor, query_chunk: int = 2 ** 20, box_warp: float = 2.0):
        """
        :param planes: triplane features, shape [N, 3, D, H, W]. Stored on CPU.
        :param query_chunk: max number of coordinates (M) processed per chunk.
        :param box_warp: box warp passed through to `sample_from_planes`.
        """
        super().__init__()
        # Force CPU storage; hwc_transpose aside, keep original dtype (fp16).
        self.register_buffer('planes_cpu', planes.detach().to('cpu').contiguous(), persistent=False)
        self.query_chunk = int(query_chunk)
        self.box_warp = box_warp
        # plane_axes is a tiny constant; lift to GPU lazily per query.
        self._plane_axes = None

    @property
    def shape(self):
        return tuple(self.planes_cpu.shape)

    @property
    def device(self):
        # The "logical" device is CPU (the resident copy). Queries happen on GPU.
        return torch.device('cpu')

    def to(self, *args, **kwargs):
        # Keep the resident copy on CPU regardless of .to() calls so we never
        # accidentally pin the whole planes on the GPU.
        return self

    def query_planes(self, sample_coordinates: torch.Tensor, device: torch.device):
        """
        Chunked resampling of `planes` at `sample_coordinates`.

        :param sample_coordinates: [N, M, 3] (GPU or CPU). Chunked along M.
        :param device: target GPU device for the sampling kernels.
        :return: [N, 3, M, C] sampled features on `device`.
        """
        if self._plane_axes is None:
            from .utils.renderer import generate_planes
            self._plane_axes = generate_planes()

        coords = sample_coordinates.to(device) if sample_coordinates.device != device else sample_coordinates
        plane_axes = self._plane_axes.to(dtype=coords.dtype, device=device)
        N, M, _ = coords.shape
        if M <= self.query_chunk:
            # Single-shot: still stream planes per call so the full tensor is not
            # pinned on the GPU between queries.
            planes_gpu = self.planes_cpu.to(device)
            out = sample_from_planes(plane_axes, planes_gpu, coords, padding_mode='zeros', box_warp=self.box_warp)
            del planes_gpu
            return out

        chunks = []
        for start in range(0, M, self.query_chunk):
            end = min(start + self.query_chunk, M)
            coords_chunk = coords[:, start:end]
            planes_gpu = self.planes_cpu.to(device)  # per-chunk upload
            out_chunk = sample_from_planes(
                plane_axes, planes_gpu, coords_chunk, padding_mode='zeros', box_warp=self.box_warp)
            chunks.append(out_chunk.to(device))
            del planes_gpu, out_chunk
        return torch.cat(chunks, dim=2)

    def forward(self, sample_coordinates: torch.Tensor, device: torch.device):
        return self.query_planes(sample_coordinates, device)
