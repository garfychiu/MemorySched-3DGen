# ORIGINAL LICENSE
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Modified by Jiale Xu
# The modifications are subject to the same license as the original.

import itertools
import torch
import torch.nn as nn

from .utils.renderer import generate_planes, project_onto_planes, sample_from_planes
from .cpu_offload import CpuOffloadedTriplane


class OSGDecoder(nn.Module):
    """
    Triplane decoder that gives RGB and sigma values from sampled features.
    Using ReLU here instead of Softplus in the original implementation.
    
    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
    """
    def __init__(self, n_features: int,
                 hidden_dim: int = 64, num_layers: int = 4, activation: nn.Module = nn.ReLU):
        super().__init__()

        self.net_sdf = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 1),
        )
        self.net_rgb = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 3),
        )
        self.net_deformation = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 3),
        )
        self.net_weight = nn.Sequential(
            nn.Linear(8 * 3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 21),
        )

        # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def get_geometry_prediction(self, sampled_features, flexicubes_indices):
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)

        sdf = self.net_sdf(sampled_features)
        deformation = self.net_deformation(sampled_features)

        grid_features = torch.index_select(input=sampled_features, index=flexicubes_indices.reshape(-1), dim=1)
        grid_features = grid_features.reshape(
            sampled_features.shape[0], flexicubes_indices.shape[0], flexicubes_indices.shape[1] * sampled_features.shape[-1])
        weight = self.net_weight(grid_features) * 0.1

        return sdf, deformation, weight
    
    def get_texture_prediction(self, sampled_features):
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)

        rgb = self.net_rgb(sampled_features)
        rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001  # Uses sigmoid clamping from MipNeRF

        return rgb


class TriplaneSynthesizer(nn.Module):
    """
    Synthesizer that renders a triplane volume with planes and a camera.
    
    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L19
    """

    DEFAULT_RENDERING_KWARGS = {
        'ray_start': 'auto',
        'ray_end': 'auto',
        'box_warp': 2.,
        'white_back': True,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'sampler_bbox_min': -1.,
        'sampler_bbox_max': 1.,
    }

    def __init__(self, triplane_dim: int, samples_per_ray: int, query_chunk: int = 0):
        super().__init__()

        # attributes
        self.triplane_dim = triplane_dim
        self.query_chunk = int(query_chunk)  # 0 => sample all coordinates at once
        self.rendering_kwargs = {
            **self.DEFAULT_RENDERING_KWARGS,
            'depth_resolution': samples_per_ray // 2,
            'depth_resolution_importance': samples_per_ray // 2,
        }

        # modules
        self.plane_axes = generate_planes()
        self.decoder = OSGDecoder(n_features=triplane_dim)

    def _sample_planes(self, planes, sample_coordinates, allow_chunk: bool = True):
        """
        Resample `planes` at `sample_coordinates`, supporting three modes:

        1. CPU-offloaded planes (a `CpuOffloadedTriplane`): the full tensors stays
           on CPU and is streamed per coordinate-chunk to the GPU inside the wrapper.
        2. Chunked GPU sampling: when `self.query_chunk > 0` and the query is large,
           split `sample_coordinates` along M and sample each block, then concat.
           This bounds peak GPU memory to one chunk of sampled features.
        3. Plain GPU sampling (default).

        :param allow_chunk: set False for geometry prediction, because the
            FlexiCubes weight MLP gathers tetrahedron corners from the *full* vertex
            set via `index_select` and cannot be split along M.
        """
        coords = sample_coordinates
        device = coords.device

        if isinstance(planes, CpuOffloadedTriplane):
            return planes.query_planes(coords, device)

        if allow_chunk and self.query_chunk > 0 and coords.shape[1] > self.query_chunk:
            plane_axes = self.plane_axes.to(dtype=coords.dtype, device=device)
            chunks = []
            for start in range(0, coords.shape[1], self.query_chunk):
                end = min(start + self.query_chunk, coords.shape[1])
                chunk = sample_from_planes(
                    plane_axes, planes, coords[:, start:end],
                    padding_mode='zeros', box_warp=self.rendering_kwargs['box_warp'])
                chunks.append(chunk)
            return torch.cat(chunks, dim=2)

        plane_axes = self.plane_axes.to(dtype=coords.dtype, device=device)
        return sample_from_planes(
            plane_axes, planes, coords, padding_mode='zeros', box_warp=self.rendering_kwargs['box_warp'])

    def get_geometry_prediction(self, planes, sample_coordinates, flexicubes_indices):
        # 几何预测需全顶点集合（FlexiCubes weight 通过 index_select 取四面体顶点），
        # 故此处不允许沿 M 分块（allow_chunk=False）。
        sampled_features = self._sample_planes(planes, sample_coordinates, allow_chunk=False)

        sdf, deformation, weight = self.decoder.get_geometry_prediction(sampled_features, flexicubes_indices)
        return sdf, deformation, weight
    
    def get_texture_prediction(self, planes, sample_coordinates):
        # 纹理查询坐标量通常很大（全图像素 * 视角数），按块采样以限制峰值显存。
        sampled_features = self._sample_planes(planes, sample_coordinates, allow_chunk=True)

        rgb = self.decoder.get_texture_prediction(sampled_features)
        return rgb
