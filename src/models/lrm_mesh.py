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

import numpy as np
import torch
import torch.nn as nn
import nvdiffrast.torch as dr
from einops import rearrange, repeat

from .encoder.dino_wrapper import DinoWrapper
from .decoder.transformer import TriplaneTransformer
from .renderer.synthesizer_mesh import TriplaneSynthesizer
from .geometry.camera.perspective_camera import PerspectiveCamera
from .geometry.render.neural_render import NeuralRender
from .geometry.rep_3d.flexicubes_geometry import FlexiCubesGeometry
from ..utils.mesh_util import xatlas_uvmap, simplify_mesh


class InstantMesh(nn.Module):
    """
    Full model of the large reconstruction model.
    """
    def __init__(
        self, 
        encoder_freeze: bool = False, 
        encoder_model_name: str = 'facebook/dino-vitb16', 
        encoder_feat_dim: int = 768,
        transformer_dim: int = 1024, 
        transformer_layers: int = 16, 
        transformer_heads: int = 16,
        triplane_low_res: int = 32, 
        triplane_high_res: int = 64, 
        triplane_dim: int = 80,
        rendering_samples_per_ray: int = 128,
        grid_res: int = 128, 
        grid_scale: float = 2.0,
        quantize_triplane: bool = False,
        triplane_quant_bits: int = 8,
        compress_triplane_dim: int = 0,
        triplane_query_chunk: int = 0,
        triplane_offload: bool = False,
        grid_extract_chunk: int = 0,
    ):
        super().__init__()
        
        # attributes
        self.grid_res = grid_res
        self.grid_scale = grid_scale
        self.deformation_multiplier = 4.0
        # 通道压缩（Step 4）：>0 时把 triplane_dim 降到该值（需 Step 6 蒸馏恢复精度）。
        self.compress_triplane_dim = int(compress_triplane_dim)
        self.triplane_query_chunk = int(triplane_query_chunk)
        self.triplane_offload = bool(triplane_offload)
        # 分块栅格提取块大小（Stage 2 显存上限开关；0 = 关闭，走原整批路径）
        self.grid_extract_chunk = int(grid_extract_chunk)
        # 下游 decoder 实际消费的通道数（压缩后变少）。
        decoder_triplane_dim = self.compress_triplane_dim if self.compress_triplane_dim > 0 else triplane_dim

        # modules
        self.encoder = DinoWrapper(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
        )

        self.transformer = TriplaneTransformer(
            inner_dim=transformer_dim, 
            num_layers=transformer_layers, 
            num_heads=transformer_heads,
            image_feat_dim=encoder_feat_dim,
            triplane_low_res=triplane_low_res, 
            triplane_high_res=triplane_high_res, 
            triplane_dim=triplane_dim,
            quantize_triplane=quantize_triplane,
            triplane_quant_bits=triplane_quant_bits,
            compress_triplane_dim=self.compress_triplane_dim,
        )
        
        self.synthesizer = TriplaneSynthesizer(
            triplane_dim=decoder_triplane_dim, 
            samples_per_ray=rendering_samples_per_ray,
            query_chunk=self.triplane_query_chunk,
        )

    def init_flexicubes_geometry(self, device, fovy=50.0, grid_res=None):
        """
        初始化 FlexiCubes 几何场。

        :param device: 计算设备
        :param fovy: 相机视场角（度）
        :param grid_res: 动态网格分辨率。若提供则覆盖 self.grid_res，
                         支持运行时切换 LOD（如从 128 降到 64）。
        """
        if grid_res is not None:
            self.grid_res = int(grid_res)
        self._fc_fovy = fovy
        self._fc_device = device
        # 缓存相机与渲染器，避免在切换 LOD 时反复重建 nvdiffrast 上下文。
        if not hasattr(self, '_fc_camera') or getattr(self, '_fc_fovy_cached', None) != fovy:
            self._fc_camera = PerspectiveCamera(fovy=fovy, device=device)
            self._fc_renderer = NeuralRender(device, camera_model=self._fc_camera)
            self._fc_fovy_cached = fovy
        self.geometry = FlexiCubesGeometry(
            grid_res=self.grid_res, 
            scale=self.grid_scale, 
            renderer=self._fc_renderer, 
            render_type='neural_render',
            device=device,
        )

    def forward_planes(self, images, cameras):
        # images: [B, V, C_img, H_img, W_img]
        # cameras: [B, V, 16]
        B = images.shape[0]

        # encode images
        image_feats = self.encoder(images, cameras)
        image_feats = rearrange(image_feats, '(b v) l d -> b (v l) d', b=B)
        
        # decode triplanes
        planes = self.transformer(image_feats)

        # 可选：动态分块加载（Step 4）。把 planes 留在 CPU，查询时再按块搬到 GPU，
        # 避免全网格顶点与整张 planes 同时驻留显存。仅影响采样路径，不改变输出。
        if self.triplane_offload:
            from .renderer.cpu_offload import CpuOffloadedTriplane
            planes = CpuOffloadedTriplane(
                planes, query_chunk=self.triplane_query_chunk or (2 ** 20),
                box_warp=self.synthesizer.rendering_kwargs['box_warp'])

        return planes
    
    def get_sdf_deformation_prediction(self, planes, grid_chunk: int = 0):
        '''
        Predict SDF and deformation for tetrahedron vertices.

        :param planes: triplane feature map for the geometry
        :param grid_chunk: when > 0, the SDF / deformation / weight MLPs are
            evaluated in vertex/tetrahedron blocks of at most `grid_chunk`
            elements, keeping only one block's activations on the GPU at a time
            (the "Triplane CPU block offload" for Stage 2). This bounds the peak
            memory of the (G+1)^3 grid extraction so a high-LOD (e.g. LOD96) mesh
            can be reconstructed without lowering the grid resolution. The split
            is mathematically identical to the full-batch path (every MLP is
            pointwise in vertices or per-tetrahedron), so geometry quality is
            preserved. Only active under inference (no_grad); training keeps the
            checkpoint (full-batch) path.
        '''
        init_position = self.geometry.verts.unsqueeze(0).expand(planes.shape[0], -1, -1)
        indices = self.geometry.indices  # [n_tets, k], k = corners per cell

        if grid_chunk and grid_chunk > 0 and not torch.is_grad_enabled():
            # ---- chunked offload path (inference, peak-memory bounded) ----
            # 1) sample planes at all grid vertices (pointwise; ~0.4GB @ G=96, held once)
            sampled = self.synthesizer._sample_planes(planes, init_position, allow_chunk=False)
            # sampled: [N, n_planes=3, M, C] -> feats: [N, M, 3*C] (same reshape the
            # original decoder.get_geometry_prediction does before the MLPs).
            N, n_planes, M, C = sampled.shape
            feats = sampled.permute(0, 2, 1, 3).reshape(N, M, n_planes * C)
            del sampled

            # 2) SDF + deformation: pointwise over vertices -> chunk along M
            sdf_list, def_list = [], []
            for s in range(0, M, grid_chunk):
                e = min(s + grid_chunk, M)
                blk = feats[:, s:e]  # [N, b, 3*C]
                sdf_list.append(self.synthesizer.decoder.net_sdf(blk))
                def_list.append(self.synthesizer.decoder.net_deformation(blk))
                del blk
            sdf = torch.cat(sdf_list, dim=1)
            deformation = torch.cat(def_list, dim=1)
            del sdf_list, def_list

            # 3) weight: per-cell corner gather -> chunk along n_tets.
            #    This gathers the dominant [N, n_tets, k*3C] tensor; chunking
            #    bounds its peak to a single block instead of the whole grid.
            weight_list = []
            n_tets = indices.shape[0]
            k = indices.shape[1]
            for s in range(0, n_tets, grid_chunk):
                e = min(s + grid_chunk, n_tets)
                idx_blk = indices[s:e]  # [b, k]
                gf = torch.index_select(feats, 1, idx_blk.reshape(-1))
                gf = gf.reshape(feats.shape[0], idx_blk.shape[0],
                                idx_blk.shape[1] * feats.shape[-1])
                weight_list.append(self.synthesizer.decoder.net_weight(gf) * 0.1)
                del gf, idx_blk
            weight = torch.cat(weight_list, dim=1)
            del weight_list, feats
        else:
            # ---- original full-batch path (training / grad-enabled) ----
            sdf, deformation, weight = torch.utils.checkpoint.checkpoint(
                self.synthesizer.get_geometry_prediction,
                planes,
                init_position,
                indices,
                use_reentrant=False,
            )

        # Step 2: Normalize the deformation to avoid the flipped triangles.
        deformation = 1.0 / (self.grid_res * self.deformation_multiplier) * torch.tanh(deformation)
        sdf_reg_loss = torch.zeros(sdf.shape[0], device=sdf.device, dtype=torch.float32)

        ####
        # Step 3: Fix some sdf if we observe empty shape (full positive or full negative)
        sdf_bxnxnxn = sdf.reshape((sdf.shape[0], self.grid_res + 1, self.grid_res + 1, self.grid_res + 1))
        sdf_less_boundary = sdf_bxnxnxn[:, 1:-1, 1:-1, 1:-1].reshape(sdf.shape[0], -1)
        pos_shape = torch.sum((sdf_less_boundary > 0).int(), dim=-1)
        neg_shape = torch.sum((sdf_less_boundary < 0).int(), dim=-1)
        zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
        if torch.sum(zero_surface).item() > 0:
            update_sdf = torch.zeros_like(sdf[0:1])
            max_sdf = sdf.max()
            min_sdf = sdf.min()
            update_sdf[:, self.geometry.center_indices] += (1.0 - min_sdf)  # greater than zero
            update_sdf[:, self.geometry.boundary_indices] += (-1 - max_sdf)  # smaller than zero
            new_sdf = torch.zeros_like(sdf)
            for i_batch in range(zero_surface.shape[0]):
                if zero_surface[i_batch]:
                    new_sdf[i_batch:i_batch + 1] += update_sdf
            update_mask = (new_sdf == 0).float()
            # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
            sdf_reg_loss = torch.abs(sdf).mean(dim=-1).mean(dim=-1)
            sdf_reg_loss = sdf_reg_loss * zero_surface.float()
            sdf = sdf * update_mask + new_sdf * (1 - update_mask)

        # Step 4: Here we remove the gradient for the bad sdf (full positive or full negative)
        final_sdf = []
        final_def = []
        for i_batch in range(zero_surface.shape[0]):
            if zero_surface[i_batch]:
                final_sdf.append(sdf[i_batch: i_batch + 1].detach())
                final_def.append(deformation[i_batch: i_batch + 1].detach())
            else:
                final_sdf.append(sdf[i_batch: i_batch + 1])
                final_def.append(deformation[i_batch: i_batch + 1])
        sdf = torch.cat(final_sdf, dim=0)
        deformation = torch.cat(final_def, dim=0)
        return sdf, deformation, sdf_reg_loss, weight
    
    def get_geometry_prediction(self, planes=None):
        '''
        Function to generate mesh with give triplanes
        :param planes: triplane features
        '''
        # Step 1: first get the sdf and deformation value for each vertices in the tetrahedon grid.
        sdf, deformation, sdf_reg_loss, weight = self.get_sdf_deformation_prediction(
            planes, grid_chunk=self.grid_extract_chunk)
        v_deformed = self.geometry.verts.unsqueeze(dim=0).expand(sdf.shape[0], -1, -1) + deformation
        tets = self.geometry.indices
        n_batch = planes.shape[0]
        v_list = []
        f_list = []
        flexicubes_surface_reg_list = []
        
        # Step 2: Using marching tet to obtain the mesh
        for i_batch in range(n_batch):
            verts, faces, flexicubes_surface_reg = self.geometry.get_mesh(
                v_deformed[i_batch], 
                sdf[i_batch].squeeze(dim=-1),
                with_uv=False, 
                indices=tets, 
                weight_n=weight[i_batch].squeeze(dim=-1),
                is_training=self.training,
            )
            flexicubes_surface_reg_list.append(flexicubes_surface_reg)
            v_list.append(verts)
            f_list.append(faces)
        
        flexicubes_surface_reg = torch.cat(flexicubes_surface_reg_list).mean()
        flexicubes_weight_reg = (weight ** 2).mean()
        
        return v_list, f_list, sdf, deformation, v_deformed, (sdf_reg_loss, flexicubes_surface_reg, flexicubes_weight_reg)
    
    def get_texture_prediction(self, planes, tex_pos, hard_mask=None):
        '''
        Predict Texture given triplanes
        :param planes: the triplane feature map
        :param tex_pos: Position we want to query the texture field
        :param hard_mask: 2D silhoueete of the rendered image
        '''
        tex_pos = torch.cat(tex_pos, dim=0)
        if not hard_mask is None:
            tex_pos = tex_pos * hard_mask.float()
        batch_size = tex_pos.shape[0]
        tex_pos = tex_pos.reshape(batch_size, -1, 3)
        ###################
        # We use mask to get the texture location (to save the memory)
        if hard_mask is not None:
            n_point_list = torch.sum(hard_mask.long().reshape(hard_mask.shape[0], -1), dim=-1)
            sample_tex_pose_list = []
            max_point = n_point_list.max()
            expanded_hard_mask = hard_mask.reshape(batch_size, -1, 1).expand(-1, -1, 3) > 0.5
            for i in range(tex_pos.shape[0]):
                tex_pos_one_shape = tex_pos[i][expanded_hard_mask[i]].reshape(1, -1, 3)
                if tex_pos_one_shape.shape[1] < max_point:
                    tex_pos_one_shape = torch.cat(
                        [tex_pos_one_shape, torch.zeros(
                            1, max_point - tex_pos_one_shape.shape[1], 3,
                            device=tex_pos_one_shape.device, dtype=torch.float32)], dim=1)
                sample_tex_pose_list.append(tex_pos_one_shape)
            tex_pos = torch.cat(sample_tex_pose_list, dim=0)

        tex_feat = torch.utils.checkpoint.checkpoint(
            self.synthesizer.get_texture_prediction,
            planes, 
            tex_pos,
            use_reentrant=False,
        )

        if hard_mask is not None:
            final_tex_feat = torch.zeros(
                planes.shape[0], hard_mask.shape[1] * hard_mask.shape[2], tex_feat.shape[-1], device=tex_feat.device)
            expanded_hard_mask = hard_mask.reshape(hard_mask.shape[0], -1, 1).expand(-1, -1, final_tex_feat.shape[-1]) > 0.5
            for i in range(planes.shape[0]):
                final_tex_feat[i][expanded_hard_mask[i]] = tex_feat[i][:n_point_list[i]].reshape(-1)
            tex_feat = final_tex_feat

        return tex_feat.reshape(planes.shape[0], hard_mask.shape[1], hard_mask.shape[2], tex_feat.shape[-1])
    
    def render_mesh(self, mesh_v, mesh_f, cam_mv, render_size=256):
        '''
        Function to render a generated mesh with nvdiffrast
        :param mesh_v: List of vertices for the mesh
        :param mesh_f: List of faces for the mesh
        :param cam_mv:  4x4 rotation matrix
        :return:
        '''
        return_value_list = []
        for i_mesh in range(len(mesh_v)):
            return_value = self.geometry.render_mesh(
                mesh_v[i_mesh],
                mesh_f[i_mesh].int(),
                cam_mv[i_mesh],
                resolution=render_size,
                hierarchical_mask=False
            )
            return_value_list.append(return_value)

        return_keys = return_value_list[0].keys()
        return_value = dict()
        for k in return_keys:
            value = [v[k] for v in return_value_list]
            return_value[k] = value

        mask = torch.cat(return_value['mask'], dim=0)
        hard_mask = torch.cat(return_value['hard_mask'], dim=0)
        tex_pos = return_value['tex_pos']
        depth = torch.cat(return_value['depth'], dim=0)
        normal = torch.cat(return_value['normal'], dim=0)
        return mask, hard_mask, tex_pos, depth, normal
    
    def forward_geometry(self, planes, render_cameras, render_size=256):
        '''
        Main function of our Generator. It first generate 3D mesh, then render it into 2D image
        with given `render_cameras`.
        :param planes: triplane features
        :param render_cameras: cameras to render generated 3D shape
        '''
        B, NV = render_cameras.shape[:2]

        # Generate 3D mesh first
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(planes)

        # Render the mesh into 2D image (get 3d position of each image plane)
        cam_mv = render_cameras
        run_n_view = cam_mv.shape[1]
        antilias_mask, hard_mask, tex_pos, depth, normal = self.render_mesh(mesh_v, mesh_f, cam_mv, render_size=render_size)

        tex_hard_mask = hard_mask
        tex_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in tex_pos]
        tex_hard_mask = torch.cat(
            [torch.cat(
                [tex_hard_mask[i * run_n_view + i_view: i * run_n_view + i_view + 1]
                 for i_view in range(run_n_view)], dim=2)
                for i in range(planes.shape[0])], dim=0)

        # Querying the texture field to predict the texture feature for each pixel on the image
        tex_feat = self.get_texture_prediction(planes, tex_pos, tex_hard_mask)
        background_feature = torch.ones_like(tex_feat)      # white background

        # Merge them together
        img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)

        # We should split it back to the original image shape
        img_feat = torch.cat(
            [torch.cat(
                [img_feat[i:i + 1, :, render_size * i_view: render_size * (i_view + 1)]
                 for i_view in range(run_n_view)], dim=0) for i in range(len(tex_pos))], dim=0)

        img = img_feat.clamp(0, 1).permute(0, 3, 1, 2).unflatten(0, (B, NV))
        antilias_mask = antilias_mask.permute(0, 3, 1, 2).unflatten(0, (B, NV))
        depth = -depth.permute(0, 3, 1, 2).unflatten(0, (B, NV))        # transform negative depth to positive
        normal = normal.permute(0, 3, 1, 2).unflatten(0, (B, NV))

        out = {
            'img': img,
            'mask': antilias_mask,
            'depth': depth,
            'normal': normal,
            'sdf': sdf,
            'mesh_v': mesh_v,
            'mesh_f': mesh_f,
            'sdf_reg_loss': sdf_reg_loss,
        }
        return out

    def forward_geometry_cached(self, planes, mesh_v, mesh_f, render_cameras, render_size=256):
        '''
        Render pre-generated mesh from cached mesh vertices and faces.
        This avoids recomputing the full SDF grid for every video frame.
        '''
        B, NV = render_cameras.shape[:2]

        cam_mv = render_cameras
        run_n_view = cam_mv.shape[1]
        antilias_mask, hard_mask, tex_pos, depth, normal = self.render_mesh(
            mesh_v, mesh_f, cam_mv, render_size=render_size
        )

        tex_hard_mask = hard_mask
        tex_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in tex_pos]
        tex_hard_mask = torch.cat(
            [torch.cat(
                [tex_hard_mask[i * run_n_view + i_view: i * run_n_view + i_view + 1]
                 for i_view in range(run_n_view)], dim=2)
                for i in range(planes.shape[0])], dim=0)

        tex_feat = self.get_texture_prediction(planes, tex_pos, tex_hard_mask)
        background_feature = torch.ones_like(tex_feat)

        img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)

        img_feat = torch.cat(
            [torch.cat(
                [img_feat[i:i + 1, :, render_size * i_view: render_size * (i_view + 1)]
                 for i_view in range(run_n_view)], dim=0) for i in range(len(tex_pos))], dim=0)

        img = img_feat.clamp(0, 1).permute(0, 3, 1, 2).unflatten(0, (B, NV))
        antilias_mask = antilias_mask.permute(0, 3, 1, 2).unflatten(0, (B, NV))
        depth = -depth.permute(0, 3, 1, 2).unflatten(0, (B, NV))
        normal = normal.permute(0, 3, 1, 2).unflatten(0, (B, NV))

        out = {
            'img': img,
            'mask': antilias_mask,
            'depth': depth,
            'normal': normal,
        }
        return out

    def forward(self, images, cameras, render_cameras, render_size: int):
        # images: [B, V, C_img, H_img, W_img]
        # cameras: [B, V, 16]
        # render_cameras: [B, M, D_cam_render]
        # render_size: int
        B, M = render_cameras.shape[:2]

        planes = self.forward_planes(images, cameras)
        out = self.forward_geometry(planes, render_cameras, render_size=render_size)

        return {
            'planes': planes,
            **out
        }

    def extract_mesh(
        self, 
        planes: torch.Tensor, 
        use_texture_map: bool = False,
        texture_resolution: int = 1024,
        grid_res: int = None,
        simplify_ratio: float = 0.0,
        **kwargs,
    ):
        '''
        Extract a 3D mesh from FlexiCubes. Only support batch_size 1.
        :param planes: triplane features
        :param use_texture_map: use texture map or vertex color
        :param texture_resolution: the resolution of texure map
        :param grid_res: 动态 LOD 网格分辨率。若提供且与当前 self.grid_res 不同，
                         则先重建几何场再提取（支持运行时切换分辨率）。
        :param simplify_ratio: 提取后二次边折叠简化的目标面片比例（0 表示不简化）。
        '''
        assert planes.shape[0] == 1
        # 计算设备：纹理解码器（OSGDecoder）与 FlexiCubes 几何场都在 GPU 上运行。
        # 注意：启用 --cpu_offload_triplane 时 planes 是 CpuOffloadedTriplane，其 .device
        # 逻辑上返回 cpu（仅常驻副本所在），不可用于放置坐标张量，否则简化后顶点落到
        # CPU，与 CUDA 上的 net_rgb 报设备不一致（Full-Light 报错）。以解码器真实设备为准。
        device = next(self.synthesizer.decoder.parameters()).device

        # 动态 LOD：若请求的 grid_res 与当前几何场不同，则在提取前重建几何场。
        if grid_res is not None and int(grid_res) != self.grid_res:
            print(f'  Switching FlexiCubes grid_res {self.grid_res} -> {int(grid_res)} for LOD extraction ...')
            self.init_flexicubes_geometry(
                device, fovy=getattr(self, '_fc_fovy', 30.0), grid_res=int(grid_res))

        # predict geometry first
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(planes)
        vertices, faces = mesh_v[0], mesh_f[0]

        if not use_texture_map:
            # query vertex colors
            vertices_tensor = vertices.unsqueeze(0)
            vertices_colors = self.synthesizer.get_texture_prediction(
                planes, vertices_tensor).clamp(0, 1).squeeze(0).cpu().numpy()
            vertices_colors = (vertices_colors * 255).astype(np.uint8)

            # 输出面片数控制：在提取后做二次边折叠简化
            if simplify_ratio and simplify_ratio < 1.0:
                vertices_np, faces_np = simplify_mesh(
                    vertices.cpu().numpy(), faces.cpu().numpy(), simplify_ratio)
                # 简化后顶点集合改变，必须用简化后的顶点重新查询纹理场，
                # 否则返回的仍是原始顶点色（数量不匹配 -> 保存时丢失颜色）。
                v_simp_t = torch.from_numpy(vertices_np).float().to(device).unsqueeze(0)
                colors_simp = self.synthesizer.get_texture_prediction(
                    planes, v_simp_t).clamp(0, 1).squeeze(0).cpu().numpy()
                colors_simp = (colors_simp * 255).astype(np.uint8)
                return vertices_np, faces_np, colors_simp

            return vertices.cpu().numpy(), faces.cpu().numpy(), vertices_colors

        # use x-atlas to get uv mapping for the mesh
        ctx = dr.RasterizeCudaContext(device=device)
        uvs, mesh_tex_idx, gb_pos, tex_hard_mask = xatlas_uvmap(
            self.geometry.renderer.ctx, vertices, faces, resolution=texture_resolution)
        tex_hard_mask = tex_hard_mask.float()

        # query the texture field to get the RGB color for texture map
        tex_feat = self.get_texture_prediction(
            planes, [gb_pos], tex_hard_mask)
        background_feature = torch.zeros_like(tex_feat)
        img_feat = torch.lerp(background_feature, tex_feat, tex_hard_mask)
        texture_map = img_feat.permute(0, 3, 1, 2).squeeze(0)

        # 注：带纹理贴图模式下，UV 映射已绑定原始拓扑，简化会改变顶点索引，
        # 故简化仅在顶点色（use_texture_map=False）路径生效。
        if simplify_ratio and simplify_ratio < 1.0:
            print('  [warn] simplify_ratio ignored for texture-map mode (UV binding would break). '
                  'Use vertex-color mode to apply mesh simplification.')

        return vertices, faces, uvs, mesh_tex_idx, texture_map