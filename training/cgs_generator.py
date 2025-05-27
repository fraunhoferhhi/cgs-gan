# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import torch

from camera_utils import focal2fov
from torch_utils import persistence
from training.gaussian3d_splatting.custom_cam import CustomCam
from training.networks_stylegan2 import MappingNetwork
from training.gaussian3d_splatting.renderer import Renderer

from training.point_generator import PointGenerator


@persistence.persistent_class
class CGSGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_channels=img_channels
        self.resolution = img_resolution
        self.rendering_kwargs = rendering_kwargs
        self.custom_options = rendering_kwargs['custom_options']

        self.num_pts = self.custom_options['num_pts']
        random_coords = torch.randn((self.num_pts, 3))
        self._xyz = torch.nn.Parameter(random_coords * torch.rsqrt(torch.mean(random_coords ** 2, dim=1, keepdim=True) + 1e-8) * 0.5 * 0.6)
        self.point_gen = PointGenerator(w_dim=w_dim, options=self.custom_options)
        self.renderer_gaussian3d = Renderer(sh_degree=0)
        self.mapping_network = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.point_gen.num_ws + 1, **mapping_kwargs)


    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        return self.mapping_network(z, torch.zeros_like(c), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, resolution=None, update_emas=False, gs_params=None, random_bg=True, render_output=True, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if resolution is None:
            resolution = self.resolution
        else:
            self.resolution = resolution

        focalx, focaly, near, far = intrinsics[:, 0,0], intrinsics[:, 1,1], 0.1, 10

        sample_coordinates = torch.tanh(self._xyz.unsqueeze(0).repeat(len(ws), 1, 1))
        sample_coordinates, sample_scale, sample_rotation, sample_color, sample_opacity, anchors = self.point_gen(sample_coordinates, ws)
        dec_out = {}
        dec_out["sample_coordinates"] = sample_coordinates
        dec_out["scale"] = sample_scale
        dec_out["rotation"] = sample_rotation
        dec_out["color"] = sample_color
        dec_out["opacity"] = sample_opacity

        gaussian_params = []
        rendered_images = []
        for batch_idx in range(len(ws)):
            gaussian_params_i = {}
            gaussian_params_i["_xyz"] = dec_out['sample_coordinates'][batch_idx]
            gaussian_params_i["_features_dc"] = dec_out["color"][batch_idx].unsqueeze(1).contiguous() # self._features_dc # 3
            gaussian_params_i["_features_rest"] = dec_out["color"][batch_idx].unsqueeze(1)[:, 0:0].contiguous() # self._features_rest # 3
            gaussian_params_i["_scaling"] = dec_out["scale"][batch_idx] # self._scaling # 3
            gaussian_params_i["_rotation"] = dec_out["rotation"][batch_idx] # self._rotation # 4
            gaussian_params_i["_opacity"] = dec_out["opacity"][batch_idx] # self._opacity # 1
            gaussian_params.append(gaussian_params_i)

            if render_output:
                fovx = focal2fov(focalx[batch_idx])
                fovy = focal2fov(focaly[batch_idx])
                cur_cam = CustomCam(resolution, resolution, fovy=fovy, fovx=fovx, extr=cam2world_matrix[batch_idx])
                if random_bg:
                    bg = torch.rand(3, device=ws.device)
                else:
                    bg = torch.ones(3, device=ws.device)
                ret_dict = self.renderer_gaussian3d.render(gaussian_params_i, cur_cam, bg=bg)
                rendered_images.append(ret_dict["image"].unsqueeze(0))

        return_dict = {'anchors': anchors[0], 'gaussian_params': gaussian_params}
        if render_output:
            return_dict["image"] = torch.cat(rendered_images, dim=0).to(ws.device)
        return return_dict

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, random_bg=True, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, random_bg=random_bg, **synthesis_kwargs)


