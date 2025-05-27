# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from torch_utils.logger import CustomLogger
from typing import Union, Iterable
import torch

from camera_utils import focal2fov
from torch.nn.utils.clip_grad import clip_grad_norm_
from training.gaussian3d_splatting.custom_cam import CustomCam
from training.gaussian3d_splatting.renderer import Renderer

PARAMETERS_DTYPE = Union[torch.Tensor, Iterable[torch.Tensor]]


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, logger: CustomLogger): # to be overridden by subclass
        raise NotImplementedError()


class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, r1_gamma=10, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, resolution=512, filter_mode='antialiased', loss_custom_options={}):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.r1_gamma           = r1_gamma
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.resolution = resolution
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)

        self.coeffs = loss_custom_options
        self.renderer_gaussian3d = Renderer(sh_degree=0)

    def run_G(self, z, c, resolution, update_emas=False, render_output=True):
        c_gen_conditioning = torch.zeros_like(c)
        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        gen_output = self.G.synthesis(ws, c, resolution=resolution, update_emas=update_emas, render_output=render_output)
        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, logger: CustomLogger):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        logger.add("Training", "blur sigma", blur_sigma)
        r1_gamma = self.r1_gamma

        real_img = {'image': real_img}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                if self.coeffs["use_multivew_reg"]:
                    # multi view regularization
                    gen_result, _gen_ws = self.run_G(gen_z, gen_c, resolution=self.resolution, render_output=False)
                    num_opt_steps = self.coeffs["num_multiview"]
                    fov = focal2fov(gen_c[0, 20])
                    loss_Gmain = 0
                    for i in range(num_opt_steps):
                        batch_renderings = []
                        batch_cams = []
                        gen_c = torch.roll(gen_c, 1, dims=0)
                        for batch_idx, current_scene in enumerate(gen_result["gaussian_params"]):
                            extrinsic = gen_c[batch_idx, :16].reshape(4, 4)
                            intrinsics = torch.tensor([
                                gen_c[0, 16], 0.0,    0.5,
                                0.0,    gen_c[0, 20], 0.5,
                                0.0,    0.0,    1.0
                            ], device="cuda")

                            render_cam = CustomCam(self.resolution, self.resolution, fovy=fov, fovx=fov, extr=extrinsic)
                            bg = torch.rand(3, device=gen_z.device)
                            ret_dict = self.renderer_gaussian3d.render(gaussian_params=current_scene, viewpoint_camera=render_cam, bg=bg)
                            batch_renderings.append(ret_dict["image"])
                            batch_cams.append(torch.concat([extrinsic.reshape(-1), intrinsics], dim=0))

                        renderings = torch.stack(batch_renderings, dim=0)
                        cams = torch.stack(batch_cams, dim=0)
                        gen_logits = self.run_D({"image": renderings}, cams, blur_sigma=blur_sigma)
                        loss_Gmain += torch.nn.functional.softplus(-gen_logits)
                    loss_Gmain /= num_opt_steps
                else:
                    # normal pass
                    gen_result, _gen_ws = self.run_G(gen_z, gen_c, resolution=self.resolution)
                    gen_logits = self.run_D(gen_result, gen_c, blur_sigma=blur_sigma)
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits)

                anchors = gen_result['anchors'][:, :self.G.num_pts]
                dist = knn_distance(anchors, k=self.coeffs['knn_num_ks']).mean()
                dist_center = anchors.mean(dim=1).square().mean()

                logger.add("Dist Loss", "KNN", dist)
                logger.add("Dist Loss", "Center", dist_center)
                logger.add("Loss", "D_loss", gen_logits)
                logger.add("Loss_Sign", "signs_fake", gen_logits.sign())
                logger.add("Loss", "G_loss", loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                ((loss_Gmain).mean().mul(gain) + dist_center.mean() * self.coeffs['center_dists'] + dist * self.coeffs['knn_dists']).backward()
                clip_grad_norm_(self.G.parameters(), max_norm=20)

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_result, _gen_ws = self.run_G(gen_z, gen_c, resolution=self.resolution, update_emas=True)
                logger.add_tensor_stats("3dgs", "_xyz", gen_result["gaussian_params"][0]["_xyz"])
                logger.add_tensor_stats("3dgs", "_features_dc", gen_result["gaussian_params"][0]["_features_dc"])
                logger.add_tensor_stats("3dgs", "_scaling", gen_result["gaussian_params"][0]["_scaling"])
                logger.add_tensor_stats("3dgs", "_rotation", gen_result["gaussian_params"][0]["_rotation"])
                logger.add_tensor_stats("3dgs", "_opacity", gen_result["gaussian_params"][0]["_opacity"])

                gen_logits = self.run_D(gen_result, gen_c, blur_sigma=blur_sigma, update_emas=True)
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
                logger.add("Scores", "scores_fake", gen_logits)
                logger.add("Loss_Sign", "signs_fake", gen_logits.sign())
                
            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen).mean().mul(gain).backward() # Do not use contrastive loss for D_gen
                clip_grad_norm_(self.D.parameters(), max_norm=5)

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image}
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)

                logger.add("Scores", "scores_real", real_logits)
                logger.add("Loss_Sign", "signs_real", real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    logger.add("Loss", "D_loss", loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                        r1_grads_image = r1_grads[0]
                    r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    logger.add("Reg", "D_reg", loss_Dr1)
                    logger.add("Reg", "r1_penalty", r1_penalty)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()
                clip_grad_norm_(self.D.parameters(), max_norm=5)
              

def knn_distance(pos, k):
    x = pos.permute(0, 2, 1)
    B, dims, N = x.shape

    xt = x.permute(0, 2, 1)
    xi = -2 * torch.bmm(xt, x)
    xs = torch.sum(xt**2, dim=2, keepdim=True)
    xst = xs.permute(0, 2, 1)
    dist = xi + xs + xst # [B, N, N]

    # get k NN id
    _, idx_o = torch.sort(dist, dim=2)
    idx = idx_o[: ,: ,1:k+1] # [B, N, k]
    idx = idx.contiguous().view(B, N*k)

    # gather
    neighbors = []
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b]) # [d, N*k] <- [d, N], 0, [N*k]
        tmp = tmp.view(dims, N, k)
        neighbors.append(tmp)

    neighbors = torch.stack(neighbors) # [B, d, N, k]

    # centralize
    central = x.unsqueeze(3) # [B, d, N, 1]
    central = central.repeat(1, 1, 1, k) # [B, d, N, k]

    return (central - neighbors).square().mean(dim=[1, 3])

