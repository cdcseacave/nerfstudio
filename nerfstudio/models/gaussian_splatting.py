# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.transforms.functional as TF

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.math import quaternion_from_vectors
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig

from gsplat._torch_impl import quat_to_rotmat
from gsplat.rasterize import RasterizeGaussians
from gsplat.project_gaussians import ProjectGaussians
from gsplat.sh import SphericalHarmonics, num_sh_bases
from pytorch_msssim import SSIM

# need following import for background color override
from nerfstudio.model_components import renderers


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def projection_matrix(znear, zfar, fovx, fovy, device: Union[str, torch.device] = "cpu"):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )


@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: GaussianSplattingModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled"""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians"""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 13000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="off")
    """camera optimizer config"""
    max_gauss_ratio: float = 50.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    scale_lambda: float = 0.01 # Weight of scale loss
    cull_visibility_thresh: float = 0.01 # threshold of visibility for culling gaussians
    visibility_lambda: float = 0.001 # Weight of visibility loss
    init_pts_sphere_num: int = 20000 # Initialize gaussians at a sphere with this many randomly placed points. Set to 0 to disable
    init_pts_sphere_rad_pct: float = 0.98 # Initialize gaussians at a sphere: set radius based on looking at the 99th percentile of initial points' distance from origin
    init_pts_sphere_rad_mult: float = 1.1 # Initialize gaussians at a sphere: set radius based on init_pts_sphere_rad_pct * this value
    init_pts_sphere_rad_min: float = 5.0 # Initialize gaussians at a sphere with this as the minimum radius
    init_pts_sphere_rad_max: float = 20.0 # Initialize gaussians at a sphere with this as the maximum radius


class GaussianSplattingModel(Model):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: GaussianSplattingModelConfig

    def __init__(self, *args, **kwargs):
        if "seed_points" in kwargs:
            self.seed_pts = kwargs["seed_points"]
        else:
            self.seed_pts = None
        if "base_dir" in kwargs:
            self.base_dir = kwargs["base_dir"]
            # open json file to save cull stats
            self.cull_stats_file = open(self.base_dir / "cull_stats.cvs", "w")
            self.cull_stats_file.write(f"Step,Points,Culled,Below alpha,Below visibility\n")
        else:
            self.base_dir = None
        super().__init__(*args, **kwargs)

    def __del__(self):
        if self.base_dir is not None:
            self.cull_stats_file.close()

    def init_scale_rotation(
        self, points: torch.Tensor, normals: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_points = points.shape[0]
        distances, _ = self.k_nearest_sklearn(points, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        if normals is None:
            # use random initialization of the covariance
            print(f"Initializing {num_points} splats")
            scales = torch.log(avg_dist.repeat(1, 3))
            quats = random_quat_tensor(num_points)
        else:
            # use the normals to initialize the covariance
            print(f"Initializing {num_points} splats with normals")
            scales = torch.log(torch.cat(
                [
                    avg_dist * (1.0 / self.config.max_gauss_ratio),
                    avg_dist,
                    avg_dist,
                ],
                dim=-1,
            ))
            quats = torch.from_numpy(
                np.array([quaternion_from_vectors(n).q.astype(np.float32) for n in normals.cpu().numpy()])
            )
        return scales, quats

    def add_init_sphere_pts(self):
        # Estimate phere radius
        dists = torch.linalg.vector_norm(self.means, dim=1).detach().numpy()
        dist_raw = np.percentile(dists, self.config.init_pts_sphere_rad_pct*100, overwrite_input=False) * self.config.init_pts_sphere_rad_mult
        self.outer_sphere_rad = max(min(dist_raw, self.config.init_pts_sphere_rad_max), self.config.init_pts_sphere_rad_min)
        # Generate sphere points
        sphere_pts = (torch.rand((self.config.init_pts_sphere_num, 3)) - 0.5) * 2
        dists = torch.linalg.vector_norm(sphere_pts, dim=1, keepdim=True)
        rescale = self.outer_sphere_rad / (dists + 1e-8)
        sphere_pts = sphere_pts * torch.cat([rescale, rescale, rescale], dim=1)
        # Initialize points, scale and rotation; use negated positions as normals
        sphere_scales, sphere_quats = self.init_scale_rotation(sphere_pts, -sphere_pts / self.outer_sphere_rad)
        # Initialize colors, opacities
        dim_sh = num_sh_bases(self.config.sh_degree)
        sphere_colors     = torch.rand(self.config.init_pts_sphere_num, 1, 3).to(self.colors_all.device)
        sphere_shs_rest   = torch.zeros((self.config.init_pts_sphere_num, dim_sh - 1, 3)).to(self.colors_all.device)
        sphere_colors_all = torch.nn.Parameter(torch.cat([sphere_colors, sphere_shs_rest], dim=1))
        sphere_opacities  = torch.logit(0.1 * torch.ones(self.config.init_pts_sphere_num, 1)).to(self.opacities.device)
        sphere_visibility = torch.logit(torch.ones(self.config.init_pts_sphere_num, 1)).to(self.opacities.device)
        # Add to model
        self.means        = torch.nn.Parameter(torch.cat([self.means.detach(),      sphere_pts],        dim=0))
        self.scales       = torch.nn.Parameter(torch.cat([self.scales.detach(),     sphere_scales],     dim=0))
        self.quats        = torch.nn.Parameter(torch.cat([self.quats.detach(),      sphere_quats],      dim=0))
        self.colors_all   = torch.nn.Parameter(torch.cat([self.colors_all.detach(), sphere_colors_all], dim=0))
        self.opacities    = torch.nn.Parameter(torch.cat([self.opacities.detach(),  sphere_opacities],  dim=0))
        self.visibility   = torch.nn.Parameter(torch.cat([self.visibility.detach(), sphere_visibility], dim=0))
        print(f"Initialized {self.config.init_pts_sphere_num} splats on a sphere with radius {self.outer_sphere_rad}")

    def populate_modules(self):
        self.xys_grad_norm = None
        self.max_2Dsize = None

        dim_sh = num_sh_bases(self.config.sh_degree)
        # self.seed_pts contains (Location, Color, Normal)
        if self.seed_pts is not None and not self.config.random_init:
            points = self.seed_pts[0]
            scales, quats = self.init_scale_rotation(points, self.seed_pts[2] if len(self.seed_pts) > 2 else None)
            fused_color = RGB2SH(self.seed_pts[1] / 255)
            shs = torch.zeros((fused_color.shape[0], dim_sh, 3), dtype=torch.float)
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
            self.colors_all = torch.nn.Parameter(shs)
        else:
            num_points = 500000
            points = (torch.rand((num_points, 3), dtype=torch.float) - 0.5) * 10
            scales, quats = self.init_scale_rotation(points)
            colors = torch.rand((num_points, 1, 3), dtype=torch.float)
            shs_rest = torch.zeros((num_points, dim_sh - 1, 3), dtype=torch.float)
            shs = torch.cat([colors, shs_rest], dim=1)

        self.means = torch.nn.Parameter(points)
        self.scales = torch.nn.Parameter(scales)
        self.quats = torch.nn.Parameter(quats)
        self.colors_all = torch.nn.Parameter(shs)
        self.opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1)))
        self.visibility = torch.nn.Parameter(torch.logit(torch.ones(self.num_points, 1)))

        # Init outer sphere dimensions & points
        if self.config.init_pts_sphere_num > 0 and self.training:
            self.add_init_sphere_pts()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        self.back_color = torch.zeros(3)

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        print(f"Initialized {self.num_points} splats with SH degree {self.config.sh_degree}")

    @property
    def colors(self):
        return self.colors_all[:, 0, :]

    @property
    def shs_rest(self):
        return self.colors_all[:, 1:, :]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        newp = dict["means"].shape[0]
        self.means = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.scales = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.quats = torch.nn.Parameter(torch.zeros(newp, 4, device=self.device))
        self.opacities = torch.nn.Parameter(torch.zeros(newp, 1, device=self.device))
        self.colors_all = torch.nn.Parameter(torch.zeros(newp, num_sh_bases(self.config.sh_degree), 3, device=self.device))
        self.visibility = torch.nn.Parameter(torch.zeros(newp, 1, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
        Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
        param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
        param_state["exp_avg"] = torch.cat(
            [param_state["exp_avg"], torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims)],
            dim=0,
        )
        param_state["exp_avg_sq"] = torch.cat(
            [
                param_state["exp_avg_sq"],
                torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
            ],
            dim=0,
        )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def after_train(self, step: int):
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.grad.detach().norm(dim=-1)  # TODO fill in
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None and len(visible_mask) == len(self.vis_counts)
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask], newradii / float(max(self.last_size[0], self.last_size[1]))
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, back_color: torch.Tensor):
        assert back_color.shape == (3,)
        self.back_color = back_color

    def refinement_after(self, optimizers: Optimizers, step):
        if self.step >= self.config.warmup_length:
            with torch.no_grad():
                # only split/cull if we've seen every image since opacity reset
                reset_interval = self.config.reset_alpha_every * self.config.refine_every
                splits_mask = None
                if (
                    self.step < self.config.stop_split_at
                    and self.step % reset_interval > self.num_train_data + self.config.refine_every
                ):
                    # then we densify
                    assert (
                        self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                    )
                    avg_grad_norm = (
                        (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                    )
                    high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                    splits = (self.weighted_scales().exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                    if self.step < self.config.stop_screen_size_at:
                        splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                    splits &= high_grads
                    nsamps = self.config.n_split_samples
                    (
                        split_means,
                        split_colors,
                        split_opacities,
                        split_scales,
                        split_quats,
                        split_visibility,
                    ) = self.split_gaussians(splits, nsamps)

                    dups = (self.weighted_scales().exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                    dups &= high_grads
                    (
                        dup_means,
                        dup_colors,
                        dup_opacities,
                        dup_scales,
                        dup_quats,
                        dup_visibility
                    ) = self.dup_gaussians(dups)
                    self.means = Parameter(torch.cat([self.means.detach(), split_means, dup_means], dim=0))
                    self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))
                    self.opacities = Parameter(torch.cat([self.opacities.detach(), split_opacities, dup_opacities], dim=0))
                    self.scales = Parameter(torch.cat([self.scales.detach(), split_scales, dup_scales], dim=0))
                    self.quats = Parameter(torch.cat([self.quats.detach(), split_quats, dup_quats], dim=0))
                    self.visibility = Parameter(torch.cat([self.visibility.detach(), split_visibility, dup_visibility], dim=0))
                    # append zeros to the max_2Dsize tensor
                    self.max_2Dsize = torch.cat(
                        [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                        dim=0,
                    )
                    split_idcs = torch.where(splits)[0]
                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        self.dup_in_optim(optimizers.optimizers[group], split_idcs, param, n=nsamps)
                    dup_idcs = torch.where(dups)[0]

                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        self.dup_in_optim(optimizers.optimizers[group], dup_idcs, param, 1)

                    # After a guassian is split into two new gaussians, the original one should also be pruned.
                    splits_mask = torch.cat(
                        (splits, torch.zeros(nsamps * splits.sum() + dups.sum(), device=self.device, dtype=torch.bool))
                    )

                # Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
                if self.step % reset_interval > self.num_train_data + self.config.refine_every:
                    deleted_mask = self.cull_gaussians(splits_mask)
                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
                    torch.cuda.empty_cache()

                # if self.step % reset_interval == self.config.refine_every:
                #     # Reset value is set to under the cull_alpha_thresh
                #     reset_value = self.config.cull_alpha_thresh * 0.8
                #     self.opacities.data = torch.clamp(
                #         self.opacities.data, max=torch.logit(torch.tensor(reset_value, device=self.device)).item()
                #     )
                #     # reset the exp of optimizer
                #     optim = optimizers.optimizers["opacity"]
                #     param = optim.param_groups[0]["params"][0]
                #     param_state = optim.state[param]
                #     param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                #     param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.weighted_opacities()) < self.config.cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        visibility_culls = (torch.sigmoid(self.visibility) <= self.config.cull_visibility_thresh).squeeze()
        below_visibility_count = torch.sum(visibility_culls).item()
        culls = culls | visibility_culls
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (self.weighted_scales().exp().max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            culls = culls | toobigs
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
        self.means = Parameter(self.means[~culls].detach())
        self.scales = Parameter(self.scales[~culls].detach())
        self.quats = Parameter(self.quats[~culls].detach())
        self.colors_all = Parameter(self.colors_all[~culls].detach())
        self.opacities = Parameter(self.opacities[~culls].detach())
        self.visibility = Parameter(self.visibility[~culls].detach())

        print(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {below_visibility_count} below visibility thresh, {self.num_points} remaining)"
        )
        if self.base_dir is not None:
            # save cull stats and flush to disk
            self.cull_stats_file.write(f"{self.step},{self.num_points},{n_bef - self.num_points},{below_alpha_count},{below_visibility_count}\n")
            self.cull_stats_file.flush()
        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        print(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples)
        rots = quat_to_rotmat(self.quats[split_mask].repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        new_scales = self.scales[split_mask].repeat(samps, 1)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        # step 6, sample new visibility
        new_visibility = self.visibility[split_mask].repeat(samps, 1)
        return new_means, new_colors_all, new_opacities, new_scales, new_quats, new_visibility

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        dup_means = self.means[dup_mask]
        dup_colors = self.colors_all[dup_mask]
        dup_opacities = self.opacities[dup_mask]
        dup_scales = self.scales[dup_mask]
        dup_quats = self.quats[dup_mask]
        dup_visibility = self.visibility[dup_mask]
        return dup_means, dup_colors, dup_opacities, dup_scales, dup_quats, dup_visibility

    @property
    def num_points(self):
        return self.means.shape[0]

    def visibility_weights(self, visibility: Optional[torch.Tensor] = None):
        visibility_weight = torch.sigmoid(self.visibility if visibility is None else visibility)
        visibility_weight = ((visibility_weight > self.config.cull_visibility_thresh).float() - visibility_weight).detach() + visibility_weight
        return visibility_weight

    def weighted_scales(self, visibility_weight: Optional[torch.Tensor] = None, scales: Optional[torch.Tensor] = None):
        if visibility_weight is None:
            visibility_weight = self.visibility_weights()
        scales = (self.scales if scales is None else scales) * visibility_weight
        return scales

    def weighted_opacities(self, visibility_weight: Optional[torch.Tensor] = None, opacities: Optional[torch.Tensor] = None):
        if visibility_weight is None:
            visibility_weight = self.visibility_weights()
        opacities = (self.opacities if opacities is None else opacities) * visibility_weight
        return opacities

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "xyz": [self.means],
            "color": [self.colors_all],
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats],
            "visibility": [self.visibility],
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        # add camera optimizer param groups
        self.camera_optimizer.get_param_groups(gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.config.num_downscales - self.step // self.config.resolution_schedule), 0)
        else:
            return 1

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"
        if self.training:
            # currently relies on the branch vickie/camera-grads
            self.camera_optimizer.apply_to_camera(camera)
            background = torch.rand(3, device=self.device)
        else:
            # logic for setting the background of the scene
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE
            else:
                background = self.back_color.to(self.device)
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return {"rgb": background.repeat(camera.height.item(), camera.width.item(), 1)}
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = camera.width.item(), camera.height.item()
        if self.training:
            self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)
        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = (
            (W + BLOCK_X - 1) // BLOCK_X,
            (H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            colors_crop = self.colors_all[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            visibility_crop = self.visibility[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            colors_crop = self.colors_all
            scales_crop = self.scales
            quats_crop = self.quats
            visibility_crop = self.visibility
        if self.training and self.step < self.config.stop_split_at:
            # cull based on visibility
            visibility_weight = self.visibility_weights(visibility_crop)
            scales_crop = self.weighted_scales(visibility_weight, scales_crop)
            opacities_crop = self.weighted_opacities(visibility_weight, opacities_crop)
        xys, depths, radii, conics, num_tiles_hit, _ = ProjectGaussians.apply(
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            tile_bounds,
        )
        if (radii).sum() == 0:
            return {"rgb": background.repeat(H, W, 1)}

        # Important to allow xys grads to populate properly
        if self.training and xys.requires_grad:
            xys.retain_grad()
        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = torch.clamp(SphericalHarmonics.apply(n, viewdirs, colors_crop) + 0.5, 0.0, 1.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(self.get_colors.squeeze())  # (N, 3)
        rgb = RasterizeGaussians.apply(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            rgbs,
            torch.sigmoid(opacities_crop),
            H,
            W,
            background,
        )
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        if self.training:
            # Only save xys and radii if we're training
            self.xys = xys
            self.radii = radii
            depth_im = None
        else:
            depth_im = RasterizeGaussians.apply(  # type: ignore
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                depths[:, None].repeat(1, 3),
                torch.sigmoid(opacities_crop),
                H,
                W,
                torch.ones(3, device=self.device) * 10,
            )[..., 0:1]
        # rescale the camera back to original dimensions
        camera.rescale_output_resolution(camera_downscale)
        return {"rgb": rgb, "depth": depth_im}  # type: ignore

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
        metrics_dict = {}
        gt_rgb = gt_img.to(self.device)  # RGB or RGBA image
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        metrics_dict["gaussian_count"] = self.num_points
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
        Ll1 = torch.abs(gt_img - outputs["rgb"]).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], outputs["rgb"].permute(2, 0, 1)[None, ...])
        if self.step % 10 == 0 and self.step < self.config.stop_split_at:
            # Before, we made split sh and colors onto different optimizer, with shs having a low learning rate
            # This is slow, instead we apply a regularization every few steps
            sh_reg = self.colors_all[:, 1:, :].norm(dim=1).mean()
        else:
            sh_reg = torch.tensor(0.0).to(self.device)
        if self.step < self.config.stop_split_at:
            scale_exp = torch.exp(self.scales)
            # scale_reg = (
            #     torch.maximum(
            #         scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(self.config.max_gauss_ratio)
            #     )
            #     - self.config.max_gauss_ratio
            # ).mean() * 0.1
            scale_L1 = scale_exp.amin(dim=-1).mean() * self.config.scale_lambda # Penalize for high scale values
            visibility_L1 = torch.sigmoid(self.visibility).mean() * self.config.visibility_lambda # Penalize for high visibility values
        else:
            # scale_reg = torch.tensor(0.0).to(self.device)
            scale_L1 = torch.tensor(0.0).to(self.device)
            visibility_L1 = torch.tensor(0.0).to(self.device)
        return {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "sh_reg": sh_reg,
            # "scale_reg": scale_reg,
            "scale": scale_L1,
            "visibility": visibility_L1,
        }

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
            predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
            predicted_rgb = outputs["rgb"]

        gt_rgb = gt_img.to(self.device)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
