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
import random
from nerfstudio.data.scene_box import OrientedBox
from copy import deepcopy
from nerfstudio.cameras.rays import RayBundle
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from nerfstudio.cameras.cameras import Cameras
from gsplat._torch_impl import quat_to_rotmat
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
import viser.transforms as vtf


from gsplat.rasterize import RasterizeGaussians
from gsplat.project_gaussians import ProjectGaussians
from gsplat.sh import SphericalHarmonics, num_sh_bases
from pytorch_msssim import  SSIM


def random_quat_tensor(N, **kwargs):
    u = torch.rand(N, **kwargs)
    v = torch.rand(N, **kwargs)
    w = torch.rand(N, **kwargs)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def projection_matrix(znear, zfar, fovx, fovy, **kwargs):
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
        **kwargs,
    )


@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: GaussianSplattingModel)

    # High level settings
    # NOTE: to change max_iterations in the terminal arguments, the following arguments need to be all set to match where n = max_iterations_new:
    # --pipeline.model.max_iterations=n
    # --max-num-iterations=n 
    # --optimizers.xyz.scheduler.max-steps=n 
    # --optimizers.color.scheduler.max-steps=n 
    # --optimizers.shs.scheduler.max-steps=n 
    # --optimizers.scaling.scheduler.max-steps=n 
    # --optimizers.rotation.scheduler.max-steps=n
    max_iterations: int = 30000 # Sets the defaults in method_configs.py. Multiple config values will be messed up if num_iterations is set via the command line arg. Change it here instead
    max_gaussians: int = 5000000 # Max number of 3D gaussians. As this number is approached, densify_grad_thresh is increased to slow down densification. It's possible for n_gaussians to go over this number by a bit, but is unlikely
    sh_degree: int = 2 # Maximum degree of spherical harmonics to use

    # Refinement (densification & culling) settings
    warmup_length: int = 1000 # Period of steps where refinement is turned off
    stop_refine_at: float = 0.7 # [Multiply this value (0-1) by max_iterations] Stop splitting & culling at this step. Must be < max_iterations or remove_gaussians_min_cameras_in_fov won't work
    refine_every: int = 100 # Period of steps where gaussians are culled and densified
    cull_alpha_thresh: float = 0.01 # Threshold of opacity for culling gaussians
    cull_scale_thresh: float = 10.0 # Threshold of scale for culling gaussians. Make this large enough for now to make it irrelevant. Use cull_screen_size instead
    cull_screen_size_init: float = 2.0 # If a gaussian is more than this percent of screen space, cull it
    cull_screen_size_final: float = 0.20 # If a gaussian is more than this percent of screen space, cull it
    split_screen_size: float = 0.05 # If a gaussian is more than this percent of screen space, split it
    reset_alpha_every: int = 30 # Every this many refinement steps, reset the alpha
    densify_size_thresh: float = 0.01 # Below this size, gaussians are *duplicated*, otherwise split
    densify_grad_thresh_init: float = 0.0002 # Threshold of positional gradient norm for densifying gaussians. Also see densify_grad_thresh_start_increase
    densify_grad_thresh_final: float = 10.0 * densify_grad_thresh_init # See densify_grad_thresh_start_increase
    densify_grad_thresh_start_increase: float = 0.7 # [0-1] Start increasing densify_grad_thresh from init -> final once there are densify_grad_thresh_start_increase * max_num_splats of splats
    densify_until_iter_start_increase: float = 0.5 # [Multiply this value (0-1) by max_iterations] After this many iterations, make it harder to densify (value should be lower than densify_until_iter)
    densify_snap_to_outer_sphere: bool = False # If true: snap densified points to the outer_sphere
    
    # Image resolution
    resolution_schedule: int = 200 # Training starts at 1/d resolution, every n steps this is doubled
    num_downscales: int = 2 # At the beginning, resolution is 1/2^d, where d is this number

    # Initialization
    random_init: bool = False # Whether to initialize the positions uniformly randomly (not SFM points)
    init_pts_sphere_on_iter: int = -1 # -1 to initialize immediately. Higher value to initialize later in the optimization
    init_pts_sphere_num: int = 20000 # Initialize gaussians at a sphere with this many randomly placed points. Set to 0 to disable
    init_pts_sphere_rad_pct: float = 0.98 # Initialize gaussians at a sphere: set radius based on looking at the 99th percentile of initial points' distance from origin
    init_pts_sphere_rad_mult: float = 1.1 # Initialize gaussians at a sphere: set radius based on init_pts_sphere_rad_pct * this value
    init_pts_sphere_rad_min: float = 5.0 # Initialize gaussians at a sphere with this as the minimum radius
    init_pts_sphere_rad_max: float = 20.0 # Initialize gaussians at a sphere with this as the maximum radius
    init_pts_hemisphere: bool = True # Bottom half of the sphere is flat (hemisphere)
    
    # Loss function
    ssim_lambda: float = 0.2 # Weight of ssim loss
    dist2cam_loss_size: float = 30.0 # Pull points away from camera until it gets this far away. Should probably be a bit larger than init_pts_sphere_rad_max
    dist2cam_loss_lambda: float = 0.01 # Pull points away from camera: add L1 term to loss function that has a range [0-1] -> multiply by this number
    regularize_sh_lambda: float = 0.01 # sh coeffs are multiplied by x, y, z, xx, xy, xz, etc. where [x,y,z] is a unit vector (no coeff normalization needed) so all coeffs are summed together for L2 loss -> multiply by this number
    outside_outer_sphere_lambda: float = 1.0 # Penalty for gaussians going outside the outer sphere
    under_hemisphere_lambda: float = 1.0 # Penalty for gaussians going under the outer hemisphere

    # Early stopping
    early_stop_check_every: int = 500 # Check every n steps if the loss has stopped decreasing
    early_stop_loss_diff_threshold: float = 2.8e-6 # If the loss has decreased by less than this amount per step, stop training
    early_stop_max_loss: float = 0.2 # Don't stop early if the loss is greater than this
    early_stop_additional_steps: int = 2000 # If early stopping is triggered, train for this many more steps before stopping

    # Other
    sh_degree_interval: int = 500 # Every n intervals turn on another sh degree

    # Post-processing
    remove_gaussians_min_cameras_in_fov: int = 2 # At the end of training, only retain gaussians if they are in the field of view of at least this many cameras. Set to 0 to disable
    remove_gaussians_outside_sphere: float = 1.2 # At the end of training, remove gaussians if they are outside the outer_sphere_radius * this value. Set to 0 to disable


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
        super().__init__(*args, **kwargs)

    def get_init_sphere_radius(self) -> float:
        dists = torch.linalg.vector_norm(self.means, dim=1).detach().numpy()
        dist_pct = np.percentile(dists, self.config.init_pts_sphere_rad_pct*100, overwrite_input=False)
        dist_raw = dist_pct * self.config.init_pts_sphere_rad_mult
        rad = max(min(dist_raw, self.config.init_pts_sphere_rad_max), self.config.init_pts_sphere_rad_min)
        print("Initializing " + str(self.config.init_pts_sphere_num) + " 3D gaussians at radius = " + str(rad))
        return rad
    
    def get_init_bottom_plane(self) -> float:
        zs = self.means[:,2].detach().numpy()
        z_low = np.percentile(zs, (1.0 - self.config.init_pts_sphere_rad_pct)*100, overwrite_input=False)
        print("Initializing bottom of hemisphere to " + str(z_low))
        return z_low
    
    def snap_to_outer_sphere_if_outside(self, means):
        # If already inside the outer sphere, this does nothing, else: find closest point inside the outer sphere -> return that
        dists = torch.linalg.vector_norm(means, dim=1, keepdim=True)
        snap_to_sphere = (dists > self.outer_sphere_rad).squeeze()
        rescale = self.outer_sphere_rad / dists[snap_to_sphere]
        means[snap_to_sphere,:] = means[snap_to_sphere,:] * torch.cat([rescale, rescale, rescale], dim=1)
        if self.config.init_pts_hemisphere:
            means[(means[:,2] < self.outer_sphere_z_low).squeeze(),2] = self.outer_sphere_z_low

    def get_init_sphere_pts(self):
        sphere_pts = (torch.rand((self.config.init_pts_sphere_num, 3)) - 0.5) * 2
        dists = torch.linalg.vector_norm(sphere_pts, dim=1, keepdim=True)
        rescale = self.outer_sphere_rad / (dists + 1e-8)
        sphere_pts = sphere_pts * torch.cat([rescale, rescale, rescale], dim=1)
        if self.config.init_pts_hemisphere:
            sphere_pts[(sphere_pts[:,2] < self.outer_sphere_z_low).squeeze(),2] = self.outer_sphere_z_low
        return sphere_pts
    
    def add_init_sphere_pts(self):
        # Get new points
        sphere_pts = self.get_init_sphere_pts().to(self.means.device)
        dim_sh = num_sh_bases(self.config.sh_degree)
        distances, _ = self.k_nearest_sklearn(sphere_pts.data, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        sphere_colors    = torch.rand(self.config.init_pts_sphere_num, 1, 3).to(self.colors.device)
        sphere_shs_rest  = torch.zeros((self.config.init_pts_sphere_num, dim_sh - 1, 3)).to(self.shs_rest.device)
        sphere_scales    = torch.log(avg_dist.repeat(1, 3)).to(self.scales.device)
        sphere_opacities = torch.logit(0.1 * torch.ones(self.config.init_pts_sphere_num, 1)).to(self.opacities.device)
        sphere_quats     = random_quat_tensor(self.config.init_pts_sphere_num).to(self.quats.device)
        # Add to model
        self.means     = torch.nn.Parameter(torch.cat([self.means.detach(),     sphere_pts],       dim=0))
        self.colors    = torch.nn.Parameter(torch.cat([self.colors.detach(),    sphere_colors],    dim=0))
        self.shs_rest  = torch.nn.Parameter(torch.cat([self.shs_rest.detach(),  sphere_shs_rest],  dim=0))
        self.scales    = torch.nn.Parameter(torch.cat([self.scales.detach(),    sphere_scales],    dim=0))
        self.opacities = torch.nn.Parameter(torch.cat([self.opacities.detach(), sphere_opacities], dim=0))
        self.quats     = torch.nn.Parameter(torch.cat([self.quats.detach(),     sphere_quats],     dim=0))

    def populate_modules(self):
        self.outer_sphere_rad = 0.5 * (self.config.init_pts_sphere_rad_min + self.config.init_pts_sphere_rad_max) # Usually overwritten
        if self.seed_pts is not None and not self.config.random_init:
            self.means = torch.nn.Parameter(self.seed_pts[0])  # (Location, Color)
        else:
            self.means = torch.nn.Parameter((torch.rand((100000, 3)) - 0.5) * 10)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(self.means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        self.scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))

        self.quats = torch.nn.Parameter(random_quat_tensor(self.num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if self.seed_pts is not None and not self.config.random_init:
            fused_color = RGB2SH(self.seed_pts[1] / 255)
            shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().cuda()
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
            self.colors = torch.nn.Parameter(shs[:, 0:1, :])
            self.shs_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            self.colors = torch.nn.Parameter(torch.rand(self.num_points, 1, 3))
            self.shs_rest = torch.nn.Parameter(torch.zeros((self.num_points, dim_sh - 1, 3)))
        self.opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1)))

        # Init outer sphere dimensions & points
        self.outer_sphere_rad = self.get_init_sphere_radius()
        if self.config.init_pts_hemisphere:
            self.outer_sphere_z_low = self.get_init_bottom_plane()
        if self.config.init_pts_sphere_on_iter < 0 and self.config.init_pts_sphere_num > 0:
            self.add_init_sphere_pts()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.step = 0
        self.crop_box: Optional[OrientedBox] = None
        # record camera positions
        self.cameras_loaded = False
        self.cameras = []

        self.early_stop_at_step: Optional[int] = None
        self.avg_loss = 1.0
        self.min_avg_loss = 1.0
        self.prev_avg_loss = 1.0

    @property
    def get_colors(self):
        color = self.colors
        shs_rest = self.shs_rest
        return torch.cat((color, shs_rest), dim=1)

    def load_state_dict(self, dict, **kwargs):
        # resize the parameters to match the new number of points
        self.step = self.config.max_iterations
        newp = dict["means"].shape[0]
        self.means = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.scales = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.quats = torch.nn.Parameter(torch.zeros(newp, 4, device=self.device))
        self.colors = torch.nn.Parameter(torch.zeros(self.num_points, 1, 3, device=self.device))
        self.shs_rest = torch.nn.Parameter(
            torch.zeros(self.num_points, num_sh_bases(self.config.sh_degree) - 1, 3, device=self.device)
        )
        self.opacities = torch.nn.Parameter(torch.zeros(newp, 1, device=self.device))
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

    @profiler.time_function
    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

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

    @profiler.time_function
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

    @profiler.time_function
    def add_to_optim(self, optimizer, new_params, n):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        param_size = tuple(param_state["exp_avg"].size())
        param_size = (n,) + param_size[1:]
        param_state["exp_avg"] = torch.cat([param_state["exp_avg"], torch.zeros(param_size).cuda()], dim=0)
        param_state["exp_avg_sq"] = torch.cat([param_state["exp_avg_sq"], torch.zeros(param_size).cuda()], dim=0)
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    @profiler.time_function
    def after_train(self, step):
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.grad.detach().norm(dim=-1)  # TODO fill in
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii,dtype=torch.float32)
            newradii = self.radii[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(self.max_2Dsize[visible_mask], newradii / float(max(self.last_size[0], self.last_size[1])))

            if (step + 5) % self.config.early_stop_check_every == 0:
                if not self.early_stop_at_step and self.should_stop_early(step):
                    self.early_stop_at_step = step + self.config.early_stop_additional_steps
                    print('\n' * 15
                          + f"Early stopping triggered at step {step}, stopping at step {self.early_stop_at_step}"
                          + '\n' * 15)
                # New target to compare against
                self.prev_avg_loss = self.avg_loss
                # Also reset the min, because sometimes the loss increases
                self.min_avg_loss = self.avg_loss

    def should_stop_early(self, step: int) -> bool:
        # Don't stop early if we've recently reset the alpha
        reset_interval = self.config.reset_alpha_every * self.config.refine_every
        steps_since_reset = step % reset_interval
        if steps_since_reset < self.config.early_stop_check_every:
            return False

        # Don't stop early if we're almost done
        if self.config.max_iterations - step < self.config.early_stop_additional_steps:
            return False

        # Don't stop early if loss is too high
        if self.avg_loss > self.config.early_stop_max_loss:
            return False

        # Ensure alpha reset runs at least once
        if step < reset_interval:
            return False

        threshold = self.config.early_stop_loss_diff_threshold * self.config.early_stop_check_every
        print(f'Checking if we should stop early. Min avg loss: {self.min_avg_loss}, '
              f'previous avg loss: {self.prev_avg_loss}, threshold: {threshold}')
        return self.prev_avg_loss - self.min_avg_loss < threshold

    def set_crop(self, crop_box: OrientedBox):
        self.crop_box = crop_box

    def set_background(self, back_color: torch.Tensor):
        assert back_color.shape == (3,)
        self.back_color = back_color

    @profiler.time_function
    def refinement_after(self, optimizers: Optimizers, step):
        # Don't split or cull if we're stopping early
        if self.early_stop_at_step is not None:
            return
        stop_refine_at_iter = self.config.stop_refine_at * self.config.max_iterations
        if self.step > self.config.warmup_length and self.step < stop_refine_at_iter:
            if self.num_points < self.config.max_gaussians:
                # Set densify_grad_thresh between densify_grad_thresh_init & densify_grad_thresh_final
                ratio = 0 # [0-1] where 0 -> densify_grad_thresh_init and 1 -> densify_grad_thresh_final
                num_splats_start_increase = self.config.max_gaussians * self.config.densify_grad_thresh_start_increase
                if self.num_points > num_splats_start_increase:
                    ratio1 = (self.num_points - num_splats_start_increase) / (self.config.max_gaussians - num_splats_start_increase)
                    if ratio1 > ratio and ratio1 <= 1:
                        ratio = ratio1
                densify_until_iter_start_increase = self.config.densify_until_iter_start_increase * self.config.max_iterations
                if self.step > densify_until_iter_start_increase:
                    ratio2 = (self.step - densify_until_iter_start_increase) / (stop_refine_at_iter - densify_until_iter_start_increase)
                    if ratio2 > ratio and ratio2 <= 1:
                        ratio = ratio2
                const = math.log(self.config.densify_grad_thresh_final / self.config.densify_grad_thresh_init)
                densify_grad_thresh = self.config.densify_grad_thresh_init * math.exp(const * ratio)
                with torch.no_grad():
                    # then we densify
                    avg_grad_norm = (
                        (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                    )
                    high_grads = (avg_grad_norm > densify_grad_thresh).squeeze()
                    splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                    splits &= high_grads
                    nsamps = 2
                    (
                        split_means,
                        split_colors,
                        split_shs,
                        split_opacities,
                        split_scales,
                        split_quats,
                    ) = self.split_gaussians(splits, nsamps)

                    dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                    dups &= high_grads
                    dup_means, dup_colors, dup_shs, dup_opacities, dup_scales, dup_quats = self.dup_gaussians(dups)
                    self.means = Parameter(torch.cat([self.means.detach(), split_means, dup_means], dim=0))
                    self.colors = Parameter(torch.cat([self.colors.detach(), split_colors, dup_colors], dim=0))
                    self.shs_rest = Parameter(torch.cat([self.shs_rest.detach(), split_shs, dup_shs], dim=0))
                    self.opacities = Parameter(
                        torch.cat([self.opacities.detach(), split_opacities, dup_opacities], dim=0)
                    )
                    self.scales = Parameter(torch.cat([self.scales.detach(), split_scales, dup_scales], dim=0))
                    self.quats = Parameter(torch.cat([self.quats.detach(), split_quats, dup_quats], dim=0))
                    # append zeros to the max_2Dsize tensor
                    self.max_2Dsize = torch.cat(
                        [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                        dim=0,
                    )
                    split_idcs = torch.where(splits)[0]
                    param_groups = self.get_param_groups()
                    for group, param in param_groups.items():
                        self.dup_in_optim(optimizers.optimizers[group], split_idcs, param, n=nsamps)
                    dup_idcs = torch.where(dups)[0]
                    param_groups = self.get_param_groups()
                    for group, param in param_groups.items():
                        self.dup_in_optim(optimizers.optimizers[group], dup_idcs, param, 1)

                    # only cull if we've seen every image since opacity reset
                    reset_interval = self.config.reset_alpha_every * self.config.refine_every
                    if self.step % reset_interval > self.num_train_data:
                        # then cull
                        deleted_mask = self.get_cull_gaussians()
                        self.cull_gaussians(optimizers, deleted_mask)

                    if self.step // self.config.refine_every % self.config.reset_alpha_every == 0:
                        reset_value = 0.01
                        self.opacities.data = torch.full_like(
                            self.opacities.data, torch.logit(torch.tensor(reset_value)).item()
                        )
                        # reset the exp of optimizer
                        optim = optimizers.optimizers["opacity"]
                        param = optim.param_groups[0]["params"][0]
                        param_state = optim.state[param]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
                    self.xys_grad_norm = None
                    self.vis_counts = None
                    self.max_2Dsize = None


    @profiler.time_function
    def add_sphere_pts(self, optimizers: Optimizers, step):
        if self.config.init_pts_sphere_on_iter == self.step and self.config.init_pts_sphere_num > 0:
            self.add_init_sphere_pts()
            if not self.max_2Dsize is None: # append zeros to the max_2Dsize tensor
                sphere_max_2Dsize = torch.nn.Parameter(torch.zeros((self.config.init_pts_sphere_num)).cuda())
                self.max_2Dsize = torch.cat([self.max_2Dsize, sphere_max_2Dsize], dim=0)
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None
            param_groups = self.get_param_groups()
            for group, param in param_groups.items():
                self.add_to_optim(optimizers.optimizers[group], param, self.config.init_pts_sphere_num)
        

    @profiler.time_function
    def refinement_last(self, optimizers: Optimizers, step):
        if step != (self.early_stop_at_step or self.config.max_iterations) - 1:
            return
        # At the end of training, remove gaussians that are only seen by 1 camera
        if self.config.remove_gaussians_min_cameras_in_fov > 0:
            delete_mask = (self.gaussians_camera_cnt < self.config.remove_gaussians_min_cameras_in_fov).squeeze()
            self.cull_gaussians(optimizers, delete_mask)
        # At the end of training, remove gaussians that are outside the outer sphere * remove_gaussians_outside_sphere
        if self.config.remove_gaussians_outside_sphere > 1.0:
            radius = self.config.remove_gaussians_outside_sphere * self.outer_sphere_rad
            delete_mask = (torch.linalg.vector_norm(self.means, dim=1) > radius).squeeze()
            self.cull_gaussians(optimizers, delete_mask)
    
    @profiler.time_function
    def get_cull_gaussians(self):
        """
        This function determineds which gaussians to cull under a certain opacity threshold
        """
        # cull transparent ones
        delete_mask = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        step0_cull_big = self.config.refine_every * self.config.reset_alpha_every
        if self.step > step0_cull_big:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            delete_mask = delete_mask | toobigs
            # cull big screen space
            stop_refine_at_iter = self.config.stop_refine_at * self.config.max_iterations
            step_ratio = (self.step - step0_cull_big) / (stop_refine_at_iter - step0_cull_big)
            cull_screen_size = self.config.cull_screen_size_init + (self.config.cull_screen_size_final - self.config.cull_screen_size_init) * step_ratio
            delete_mask = delete_mask | (self.max_2Dsize > cull_screen_size).squeeze()
        return delete_mask

    @profiler.time_function
    def cull_gaussians(self, optimizers: Optimizers, delete_mask):
        n_bef = self.num_points
        self.means = Parameter(self.means[~delete_mask].detach())
        self.scales = Parameter(self.scales[~delete_mask].detach())
        self.quats = Parameter(self.quats[~delete_mask].detach())
        self.colors = Parameter(self.colors[~delete_mask].detach())
        self.shs_rest = Parameter(self.shs_rest[~delete_mask].detach())
        self.opacities = Parameter(self.opacities[~delete_mask].detach())
        param_groups = self.get_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], delete_mask, param)
        print(f"Culled {n_bef - self.num_points} gaussians")

    @profiler.time_function
    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        print(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        if self.config.densify_snap_to_outer_sphere:
            self.snap_to_outer_sphere_if_outside(new_means)
        # step 2, sample new colors
        new_colors = self.colors[split_mask].repeat(samps, 1, 1)
        # step 3, sample new shs
        new_shs = self.shs_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / 1.6).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / 1.6)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        return new_means, new_colors, new_shs, new_opacities, new_scales, new_quats

    @profiler.time_function
    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        dup_means = self.means[dup_mask]
        dup_colors = self.colors[dup_mask]
        dup_shs = self.shs_rest[dup_mask]
        dup_opacities = self.opacities[dup_mask]
        dup_scales = self.scales[dup_mask]
        dup_quats = self.quats[dup_mask]
        return dup_means, dup_colors, dup_shs, dup_opacities, dup_scales, dup_quats

    @property
    def num_points(self):
        return self.means.shape[0]

    @profiler.time_function
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
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.add_sphere_pts,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_last,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        return {
            "xyz": [self.means],
            "color": [self.colors],
            "shs": [self.shs_rest],
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats],
        }

    def _get_downscale_factor(self):
        return 2 ** max((self.config.num_downscales - self.step // self.config.resolution_schedule), 0)
        
    @profiler.time_function
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
        # record the camera info for later
        if not self.cameras_loaded:
            add_cam = True
            for cam in self.cameras:
                cam_diff = np.linalg.norm((cam.camera_to_worlds - camera.camera_to_worlds).cpu().squeeze().numpy())
                if cam_diff < 1e-6:
                    add_cam = False
                    break
            if add_cam:
                self.cameras.insert(1, camera)
            self.cameras_loaded = self.step > 2 * self.num_train_data # Sometimes self.num_train_data is not accurate
            if self.cameras_loaded:
                print("Cameras loaded: " + str(len(self.cameras)) + " cameras, self.num_train_data = " + str(self.num_train_data))
                for i in range(10): # Hack to make prior print line visible
                    print("\n")
        # dont mutate the input
        camera = deepcopy(camera)
        if self.training:
            d = self._get_downscale_factor()
            camera.rescale_output_resolution(1 / d)
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return {"rgb": torch.full((camera.height.item(), camera.width.item(), 3), 0.5, device=self.device)}
        else:
            crop_ids = torch.ones_like(self.means[:, 0], dtype=torch.bool)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[..., :3, :3]  # 1 x 3 x 3
        T = camera.camera_to_worlds[..., :3, 3:4]  # 1 x 3 x 1
        R = vtf.SO3.from_matrix(R.cpu().squeeze().numpy())
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.from_numpy(R.as_matrix()[None, ...]).to(self.device, torch.float32)
        viewmat = torch.cat([R, T], dim=2)
        # add a row of zeros and a 1 to the bottom of the viewmat
        viewmat = torch.cat([viewmat, torch.tensor([[[0, 0, 0, 1]]], device=self.device)], dim=1)
        # invert it
        viewmat = torch.inverse(viewmat)
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = camera.width.item(), camera.height.item()
        self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy).to(self.device)
        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = (
            (W + BLOCK_X - 1) // BLOCK_X,
            (H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        if self.training:
            if random.uniform(0,self.config.max_iterations) > self.step and not self.early_stop_at_step:
                background = torch.rand(3, device=self.device)
            else:
                background = torch.ones(3)
        else:
            background = torch.ones(3)

        opacities_crop = self.opacities[crop_ids]
        means_crop = self.means[crop_ids]
        colors_crop = self.get_colors[crop_ids]
        self.xys, depths, self.radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(
            self.means[crop_ids],
            torch.exp(self.scales[crop_ids]),
            1,
            self.quats[crop_ids],
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
        # Important to allow xys grads to populate properly
        if self.training:
            self.xys.retain_grad()
        rend_mask = self.radii > 0
        if self.config.sh_degree > 0:
            viewdirs = means_crop[rend_mask].detach() - camera.camera_to_worlds[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            n_coeffs = num_sh_bases(n)
            with profiler.time_function("sh_coeff_permute"):
                coeffs = colors_crop[rend_mask, :n_coeffs, :]
            rgbs = SphericalHarmonics.apply(n, viewdirs, coeffs)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            coeffs = []
            rgbs = self.get_colors.squeeze()[rend_mask]  # (N, 3)
            rgbs = torch.sigmoid(rgbs)
        rgb = RasterizeGaussians.apply(
            self.xys[rend_mask],
            depths[rend_mask],
            self.radii[rend_mask],
            conics[rend_mask],
            num_tiles_hit[rend_mask],
            rgbs,
            torch.sigmoid(opacities_crop[rend_mask]),
            H,
            W,
            background,
        )
        if self.training:
            depth_im = None
        else:
            depth_im = RasterizeGaussians.apply(
                self.xys[rend_mask],
                depths[rend_mask],
                self.radii[rend_mask],
                conics[rend_mask],
                num_tiles_hit[rend_mask],
                depths[rend_mask, None].repeat(1, 3),
                torch.sigmoid(opacities_crop[rend_mask]),
                H,
                W,
                torch.ones(3, device=self.device) * 10,
            )[..., 0:1]
        # At the end of training, remove gaussians that are only seen by 1 camera -> keep track of count
        ending_step = self.early_stop_at_step or self.config.max_iterations
        if self.training and self.config.remove_gaussians_min_cameras_in_fov > 0 and self.step >= ending_step - len(self.cameras):
            if self.step == ending_step - len(self.cameras):
                self.gaussians_camera_cnt = torch.nn.Parameter(torch.zeros(self.num_points, 1, device=self.device))
            assert self.num_points == len(self.gaussians_camera_cnt)
            with torch.no_grad():
                self.gaussians_camera_cnt[rend_mask] += 1
        return {"rgb": rgb, "depth": depth_im, "means_rendered": means_crop[rend_mask], "depths_rendered": depths[rend_mask], "sh": coeffs}

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """

        return {
            'avg_loss': torch.Tensor([self.avg_loss]),
            'min_avg_loss': torch.Tensor([self.min_avg_loss]),
            'loss_diff': torch.Tensor([self.prev_avg_loss - self.min_avg_loss]),
        }
    
    @profiler.time_function
    def get_depth_loss(self, depths):
        if self.config.dist2cam_loss_lambda == 0.0:
            return 0.0
        return torch.abs(self.config.dist2cam_loss_size - depths).mean() / self.config.dist2cam_loss_size
    
    @profiler.time_function
    def get_sh_regularization_loss(self, sh_coeffs):
        if self.config.sh_degree <= 0 or self.config.regularize_sh_lambda <= 0.0:
            return 0.0
        n_gauss = sh_coeffs.size()[0]
        if sh_coeffs.size()[1] <= 1 or n_gauss <= 0:
            return 0.0
        return torch.linalg.vector_norm(sh_coeffs[:, 1:, :], dim=None) / n_gauss
    
    @profiler.time_function
    def get_outer_sphere_loss(self, means):
        if self.config.outside_outer_sphere_lambda == 0.0:
            return 0.0
        dists_out_of_sphere = torch.linalg.vector_norm(means, dim=1) - self.outer_sphere_rad
        error = torch.clamp(dists_out_of_sphere, min=0) # Only penalize if gaussians are outside the sphere
        error2 = torch.square(error)
        return error2.mean()
    
    @profiler.time_function
    def get_under_hemisphere_loss(self, means):
        if self.config.under_hemisphere_lambda == 0.0 or not self.config.init_pts_hemisphere:
            return 0.0
        error = torch.clamp(self.outer_sphere_z_low - means[:,2], min=0) # Only penalize if gaussians are below z_low
        error2 = torch.square(error)
        return error2.mean()

    @profiler.time_function
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        d = self._get_downscale_factor()
        if d > 1:
            # use torchvision to resize
            import torchvision.transforms.functional as TF

            newsize = (batch["image"].shape[0] // d, batch["image"].shape[1] // d)
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
        Ll1 = torch.abs(gt_img - outputs["rgb"]).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], outputs["rgb"].permute(2, 0, 1)[None, ...])
        sh_L2 = self.get_sh_regularization_loss(outputs["sh"]) * self.config.regularize_sh_lambda # Penalize for high spherical harmonics values
        depth_L1 = self.get_depth_loss(outputs["depths_rendered"]) * self.config.dist2cam_loss_lambda # Push gaussians away from camera
        outer_sphere_L2 = self.get_outer_sphere_loss(outputs["means_rendered"]) * self.config.outside_outer_sphere_lambda # Pull gaussians inward if they go beyond the outer_sphere
        under_hemisphere_L2 = self.get_under_hemisphere_loss(outputs["means_rendered"]) * self.config.under_hemisphere_lambda # Pull gaussians up if they go under the lower plane

        main_loss = (1 - self.config.ssim_lambda) * Ll1 +  self.config.ssim_lambda * simloss

        avg_loss_decay = 1 - 1 / (2 * self.num_train_data)
        self.avg_loss = self.avg_loss * avg_loss_decay + main_loss.item() * (1 - avg_loss_decay)
        if self.avg_loss < self.min_avg_loss:
            self.min_avg_loss = self.avg_loss

        return {
            "main": main_loss,
            "depth": depth_L1,
            "sh": sh_L2,
            "outer_sphere": outer_sphere_L2,
            "under_hemisphere": under_hemisphere_L2,
        }

    @torch.no_grad()
    @profiler.time_function
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, camera: Optional[Cameras] = None
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        assert camera is not None, "must provide camera to gaussian model"
        outs = self.get_outputs(camera.to(self.device))
        return outs

    @profiler.time_function
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
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
