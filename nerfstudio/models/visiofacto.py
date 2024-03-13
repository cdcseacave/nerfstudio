# ruff: noqa: E741
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

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from gsplat._torch_impl import quat_to_rotmat
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import num_sh_bases, spherical_harmonics
from pytorch_msssim import SSIM
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import quaternion_from_vectors
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.models.splatfacto import random_quat_tensor, RGB2SH, SH2RGB, projection_matrix

@dataclass
class VisiofactoModelConfig(ModelConfig):
    """Visiofacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: VisiofactoModel)
    warmup_length: int = 500  # splatfacto: 500, gaussian splatting: 1000
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 250  # splatfacto: 250, gaussian splatting: 200
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 0  # splatfacto: 0, gaussian splatting: 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1  # splatfacto: 0.1, gaussian splatting: 0.01
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5  # splatfacto: 0.5, gaussian splatting: 10.0
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: Optional[float] = 0.0002 #  Splatfacto 0.0002, Gaussian Splatting - dynamically computed (here none)
    """threshold of positional gradient norm for densifying gaussians. If none, is computed dynamically 
       See densify_grad_thresh_init, densify_grad_thresh_final, densify_grad_thresh_start_increase"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000  # splatfacto: 1000, gaussian splatting: 500
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    sh_degree: int = 2  # splatfacto: 3, gaussian splatting: 2
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = True
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 50.0
    """threshold of ratio of gaussian max to min scale before applying regularization
       loss from the PhysGaussian paper"""
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""

    """ Visiofacto specific """
    min_iterations: int = 9000
    """Minimum number of iterations"""
    max_iterations: int = 30000
    """Sets the defaults in method_configs.py.
       Multiple config values will be messed up if num_iterations is set via the command line arg.
       Change it here instead"""
    max_gaussians: int = 5000000
    """Max number of 3D gaussians.
       As this number is approached, densify_grad_thresh is increased to slow down densification.
       It's possible for n_gaussians to go over this number by a bit, but is unlikely"""

    """ REFINEMENT (DENSIFICATION AND CULLING) """
    stop_refine_at: float = 0.9
    stop_refine_at_step = math.ceil(max_iterations * stop_refine_at)
    """Stop splitting & culling at this step. [Multiply this value (0-1) by max_iterations]
       Must be < max_iterations or remove_gaussians_min_cameras_in_fov won't work """
    cull_screen_size_init: float = 2.0
    """If a gaussian is more than this percent of screen space, cull it """
    cull_screen_size_final: float = 0.20
    """If a gaussian is more than this percent of screen space, cull it """
    densify_grad_thresh_init: float = 0.0002
    """Threshold of positional gradient norm for densifying gaussians.
       Also see densify_grad_thresh_start_increase"""
    densify_grad_thresh_final: float = 10.0 * densify_grad_thresh_init
    """See densify_grad_thresh_start_increase"""
    densify_grad_thresh_start_increase: float = 0.7
    """[0-1] Start increasing densify_grad_thresh from init -> final once there are
       densify_grad_thresh_start_increase * max_num_splats of splats"""
    densify_until_iter_start_increase: float = 0.5
    """[Multiply this value (0-1) by max_iterations] After this many iterations,
       make it harder to densify (value should be lower than densify_until_iter)"""
    densify_snap_to_outer_sphere: bool = False
    """If true, snap densified points to the outer_sphere"""

    """ INITIALIZATION """
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    """Size of the cube to initialize random gaussians within"""
    init_pts_sphere_num: int = 20000
    """Initialize gaussians at a sphere with this many randomly placed points. Set to 0 to disable"""
    init_pts_sphere_rad_pct: float = 0.98
    """Initialize gaussians at a sphere: set radius based on looking at the 99th percentile
       of initial points' distance from origin"""
    init_pts_sphere_rad_mult: float = 1.5
    """Initialize gaussians at a sphere: set radius based on init_pts_sphere_rad_pct * this value"""
    init_pts_sphere_rad_min: float = 5.0
    """Initialize gaussians at a sphere with this as the minimum radius"""
    init_pts_sphere_rad_max: float = 20.0
    """Initialize gaussians at a sphere with this as the maximum radius"""
    init_pts_hemisphere: bool = True
    """Bottom half of the sphere is flat (hemisphere)"""

    """ LOSS FUNCTION """
    ssim_lambda: float = 0.2
    """Weight of ssim loss """
    dist2cam_loss_size: float = 30.0
    """Pull points away from camera until it gets this far away.
       Should probably be a bit larger than init_pts_sphere_rad_max"""
    dist2cam_loss_lambda: float = 0.01
    """Pull points away from camera: add L1 term to loss function that has a range [0-1] -> multiply by this number"""
    regularize_sh_lambda: float = 0.01
    """sh coeffs are multiplied by x, y, z, xx, xy, xz, etc, where [x,y,z] is a unit vector
       (no coeff normalization needed) so all coeffs are summed together for L2 loss -> multiply by this number"""
    outside_outer_sphere_lambda: float = 1.0
    """Penalty for gaussians going outside the outer sphere"""
    under_hemisphere_lambda: float = 1.0
    """Penalty for gaussians going under the outer hemisphere"""
    scale_lambda: float = 1
    """ Weight of scale loss """
    opacity_lambda: float = 0.00003
    """ Weight of opacity loss (0 to disable) """
    opacity_binarization_lambda: float = 0.0
    """ Weight of opacity binarization loss (0 to disable) """

    """ EARLY STOPPING """
    early_stop_check_every: int = 500
    """Check every n steps if the loss has stopped decreasing"""
    early_stop_loss_diff_threshold: float = 2.8e-6
    """If the loss has decreased by less than this amount per step, stop training"""
    early_stop_max_loss: float = 0.2
    """Don't stop early if the loss is greater than this"""
    early_stop_additional_steps: int = 2000
    """If early stopping is triggered, train for this many more steps before stopping"""

    """ POST-PROCESSING"""
    remove_gaussians_min_cameras_in_fov: int = 0
    """At the end of training, only retain gaussians if they are in the field of view of at least this many cameras.
       Set to 0 to disable"""
    remove_gaussians_outside_sphere: float = 1.2
    """At the end of training, remove gaussians if they are outside the outer_sphere_radius * this value.
       Set to 0 to disable"""

    """ CAMERA OPTIMIZATION """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=CameraOptimizerConfig)
    """camera optimizer configuration"""


class VisiofactoModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Visiofacto configuration to instantiate model
    """

    config: VisiofactoModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points

        if "base_dir" in kwargs and kwargs["base_dir"] != "":
            self.base_dir = Path(kwargs["base_dir"])
            # open json file to save cull stats
            self.cull_stats_file = open(self.base_dir / "cull_stats.cvs", "w")
            self.cull_stats_file.write(f"Step,Points,Culled,Below alpha,Too bigs\n")
        else:
            self.base_dir = None
            self.cull_stats_file = None

        super().__init__(*args, **kwargs)

    def get_init_sphere_radius(self) -> float:
        dists = torch.linalg.vector_norm(self.means, dim=1).detach().numpy()
        dist_pct = np.percentile(dists, self.config.init_pts_sphere_rad_pct * 100, overwrite_input=False)
        dist_raw = dist_pct * self.config.init_pts_sphere_rad_mult
        rad = max(min(dist_raw, self.config.init_pts_sphere_rad_max), self.config.init_pts_sphere_rad_min)
        return rad

    def get_init_bottom_plane(self) -> float:
        zs = self.means[:, 2].detach().numpy()
        z_low = np.percentile(zs, (1.0 - self.config.init_pts_sphere_rad_pct) * 100, overwrite_input=False)
        return z_low

    def snap_to_outer_sphere_if_outside(self, means):
        # If already inside the outer sphere, this does nothing
        # else: find closest point inside the outer sphere -> return that
        dists = torch.linalg.vector_norm(means, dim=1, keepdim=True)
        snap_to_sphere = (dists > self.outer_sphere_rad).squeeze()
        rescale = self.outer_sphere_rad / dists[snap_to_sphere]
        means[snap_to_sphere, :] = means[snap_to_sphere, :] * torch.cat([rescale, rescale, rescale], dim=1)
        if self.config.init_pts_hemisphere:
            means[(means[:, 2] < self.outer_sphere_z_low).squeeze(), 2] = self.outer_sphere_z_low

    def add_to_model(self, params):
        for k, v in params.items():
            newval = torch.nn.Parameter(torch.cat([getattr(self, k).detach(), v.to(getattr(self, k).device)], dim=0))
            setattr(self, k, newval)

    def init_scale_rotation(
        self, points: torch.Tensor, normals: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_points = points.shape[0]
        distances, _ = self.k_nearest_sklearn(points, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True) * 1.2
        if normals is None:
            # use random initialization of the covariance
            scales = torch.log(avg_dist.repeat(1, 3))
            quats = random_quat_tensor(num_points)
        else:
            # use the normals to initialize the covariance
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

    def get_init_sphere_means(self):
        sphere_pts = (torch.rand((self.config.init_pts_sphere_num, 3)) - 0.5) * 2
        dists = torch.linalg.vector_norm(sphere_pts, dim=1, keepdim=True)
        rescale = self.outer_sphere_rad / (dists + 1e-8)
        sphere_pts = sphere_pts * torch.cat([rescale, rescale, rescale], dim=1)
        if self.config.init_pts_hemisphere:
            sphere_pts = sphere_pts[(sphere_pts[:, 2] > self.outer_sphere_z_low).squeeze()]
        return sphere_pts

    def get_init_sphere_pts(self):
        new_pts = {}
        if self.config.init_pts_sphere_num <= 0:
            return new_pts
        dim_sh = num_sh_bases(self.config.sh_degree)
        new_pts['means'] = self.get_init_sphere_means()
        pts_sphere_num = new_pts['means'].shape[0]
        new_pts['features_dc'] = torch.rand(pts_sphere_num, 3)
        new_pts['features_rest'] = torch.zeros((pts_sphere_num, dim_sh - 1, 3))
        new_pts['scales'], new_pts['quats'] = self.init_scale_rotation(new_pts['means'], -new_pts['means'] / self.outer_sphere_rad)
        new_pts['opacities'] = torch.logit(0.9 * torch.ones(pts_sphere_num, 1))
        return new_pts

    def populate_modules(self):
        self.xys_grad_norm = None
        self.max_2Dsize = None

        dim_sh = num_sh_bases(self.config.sh_degree)
        # self.seed_pts contains (Location, Color, Normal)
        if self.seed_points is not None and not self.config.random_init:
            points = self.seed_points[0]
            normals = self.seed_points[2] if len(self.seed_points) > 2 else None
            CONSOLE.log(f"Initializing {points.shape[0]} splats from seed points",
                        "with normals" if normals is not None else "without normals")
            scales, quats = self.init_scale_rotation(points, normals)
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3), dtype=torch.float)
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            self.features_dc = torch.nn.Parameter(shs[:, 0, :])
            self.features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            num_points = self.config.num_random
            points = (torch.rand((num_points, 3), dtype=torch.float) - 0.5) * 10
            scales, quats = self.init_scale_rotation(points)
            self.features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            self.features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        self.means = torch.nn.Parameter(points)
        self.scales = torch.nn.Parameter(scales)
        self.quats = torch.nn.Parameter(quats)

        initial_opacity = 0.9 if self.config.opacity_lambda > 0 else 0.1
        self.opacities = torch.nn.Parameter(torch.logit(initial_opacity * torch.ones(self.num_points, 1)))

        # Init outer sphere dimensions
        self.outer_sphere_rad = self.get_init_sphere_radius()
        if self.config.init_pts_hemisphere:
            self.outer_sphere_z_low = self.get_init_bottom_plane()
            CONSOLE.log(f"Initializing bottom of hemisphere to {self.outer_sphere_z_low}")
        # Init outer sphere points
        if self.training:
            new_pts = self.get_init_sphere_pts()
            if len(new_pts) > 0:
                self.add_to_model(new_pts)
                CONSOLE.log(f"Initializing {new_pts['means'].shape[0]} 3D gaussians on a "
                            f"{'hemi' if self.config.init_pts_hemisphere else ''}sphere at radius = {self.outer_sphere_rad}")

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None

        self.early_stop_at_step: Optional[int] = None
        self.avg_loss = 1.0
        self.min_avg_loss = 1.0
        self.prev_avg_loss = 1.0

        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser.
               # This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        CONSOLE.log(f"Initialized {self.num_points} splats with SH degree {self.config.sh_degree}")

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = self.config.max_iterations
        newp = dict["means"].shape[0]
        self.means = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.scales = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.quats = torch.nn.Parameter(torch.zeros(newp, 4, device=self.device))
        self.opacities = torch.nn.Parameter(torch.zeros(newp, 1, device=self.device))
        self.features_dc = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.features_rest = torch.nn.Parameter(
            torch.zeros(newp, num_sh_bases(self.config.sh_degree) - 1, 3, device=self.device)
        )
        # TODO: Could colors be transposed? See gaussian_splatting.py.
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
        from sklearn.neighbors import NearestNeighbors

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

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
        param_state["exp_avg"] = torch.cat(
            [
                param_state["exp_avg"],
                torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
            ],
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

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

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

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_refine_at_step:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            assert self.xys.grad is not None
            grads = self.xys.grad.detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

            if step % self.config.early_stop_check_every == 0:
                if not self.early_stop_at_step and self.should_stop_early(step):
                    self.early_stop_at_step = step + self.config.early_stop_additional_steps
                    CONSOLE.log('\n' * 15
                          + f"Early stopping triggered at step {step}, stopping at step {self.early_stop_at_step}"
                          + '\n' * 15)
                # New target to compare against
                self.prev_avg_loss = self.avg_loss
                # Also reset the min, because sometimes the loss increases
                self.min_avg_loss = self.avg_loss

    def should_stop_early(self, step: int) -> bool:
        # Don't stop early if we're did not run the minimum number of iterations
        if step < self.config.min_iterations:
            return False
        
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
        CONSOLE.log(f'Checking if we should stop early. Min avg loss: {self.min_avg_loss}, '
                    f'previous avg loss: {self.prev_avg_loss}, threshold: {threshold}')
        return self.prev_avg_loss - self.min_avg_loss < threshold

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def get_densify_grad_tresh(self):
        if self.config.densify_grad_thresh is not None:
            return self.config.densify_grad_thresh
        # Set densify_grad_thresh between densify_grad_thresh_init & densify_grad_thresh_final
        ratio = 0  # [0-1] where 0 -> densify_grad_thresh_init and 1 -> densify_grad_thresh_final
        num_splats_start_increase = self.config.max_gaussians * self.config.densify_grad_thresh_start_increase
        if self.num_points > num_splats_start_increase:
            ratio1 = (self.num_points - num_splats_start_increase) / (
                        self.config.max_gaussians - num_splats_start_increase)
            if ratio1 > ratio and ratio1 <= 1:
                ratio = ratio1
        densify_until_iter_start_increase = self.config.densify_until_iter_start_increase * self.config.max_iterations
        if self.step > densify_until_iter_start_increase:
            ratio2 = (self.step - densify_until_iter_start_increase) / (
                        self.config.stop_refine_at * self.config.max_iterations - densify_until_iter_start_increase)
            if ratio2 > ratio and ratio2 <= 1:
                ratio = ratio2
        const = math.log(self.config.densify_grad_thresh_final / self.config.densify_grad_thresh_init)
        densify_grad_thresh = self.config.densify_grad_thresh_init * math.exp(const * ratio)
        return densify_grad_thresh

    def densify_gaussians(self, optimizers: Optimizers):
        assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
        densify_grad_thresh = self.get_densify_grad_tresh()

        avg_grad_norm = (
                (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
        )
        high_grads = (avg_grad_norm > densify_grad_thresh).squeeze()

        splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
        if self.step < self.config.stop_screen_size_at:
            splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()

        splits &= high_grads
        nsamps = self.config.n_split_samples
        split_pts = self.split_gaussians(splits, nsamps)

        dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
        dups &= high_grads

        dup_pts = self.dup_gaussians(dups)

        self.add_to_model(split_pts)
        self.add_to_model(dup_pts)

        # append zeros to the max_2Dsize tensor
        self.max_2Dsize = torch.cat(
            [self.max_2Dsize,
             torch.zeros_like(split_pts['scales'][:, 0]),
             torch.zeros_like(dup_pts['scales'][:, 0])],
            dim=0,
        )

        split_idcs = torch.where(splits)[0]
        self.dup_in_all_optim(optimizers, split_idcs, nsamps)

        dup_idcs = torch.where(dups)[0]
        self.dup_in_all_optim(optimizers, dup_idcs, 1)

        # After a guassian is split into two new gaussians, the original one should also be pruned.
        splits_mask = torch.cat(
            (
                splits,
                torch.zeros(
                    nsamps * splits.sum() + dups.sum(),
                    device=self.device,
                    dtype=torch.bool,
                ),
            )
        ).bool()

        return splits_mask

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step

        # Don't split or cull if we're stopping early
        # if self.early_stop_at_step is not None:
        #     return
        if self.step < self.config.warmup_length:
            return
        if self.step > self.config.stop_refine_at * self.config.max_iterations:
            return

        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                    self.step < self.config.stop_refine_at_step and self.step < 9000
                    and self.step % reset_interval > self.num_train_data + self.config.refine_every
                    and self.num_points < self.config.max_gaussians
                    and self.early_stop_at_step is None
            )

            if do_densification:
                splits_mask = self.densify_gaussians(optimizers)
                deleted_mask = self.get_cull_gaussians() | splits_mask
                self.cull_gaussians(deleted_mask, optimizers)
            elif self.config.continue_cull_post_densification: # and self.step >= self.config.stop_refine_at_step:
                deleted_mask = self.get_cull_gaussians()
                self.cull_gaussians(deleted_mask, optimizers)

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def refinement_last(self, optimizers: Optimizers, step):
        if step != (self.early_stop_at_step or self.config.max_iterations) - 1:
            return
        # At the end of training, remove gaussians that are only seen by 1 camera
        if self.config.remove_gaussians_min_cameras_in_fov > 0:
            delete_mask = (self.gaussians_camera_cnt < self.config.remove_gaussians_min_cameras_in_fov).squeeze()
            self.cull_gaussians(delete_mask, optimizers)
        # At the end of training, remove gaussians that are outside the outer sphere * remove_gaussians_outside_sphere
        if self.config.remove_gaussians_outside_sphere > 1.0:
            radius = self.config.remove_gaussians_outside_sphere * self.outer_sphere_rad
            delete_mask = (torch.linalg.vector_norm(self.means, dim=1) > radius).squeeze()
            self.cull_gaussians(delete_mask, optimizers)

    def get_cull_gaussians(self):
        n_bef = self.num_points

        culls = (torch.zeros_like(self.opacities)).squeeze().bool().to(self.device)

        # cull transparent ones
        below_alpha_count = 0
        if self.step > 2000 or self.config.opacity_lambda <= 0:
            culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
            below_alpha_count = torch.sum(culls).item()

        toobigs_count = 0
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (self.scales.exp().max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            toobigs_count = torch.sum(toobigs).item()
            culls = culls | toobigs

        cull_count = culls.sum()
        CONSOLE.log(
            f"Culled {cull_count} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {n_bef-cull_count} remaining)"
        )
        if self.cull_stats_file is not None:
            # save cull stats and flush to disk
            self.cull_stats_file.write(
                f"{self.step},{n_bef-cull_count},{cull_count},{below_alpha_count},{toobigs_count}\n")
            self.cull_stats_file.flush()

        return culls

    def cull_gaussians(self, culls: torch.Tensor, optimizers: Optional[Optimizers] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        self.means = Parameter(self.means[~culls].detach())
        self.scales = Parameter(self.scales[~culls].detach())
        self.quats = Parameter(self.quats[~culls].detach())
        self.features_dc = Parameter(self.features_dc[~culls].detach())
        self.features_rest = Parameter(self.features_rest[~culls].detach())
        self.opacities = Parameter(self.opacities[~culls].detach())

        if optimizers is not None:
            self.remove_from_all_optim(optimizers, culls)

        return culls
    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()

        new_pts = {}
        new_pts['means'] = rotated_samples + self.means[split_mask].repeat(samps, 1)
        if self.config.densify_snap_to_outer_sphere:
            self.snap_to_outer_sphere_if_outside(new_pts['means'])
        # step 2, sample new colors
        new_pts['features_dc'] = self.features_dc[split_mask].repeat(samps, 1)
        new_pts['features_rest'] = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_pts['opacities'] = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_pts['scales'] = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        # step 5, sample new quats
        new_pts['quats'] = self.quats[split_mask].repeat(samps, 1)
        return new_pts

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        dups = {}
        dups['means'] = self.means[dup_mask]
        dups['features_dc'] = self.features_dc[dup_mask]
        dups['features_rest'] = self.features_rest[dup_mask]
        dups['opacities'] = self.opacities[dup_mask]
        dups['scales'] = self.scales[dup_mask]
        dups['quats'] = self.quats[dup_mask]
        return dups

    @property
    def num_points(self):
        return self.means.shape[0]

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
                self.refinement_last,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "xyz": [self.means],
            "features_dc": [self.features_dc],
            "features_rest": [self.features_rest],
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats]
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
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    @property
    def get_colors(self):
        color = self.colors
        shs_rest = self.shs_rest
        return torch.cat((color[:,None,:], shs_rest), dim=1)

    def get_background_color(self):
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        return background

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
        assert camera.size == 1, "Only one camera at a time"

        if self.training:
            # currently relies on the branch vickie/camera-grads
            self.camera_optimizer.apply_to_camera(camera)

        background = self.get_background_color()

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = background.repeat(int(camera.height.item()), int(camera.width.item()), 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation}
        else:
            crop_ids = torch.ones_like(self.means[:, 0], dtype=torch.bool)

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
        W, H = int(camera.width.item()), int(camera.height.item())
        if self.training:
            self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)

        opacities_crop = self.opacities[crop_ids]
        means_crop = self.means[crop_ids]
        features_dc_crop = self.features_dc[crop_ids]
        features_rest_crop = self.features_rest[crop_ids]
        scales_crop = self.scales[crop_ids]
        quats_crop = self.quats[crop_ids]

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
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
            BLOCK_WIDTH,
        )  # type: ignore
        if radii.sum() == 0:
            rgb = background.repeat(int(camera.height.item()), int(camera.width.item()), 1)
            depth = background.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = background.new_zeros(*rgb.shape[:2], 1)
            return {"rgb": rgb, "depth": depth, "accumulation": accumulation}

        # Important to allow xys grads to populate properly
        if self.training and xys.requires_grad:
            xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        # rescale the camera back to original dimensions
        camera.rescale_output_resolution(camera_downscale)

        assert (num_tiles_hit > 0).any()  # type: ignore
        rgb, alpha = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            rgbs,
            torch.sigmoid(opacities_crop),
            H,
            W,
            BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )
        alpha = alpha[..., None]
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        depth_im = None
        if self.training:
            # Only save xys and radii if we're training
            self.xys = xys
            self.radii = radii
        if self.config.output_depth_during_training or not self.training:
            depth_im = rasterize_gaussians(  # type: ignore
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,  # type: ignore
                depths[:, None].repeat(1, 3),
                torch.sigmoid(opacities_crop),
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]  # type: ignore
            depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())

        # At the end of training, remove gaussians that are only seen by 1 camera -> keep track of count
        ending_step = self.early_stop_at_step or self.config.max_iterations
        if self.training and self.config.remove_gaussians_min_cameras_in_fov > 0 and self.step >= ending_step - self.num_train_data:
            if self.step == ending_step - self.num_train_data:
                self.gaussians_camera_cnt = torch.nn.Parameter(torch.zeros(self.num_points, 1, device=self.device))
            assert self.num_points == len(self.gaussians_camera_cnt)
            with torch.no_grad():
                self.gaussians_camera_cnt[radii > 0] += 1

        return {"rgb": rgb, "depth": depth_im, "accumulation": alpha}  # type: ignore

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

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            gt_img = TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            gt_img = image
        return gt_img.to(self.device)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.get_gt_img(batch["image"])
        predicted_rgb = outputs["rgb"]
        metrics_dict = {
            "psnr": self.psnr(predicted_rgb, gt_rgb),
            "gaussian_count": self.num_points,
            'avg_loss': torch.Tensor([self.avg_loss]),
            'min_avg_loss': torch.Tensor([self.min_avg_loss]),
            'loss_diff': torch.Tensor([self.prev_avg_loss - self.min_avg_loss]),
        }
        self.camera_optimizer.get_metrics_dict(metrics_dict)

        return metrics_dict

    def get_depth_loss(self, depths):
        if self.config.dist2cam_loss_lambda == 0.0:
            return 0.0
        return torch.abs(self.config.dist2cam_loss_size - depths).mean() / self.config.dist2cam_loss_size

    def get_sh_regularization_loss(self, sh_coeffs):
        if self.config.sh_degree <= 0 or self.config.regularize_sh_lambda <= 0.0:
            return 0.0
        n_gauss = sh_coeffs.size()[0]
        if sh_coeffs.size()[1] <= 1 or n_gauss <= 0:
            return 0.0
        return torch.linalg.vector_norm(sh_coeffs[:, 1:, :], dim=None) / n_gauss

    def get_outer_sphere_loss(self, means):
        if self.config.outside_outer_sphere_lambda == 0.0:
            return 0.0
        dists_out_of_sphere = torch.linalg.vector_norm(means, dim=1) - self.outer_sphere_rad
        error = torch.clamp(dists_out_of_sphere, min=0)  # Only penalize if gaussians are outside the sphere
        error2 = torch.square(error)
        return error2.mean()

    def get_under_hemisphere_loss(self, means):
        if self.config.under_hemisphere_lambda == 0.0 or not self.config.init_pts_hemisphere:
            return 0.0
        error = torch.clamp(self.outer_sphere_z_low - means[:, 2], min=0)  # Only penalize if gaussians are below z_low
        error2 = torch.square(error)
        return error2.mean()

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.get_gt_img(batch["image"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            assert batch["mask"].shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            mask = batch["mask"].to(self.device)
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        main_loss = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss

        avg_loss_decay = 1 - 1 / (2 * self.num_train_data)
        self.avg_loss = self.avg_loss * avg_loss_decay + main_loss.item() * (1 - avg_loss_decay)
        if self.avg_loss < self.min_avg_loss:
            self.min_avg_loss = self.avg_loss

        losses = {
            "main": main_loss,
        }

        if self.config.use_scale_regularization:
            scale_exp = torch.exp(self.scales)
            losses["scale_reg"] = (
                        torch.maximum(
                            scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                            torch.tensor(self.config.max_gauss_ratio)
                        )
                        - self.config.max_gauss_ratio
                ).mean() * 0.1
        if self.config.opacity_lambda > 0.0:
            losses["opacity"] = self.opacities.mean() * self.config.opacity_lambda * torch.sigmoid(0.01*(self.step - torch.tensor(1000))) * (1-torch.sigmoid(0.01*(self.step - torch.tensor(self.config.min_iterations))))
        if self.config.opacity_binarization_lambda > 0.0:
            losses["binarize_opacity"] = torch.square(torch.square(1/(3*math.sqrt(2))*(self.opacities+6)-2/math.sqrt(2)) - 1/2).mean() * self.config.opacity_binarization_lambda * torch.sigmoid(0.01*(self.step - torch.tensor(self.config.min_iterations)))

        return losses

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
        gt_rgb = self.get_gt_img(batch["image"])
        d = self._get_downscale_factor()
        if d > 1:
            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            predicted_rgb = outputs["rgb"]

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
