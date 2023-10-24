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
from nerfstudio.data.scene_box import OrientedBox
from copy import deepcopy
from nerfstudio.cameras.rays import RayBundle
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.cameras.cameras import Cameras
from gsplat._torch_impl import quat_to_rotmat
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
import viser.transforms as vtf


from gsplat.rasterize import RasterizeGaussians
from gsplat.project_gaussians import ProjectGaussians
from gsplat.sh import SphericalHarmonics, num_sh_bases


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
    max_iterations: int = 15000
    """Sets the defaults in method_configs.py"""
    warmup_length: int = 1000
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 200
    """training starts at 1/d resolution, every n steps this is doubled"""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.01
    """threshold of opacity for culling gaussians"""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    sh_degree_interval: int = 500
    """every n intervals turn on another sh degree"""
    max_screen_size: int = 150
    """maximum screen size of a gaussian, in pixels"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_refine_at: int = 0.8 * max_iterations
    """stop splitting & culling at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    sphere_loss_size: float = 50.0
    """pull points outward: 1 corresponds to a sphere with radius equal to the average camera distance from the center"""
    sphere_loss_lambda: float = 0.05
    """pull points outward: add L1 term to loss function that has a range [0-1] -> multiply by this number"""
    regularize_sh: float = 0.0
    """sh coeffs are multiplied by x, y, z, xx, xy, xz, etc. where [x,y,z] is a unit vector (no coeff normalization needed) so all coeffs are summed together for L2 loss -> multiply by this number"""


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

    def populate_modules(self):
        if self.seed_pts is not None and not self.config.random_init:
            self.means = torch.nn.Parameter(self.seed_pts[0])  # (Location, Color)
        else:
            self.means = torch.nn.Parameter((torch.rand((100000, 3)) - 0.5) * 2)
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
            shs = torch.zeros((fused_color.shape[0], 3, dim_sh)).float().cuda()
            shs[:, :3, 0] = fused_color
            shs[:, 3:, 1:] = 0.0
            self.colors = torch.nn.Parameter(shs[:, :, 0:1])
            self.shs_rest = torch.nn.Parameter(shs[:, :, 1:])
        else:
            self.colors = torch.nn.Parameter(torch.rand(self.num_points, 3, 1))
            self.shs_rest = torch.nn.Parameter(torch.zeros((self.num_points, 3, dim_sh - 1)))

        self.opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1)))
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        self.ssim = StructuralSimilarityIndexMeasure()
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
        self.crop_box: Optional[OrientedBox] = None
        self.back_color = torch.zeros(3)

        # use camera positions to init sphere center & size
        self.cameras_loaded = False
        self.cameras = []
        self.cnt = 0

    @property
    def get_colors(self):
        color = self.colors
        shs_rest = self.shs_rest
        return torch.cat((color, shs_rest), dim=2)

    def load_state_dict(self, dict, **kwargs):
        # resize the parameters to match the new number of points
        self.step = 30000
        newp = dict["means"].shape[0]
        self.means = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.scales = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.quats = torch.nn.Parameter(torch.zeros(newp, 4, device=self.device))
        self.colors = torch.nn.Parameter(torch.zeros(self.num_points, 3, 1, device=self.device))
        self.shs_rest = torch.nn.Parameter(
            torch.zeros(self.num_points, 3, num_sh_bases(self.config.sh_degree) - 1, device=self.device)
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

            # update the max screen size
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii)
            newradii = self.radii[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(self.max_2Dsize[visible_mask], newradii)

    def set_crop(self, crop_box: OrientedBox):
        self.crop_box = crop_box

    def set_background(self, back_color: torch.Tensor):
        assert back_color.shape == (3,)
        self.back_color = back_color

    def refinement_after(self, optimizers: Optimizers, step):
        if self.step > self.config.warmup_length and self.step < self.config.stop_refine_at:
            with torch.no_grad():
                # then we densify
                avg_grad_norm = (
                    (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                )
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
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
                    deleted_mask = self.cull_gaussians()
                    param_groups = self.get_param_groups()
                    for group, param in param_groups.items():
                        self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)

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

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            culls = culls | toobigs
            # cull big screen space
            culls = culls | (self.max_2Dsize > self.config.max_screen_size).squeeze()
        self.means = Parameter(self.means[~culls].detach())
        self.scales = Parameter(self.scales[~culls].detach())
        self.quats = Parameter(self.quats[~culls].detach())
        self.colors = Parameter(self.colors[~culls].detach())
        self.shs_rest = Parameter(self.shs_rest[~culls].detach())
        self.opacities = Parameter(self.opacities[~culls].detach())

        print(f"Culled {n_bef - self.num_points} gaussians")
        return culls

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
    
    def get_camera_poses_info(self):
        self.cameras_origin = torch.zeros((1,3)).to(self.device)
        for cam in self.cameras:
            self.cameras_origin += cam.camera_to_worlds[..., :3, 3:4].resize(1,3)  # 1 x 3 x 1
        self.cameras_origin /= len(self.cameras)
        self.cameras_ave_dist_origin = torch.zeros((1)).to(self.device)
        for cam in self.cameras:
            self.cameras_ave_dist_origin += (self.cameras_origin - cam.camera_to_worlds[..., :3, 3:4].resize(1,3)).norm()
        self.cameras_ave_dist_origin /= len(self.cameras)
        
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
            self.cameras_loaded = self.cnt > 3 * len(self.cameras)
            if self.cameras_loaded:
                self.get_camera_poses_info()
            self.cnt += 1
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
            background = torch.rand(3, device=self.device)
        else:
            background = self.back_color

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
            coeffs = colors_crop.permute(0, 2, 1)[rend_mask, :n_coeffs, :]
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
        return {"rgb": rgb, "depth": depth_im, "means_rendered": self.means[rend_mask], "sh": coeffs} # means_rendered: world coords

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """

        return {}
    
    def get_sphere_loss(self, means) -> torch.Tensor:
        sphere_radius = self.config.sphere_loss_size * self.cameras_ave_dist_origin
        return torch.abs(torch.linalg.vector_norm(means - self.cameras_origin, dim=1) - sphere_radius).mean() / sphere_radius

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
        if self.config.sphere_loss_lambda > 0 and self.cameras_loaded:
            outer_sphere_L1 = self.get_sphere_loss(outputs["means_rendered"])
        else:
            outer_sphere_L1 = 0.0
        if self.config.sh_degree > 0:
            sh_coeffs = outputs["sh"]
            n = sh_coeffs.size()[0]
            assert n == outputs["means_rendered"].size()[0]
            if sh_coeffs.size()[1] > 1 and n > 0:
                sh_L2 = torch.linalg.vector_norm(sh_coeffs[:, 1:, :], dim=None) / n
            else:
                sh_L2 = 0.0
        else:
            sh_L2 = 0.0
        return {"main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss + outer_sphere_L1 * self.config.sphere_loss_lambda + sh_L2 * self.config.regularize_sh}

    @torch.no_grad()
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
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
