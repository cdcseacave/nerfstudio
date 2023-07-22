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
Implementation of Instant NGP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import nerfacc
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    orientation_loss,
)
from nerfstudio.model_components.ray_samplers import UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps


@dataclass
class InstantNGPModelConfig(ModelConfig):
    """Instant NGP Model Config"""

    _target: Type = field(
        default_factory=lambda: NGPModel
    )  # We can't write `NGPModel` directly, because `NGPModel` doesn't exist yet
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    grid_levels: int = 4
    """Levels of the grid used for the field."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    num_samples: int = 64
    """Number of samples per ray."""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    use_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    background_color: Literal["random", "black", "white"] = "random"
    """The color that is given to untrained areas."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    estimate_normals: bool = False
    """Whether to predict normals or not."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""


class NGPModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """

    config: InstantNGPModelConfig
    field: NerfactoField

    def __init__(self, config: InstantNGPModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = NerfactoField(
            aabb=self.scene_box.aabb,
            appearance_embedding_dim=0 if self.config.use_appearance_embedding else 32,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
            spatial_distortion=scene_contraction,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        self.sampler = UniformSampler()

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        return []

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None

        with torch.no_grad():
            ray_samples = self.sampler(ray_bundle, num_samples=self.config.num_samples)

        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.estimate_normals)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples
        )
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }
        if self.config.estimate_normals:
            outputs["normals"] = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        if self.training:
            if self.config.estimate_normals:
                outputs["rendered_orientation_loss"] = orientation_loss(
                    weights.detach(), field_outputs[FieldHeadNames.NORMAL], ray_bundle.directions
                )
        # print(f"get_outputs torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0)/1024/1024/1024} GB")
        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        rgb_loss = self.rgb_loss(image, outputs["rgb"])
        loss_dict = {"rgb_loss": rgb_loss}
        if self.training:
            if self.config.estimate_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim),
            "lpips": float(lpips)
        }  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        return metrics_dict, images_dict
