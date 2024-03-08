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
Script for exporting NeRF into other formats.
"""


from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import mediapy
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import collect_camera_poses, generate_point_cloud, get_mesh_from_filename, export_frame_render
from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.process_data import colmap_utils
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.models.visiofacto import VisiofactoModel

# import seaborn as sns
# import matplotlib.pyplot as plt

@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""


def validate_pipeline(normal_method: str, normal_output_name: str, pipeline: Pipeline) -> None:
    """Check that the pipeline is valid for this exporter.

    Args:
        normal_method: Method to estimate normals with. Either "open3d" or "model_output".
        normal_output_name: Name of the normal output.
        pipeline: Pipeline to evaluate with.
    """
    if normal_method == "model_output":
        CONSOLE.print("Checking that the pipeline has a normal output.")
        origins = torch.zeros((1, 3), device=pipeline.device)
        directions = torch.ones_like(origins)
        pixel_area = torch.ones_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        ray_bundle = RayBundle(
            origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
        )
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print(
                "[bold yellow]Warning: Please train a model with normals "
                "(e.g., nerfacto with predicted normals turned on)."
            )
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Optional[Tuple[float, float, float]] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Optional[Tuple[float, float, float]] = (1, 1, 1)
    """Maximum of the bounding box, used if use_bounding_box is True."""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = False
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(pipeline.datamanager, (VanillaDataManager, ParallelDataManager))
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        if self.save_world_frame:
            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points
            poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            points = poses[:, :3, 3].numpy()
            pcd.points = o3d.utility.Vector3dVector(points)

        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


@dataclass
class ExportTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale the images starting from the resolution used for training."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [128, 128, 128])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    texture_method: Literal["tsdf", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
        )

        # possibly
        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the tsdf export
            mesh = get_mesh_from_filename(
                str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportPoissonMesh(Exporter):
    """
    Export a mesh using poisson surface reconstruction.
    """

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    save_point_cloud: bool = False
    """Whether to save the point cloud."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    texture_method: Literal["point_cloud", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'point_cloud' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(pipeline.datamanager, (VanillaDataManager, ParallelDataManager))
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        else:
            crop_obb = None

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()
        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")

        if self.save_point_cloud:
            CONSOLE.print("Saving Point Cloud...")
            o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

        CONSOLE.print("Computing Mesh... this may take a while.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

        CONSOLE.print("Saving Mesh...")
        o3d.io.write_triangle_mesh(str(self.output_dir / "poisson_mesh.ply"), mesh)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")

        # This will texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the poisson reconstruction
            mesh = get_mesh_from_filename(
                str(self.output_dir / "poisson_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """Export a mesh using marching cubes."""

    isosurface_threshold: float = 0.0
    """The isosurface threshold for extraction. For SDF based methods the surface is the zero level set."""
    resolution: int = 1024
    """Marching cube resolution."""
    simplify_mesh: bool = False
    """Whether to simplify the mesh."""
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Minimum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Maximum of the bounding box."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Main function."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        # TODO: Make this work with Density Field
        assert hasattr(pipeline.model.config, "sdf_field"), "Model must have an SDF field."

        CONSOLE.print("Extracting mesh with marching cubes... which may take a while")

        assert self.resolution % 512 == 0, f"""resolution must be divisible by 512, got {self.resolution}.
        This is important because the algorithm uses a multi-resolution approach
        to evaluate the SDF where the minimum resolution is 512."""

        # Extract mesh using marching cubes for sdf at a multi-scale resolution.
        multi_res_mesh = generate_mesh_with_multires_marching_cubes(
            geometry_callable_field=lambda x: cast(SDFField, pipeline.model.field)
            .forward_geonetwork(x)[:, 0]
            .contiguous(),
            resolution=self.resolution,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            isosurface_threshold=self.isosurface_threshold,
            coarse_mask=None,
        )
        filename = self.output_dir / "sdf_marching_cubes_mesh.ply"
        multi_res_mesh.export(filename)

        # load the mesh from the marching cubes export
        mesh = get_mesh_from_filename(str(filename), target_num_faces=self.target_num_faces)
        CONSOLE.print("Texturing mesh with NeRF...")
        texture_utils.export_textured_mesh(
            mesh,
            pipeline,
            self.output_dir,
            px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
            unwrap_method=self.unwrap_method,
            num_pixels_per_side=self.num_pixels_per_side,
        )


@dataclass
class ExportCameraPoses(Exporter):
    """
    Export camera poses to a .json file.
    """

    def main(self) -> None:
        """Export camera poses"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        assert isinstance(pipeline, VanillaPipeline)
        train_frames, eval_frames = collect_camera_poses(pipeline)

        for file_name, frames in [("transforms_train.json", train_frames), ("transforms_eval.json", eval_frames)]:
            if len(frames) == 0:
                CONSOLE.print(f"[bold yellow]No frames found for {file_name}. Skipping.")
                continue

            output_file_path = os.path.join(self.output_dir, file_name)

            with open(output_file_path, "w", encoding="UTF-8") as f:
                json.dump(frames, f, indent=4)

            CONSOLE.print(f"[bold green]:white_check_mark: Saved poses to {output_file_path}")

@dataclass
class ExportImages(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    output_dir: Optional[Path] = None
    load_step: Optional[int] = None
    transform_to_colmap_coordinates: bool = False
    as_training: bool = False

    thumbnail_size: int = 512
    thumbnail_fov: float = 60.0  # horizontal field-of-view in degrees

    def main(self) -> None:
        if self.output_dir is None:
            self.output_dir = self.load_config.parent / 'images'
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        def update_config_callback(config: TrainerConfig):
            assert isinstance(config.pipeline.datamanager, FullImageDatamanagerConfig)
            assert isinstance(config.pipeline.datamanager.dataparser, ColmapDataParserConfig)
            config.pipeline.datamanager.dataparser.load_3D_points = False
            config.pipeline.datamanager.cache_images = 'no-cache'
            config.load_step = self.load_step
            return config

        _, pipeline, _, step = eval_setup(self.load_config,
                                          update_config_callback=update_config_callback)

        assert isinstance(pipeline.datamanager, FullImageDatamanager)

        model = pipeline.model

        if self.as_training:
            model.train(True)

        model.step = step

        # sns.histplot(torch.sigmoid(model.opacities).detach().cpu().numpy())
        # plt.savefig(self.output_dir / 'opacity_hist.png')

        with torch.no_grad():
            for camera, filename in zip(pipeline.datamanager.train_dataset.cameras, pipeline.datamanager.train_dataset.image_filenames):
                output = model.get_outputs(camera.reshape((1,)).to(model.device))['rgb'].cpu()
                mediapy.write_image(self.output_dir / filename.name, output)
                CONSOLE.print(f'Wrote {self.output_dir / filename.name}')


@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """
    output_dir: Optional[Path] = None
    load_step: Optional[int] = None

    thumbnail_size: int = 512
    thumbnail_fov: float = 60.0  # horizontal field-of-view in degrees

    transform_to_colmap_coordinates: bool = False

    def main(self) -> None:
        if self.output_dir is None:
            self.output_dir = self.load_config.parent
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        def update_config_callback(config: TrainerConfig):
            assert isinstance(config.pipeline.datamanager, FullImageDatamanagerConfig)
            assert isinstance(config.pipeline.datamanager.dataparser, ColmapDataParserConfig)
            config.pipeline.datamanager.dataparser.load_3D_points = False
            config.pipeline.datamanager.cache_images = 'no-cache'
            config.load_step = self.load_step
            return config

        _, pipeline, _, step = eval_setup(self.load_config,
                                          update_config_callback=update_config_callback)

        assert isinstance(pipeline.model, SplatfactoModel) or isinstance(pipeline.model, VisiofactoModel)

        model: Union[SplatfactoModel, VisiofactoModel] = pipeline.model

        filename = self.output_dir / "splat.ply"

        map_to_tensors = {}

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            n = positions.shape[0]
            map_to_tensors["positions"] = positions
            map_to_tensors["normals"] = np.zeros_like(positions, dtype=np.float32)

            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
            else:
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                map_to_tensors["colors"] = (colors * 255).astype(np.uint8)

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < n:
            CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select, :]

        o3d.t.io.write_point_cloud(str(filename), o3d.t.geometry.PointCloud(map_to_tensors))
        CONSOLE.print(f'Wrote {filename}')

        export_frame_render(pipeline, self.output_dir / "render.png")

        frames_json = self.get_frames_json(pipeline.datamanager)
        with open(self.output_dir / "frames.json", "w") as f:
            json.dump(frames_json, f)
        CONSOLE.print(f'Wrote {self.output_dir / "frames.json"}')

        initial_camera_transform = self.get_initial_camera_transform(pipeline.datamanager)

        scale_transform = np.eye(4)
        scale_transform[:3, :3] *= pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        dataparser_transform = np.vstack([
            pipeline.datamanager.train_dataparser_outputs.dataparser_transform.numpy(),
            [0, 0, 0, 1],
        ])
        # dataparser_transform is applied before scaling
        input_transform = scale_transform @ dataparser_transform

        with open(self.output_dir / "splat_info.json", "w") as f:
            json.dump({
                # Camera pose of the first image in the dataset, as a column-major 4x4 matrix.
                'initialCameraTransform': initial_camera_transform.ravel('F').tolist(),
                # Transformation matrix applied to colmap poses to convert them to the same
                # coordinate system as the output splats.
                'inputTransform': input_transform.ravel('F').tolist(),
                'steps': step + 1,
                'numberOfSplats': positions.shape[0],
            }, f)
        CONSOLE.print(f'Wrote {self.output_dir / "splat_info.json"}')

        if self.transform_to_colmap_coordinates:
            self.write_transformed_ply(
                data=map_to_tensors, positions=positions, scales=scales,
                rotation_transform=dataparser_transform,
                position_transform=input_transform,
                scale_transform=pipeline.datamanager.train_dataparser_outputs.dataparser_scale,
            )

    def write_transformed_ply(
            self,
            data,
            positions: np.ndarray,
            scales: np.ndarray,
            rotation_transform: np.ndarray,
            position_transform: np.ndarray,
            scale_transform: float,
    ):
        filename = self.output_dir / 'transformed.ply'
        transformed_positions = np.linalg.inv(position_transform) @ np.concatenate([
            positions,
            np.ones((positions.shape[0], 1)),
        ], axis=1).T
        data['positions'] = transformed_positions[0:3].T.astype('float32')

        transformed_scales = torch.sigmoid(torch.from_numpy(scales))
        transformed_scales = (transformed_scales / scale_transform).T
        transformed_scales = torch.logit(transformed_scales).numpy()
        data['scale_0'] = transformed_scales[0:1].T
        data['scale_1'] = transformed_scales[1:2].T
        data['scale_2'] = transformed_scales[2:3].T
        rotation_transform = Quaternion(matrix=np.linalg.inv(rotation_transform))
        for i in range(len(data['rot_0'])):
            quat = Quaternion(data['rot_0'][i], data['rot_1'][i], data['rot_2'][i], data['rot_3'][i])
            quat = rotation_transform * quat
            data['rot_0'][i] = quat[0]
            data['rot_1'][i] = quat[1]
            data['rot_2'][i] = quat[2]
            data['rot_3'][i] = quat[3]

        o3d.t.io.write_point_cloud(str(filename), o3d.t.geometry.PointCloud(data))
        CONSOLE.print(f'Wrote {filename}')

    def get_frames_json(self, datamanager: FullImageDatamanager):
        frames = []
        for i, path in enumerate(datamanager.train_dataset.image_filenames):
            transform = np.vstack([
                datamanager.train_dataset.cameras[i].camera_to_worlds.numpy(),
                [0, 0, 0, 1],
            ])
            frames.append({
                'name': path.name,
                # Column-major 4x4 camera-to-worlds transform matrix for the camera pose.
                # This matches the frames.json that we write out for object capture.
                'transform': transform.ravel('F').tolist(),
            })
        frames = sorted(frames, key=lambda f: f['name'])

        distorted_colmap_path = datamanager.config.data / 'distorted_model'
        if distorted_colmap_path.exists():
            camera_id_to_camera = colmap_utils.read_cameras_binary(distorted_colmap_path / 'cameras.bin')
            image_id_to_image = colmap_utils.read_images_binary(distorted_colmap_path / 'images.bin')

            for image_id, image in image_id_to_image.items():
                try:
                    frame = next(f for f in frames if f['name'] == image.name)
                except StopIteration:
                    print("Skipping", image.name)
                    continue
                camera = camera_id_to_camera[image.camera_id]
                frame['params'] = colmap_utils.parse_colmap_camera_params(camera)
                frame['params']['fx'] = frame['params'].pop('fl_x')
                frame['params']['fy'] = frame['params'].pop('fl_y')
                # For camera model OPENCV, this writes out w, h, fx, fy, cx, cy, k1, k2, p1, p2.
        else:
            CONSOLE.print(f'No distorted model found at {distorted_colmap_path}. '+
                          'Not writing distortion params to frames.json.')

        return frames

    def get_initial_camera_transform(self, datamanager: FullImageDatamanager) -> np.array:
        assert isinstance(datamanager.train_dataset, InputDataset)
        first_few_image_idx = sorted(
            range(len(datamanager.train_dataset.image_filenames)),
            key=lambda i: datamanager.train_dataset.image_filenames[i],
        )[:10]
        cameras = [datamanager.train_dataset.cameras[i] for i in first_few_image_idx]
        camera_transforms = [camera.camera_to_worlds.numpy() for camera in cameras]

        # We use --center_method="focus", so the origin of the coordinate system is the focus.
        focus = np.zeros(3)
        # Average distance from each camera to the focus point.
        avg_distance = np.mean([np.linalg.norm(t[:3, 3] - focus) for t in camera_transforms])
        # Back up each camera by 0.25x the average distance.
        new_positions = [t @ np.array([0, 0, avg_distance * 0.25, 1]) for t in camera_transforms]
        # Average the new positions together, maintaining distance to the focus.
        new_position = self.average_position(new_positions, focus)

        # Point the camera in the same direction as the average camera, while aligning the up direction with the world.
        # (Z-axis points behind the camera in the camera's coordinate frame and up in the world's coordinate frame.)
        new_z = np.mean([t[:3, 2] for t in camera_transforms], axis=0)
        new_z /= np.linalg.norm(new_z)
        new_x = np.cross(np.array([0, 0, 1]), new_z)
        new_x /= np.linalg.norm(new_x)
        new_y = np.cross(new_z, new_x)

        new_transform = np.vstack([
            np.stack([new_x, new_y, new_z, new_position], axis=1),
            np.array([0, 0, 0, 1]),
        ])

        return new_transform

    def average_position(self, positions: np.array, focus: np.array) -> np.array:
        avg_distance = np.mean([np.linalg.norm(p - focus) for p in positions])
        avg_direction = np.mean([p - focus for p in positions], axis=0)
        avg_direction /= np.linalg.norm(avg_direction)
        return focus + avg_direction * avg_distance



Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
        Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="cameras")],
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="gaussian-splat")],
        Annotated[ExportImages, tyro.conf.subcommand(name="images")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
