# !/usr/bin/env python
"""
Script to run colmap on a dataset.

Usage:
    pip install -e path/to/nerfstudio  # Reinstall nerfstudio to add this script to PATH
    ns-colmap --path path/to/dataset [--sequential] [--no-densify] [--dense-resolution=1024]
"""
import json
from pathlib import Path
import shutil
import subprocess
from typing import Dict, Union

import tyro

# Consider colmap successful if it manages to register more than this fraction of input images
MIN_SUCCESS_FRACTION = 0.75

# Allow registering an image if it has at least this number of features that are already in the model
INITIAL_MIN_INLIERS = 28

# If the first try with INITIAL_MIN_INLIERS fails, decrease the value of min_inliers by
# this amount and try again.
MIN_INLIERS_DECREASE = 5

# If the first try with INITIAL_MIN_INLIERS fails, try again with a lower number of inliers
# until we reach this minimum
MIN_MIN_INLIERS = 15

def prepare_images(
    path: Path,
    sequential: bool = False,
    densify: bool = True,
    dense_resolution: int = 1024,
    clean: bool = False,
    gpu_index: int = 0,
):
    if clean:
        for item in path.iterdir():
            if item.name in {'input', 'keyframes'}:
                continue
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()

    (path / 'distorted' / 'sparse').mkdir(exist_ok=True, parents=True)
    database_path = path / 'distorted' / 'database.db'

    print('Extracting features')
    run_colmap('feature_extractor', {
        'database_path': database_path,
        'image_path': path / 'input',
        'ImageReader.single_camera': 1,
        'ImageReader.camera_model': 'OPENCV',
        'SiftExtraction.use_gpu': 1,
        'SiftExtraction.gpu_index': gpu_index,
    })

    print('Matching features')
    if sequential:
        run_colmap('sequential_matcher', {
            'database_path': database_path,
            'SiftMatching.use_gpu': 1,
            'SiftMatching.gpu_index': gpu_index,
            'SequentialMatching.loop_detection': 1,
            'SequentialMatching.vocab_tree_path': get_vocab_tree(),
        })
    else:
        image_count = len(list((path / 'input').iterdir()))
        if image_count > 150:
            run_colmap('vocab_tree_matcher', {
                'database_path': database_path,
                'SiftMatching.use_gpu': 1,
                'SiftMatching.gpu_index': gpu_index,
                'VocabTreeMatching.vocab_tree_path': get_vocab_tree(),
            })
        else:
            run_colmap('exhaustive_matcher', {
                'database_path': database_path,
                'SiftMatching.use_gpu': 1,
                'SiftMatching.gpu_index': gpu_index,
            })

    distorted_model = build_distorted_model(path)

    print('Aligning images')
    try:
        ok = run_colmap('model_aligner', {
            'input_path': distorted_model,
            'database_path': database_path,
            'ref_is_gps': 1,
            'alignment_max_error': 2.0,
            'alignment_type': 'ecef', # Earth-Centered, Earth-Fixed is a Cartesian coordinate system commonly used in geodesy, where the origin (0,0,0) is located at the center of the Earth, and the axes are fixed with respect to the Earth.
            'output_path': distorted_model,
            'transform_path': path / 'ecef_transform.txt',
        }).returncode
        if ok != 0:
            print("Model aligner with GPS failed; trying without GPS.")
            run_colmap('model_aligner', {
                'input_path': distorted_model,
                'database_path': database_path,
                'ref_is_gps': 0,
                'alignment_max_error': 2.0, # not used for plane alignment
                'alignment_type': 'plane',
                'output_path': distorted_model,
            })
    except subprocess.CalledProcessError as e:
        print(f"Model aligner failed: {e.output}")

    (path / 'distorted_model').symlink_to(distorted_model.relative_to(path),
                                          target_is_directory=True)

    print('Undistorting images')
    run_colmap('image_undistorter', {
        'image_path': path / 'input',
        'input_path': distorted_model,
        'output_path': path,
        'output_type': 'COLMAP',
    })

    if densify:
        print('Importing undistorted images into OpenMVS')
        subprocess.run(
            ['InterfaceCOLMAP', '-i', '.'],
            cwd=path,
            check=True,
        )
        print('Densifying point cloud')
        subprocess.run(
            [
                'DensifyPointCloud',
                '--crop-to-roi=0',
                '--resolution-level=0',
                '--max-resolution',
                str(dense_resolution),
                'scene.mvs',
                '-o',
                path / 'dense.ply',
            ],
            cwd=path,
            check=True,
        )

    # Move /sparse to /sparse/0
    (path / 'sparse').rename(path / 'sparse0')
    (path / 'sparse').mkdir()
    (path / 'sparse0').rename(path / 'sparse' / '0')
    print('Done')


def build_distorted_model(path: Path, min_inliers: int = INITIAL_MIN_INLIERS,
                          refine_distortion_separately: bool = False):
    for subdir in (path / 'distorted' / 'sparse').iterdir():
        if subdir.is_dir():
            shutil.rmtree(subdir)

    print('Building distorted model')
    run_colmap('mapper', {
        'database_path': path / 'distorted' / 'database.db',
        'image_path': path / 'input',
        'output_path': path / 'distorted' / 'sparse',
        'Mapper.ba_global_function_tolerance': '0.000001',
        'Mapper.max_reg_trials': 5,  # increased from default of 3
        'Mapper.abs_pose_min_num_inliers': min_inliers,  # decreased from default of 30
        'Mapper.ba_refine_extra_params': 0 if refine_distortion_separately else 1,
    })

    submodels = sorted((path / 'distorted' / 'sparse').iterdir(), key=get_image_count_in_model,
                       reverse=True)
    merged_model = submodels[0]
    if len(submodels) > 1:
        print(f'Merging {len(submodels)} submodels')
        for submodel in submodels[1:]:
            run_colmap('model_merger', {
                'input_path1': submodel,
                'input_path2': merged_model,
                'output_path': merged_model,
            })
            # If model_merger fails, colmap copies the input_path2 model to output_path.

    input_image_count = len(list((path / 'input').iterdir()))
    model_image_count = get_image_count_in_model(merged_model)
    if model_image_count / input_image_count < MIN_SUCCESS_FRACTION:
        print(f'Only registered {model_image_count} out of {input_image_count} images.')
        if min_inliers > MIN_MIN_INLIERS or not refine_distortion_separately:
            print('Trying again with lower min_inliers and without refining distortion')
            return build_distorted_model(
                path,
                min_inliers=max(min_inliers - MIN_INLIERS_DECREASE, MIN_MIN_INLIERS),
                refine_distortion_separately=True,
            )
        else:
            raise RuntimeError('Could not register enough images.')
    else:
        print(f'Registered {model_image_count} out of {input_image_count} images.')

    if len(submodels) > 1 or refine_distortion_separately:
        print('Running one more round of bundle adjustment after merging and/or with distortion params')
        run_colmap('bundle_adjuster', {
            'input_path': merged_model,
            'output_path': merged_model,
            'BundleAdjustment.max_num_iterations': 1000,
        })

    # Save a copy of the settings used for this model for analytics
    with open(path / 'colmap_info.json', 'w') as f:
        json.dump({
            'min_inliers': min_inliers,
            'refine_distortion_separately': refine_distortion_separately,
            'input_image_count': input_image_count,
            'model_image_count': model_image_count,
            'fraction': model_image_count / input_image_count,
        }, f)

    return merged_model


def get_vocab_tree(filename: str = 'vocab_tree_flickr100K_words256K.bin') -> Path:
    path = Path.home() / '.cache' / 'nerfstudio' / filename
    path.parent.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        print('Downloading vocab tree...')
        subprocess.run(['wget', '-O', str(path), f'https://demuc.de/colmap/{filename}'],
                       check=True)
    return path


def get_image_count_in_model(model: Path) -> int:
    analyze_result = run_colmap('model_analyzer', {
        'path': model,
    }, capture_output=True)
    model_info: str = analyze_result.stdout.decode('utf-8') + analyze_result.stderr.decode('utf-8')
    line = [line for line in model_info.split('\n') if 'Registered images:' in line][0]
    return int(line.rsplit(':', 1)[1].strip())


def run_colmap(subcommand: str, options: Dict[str, Union[str, Path, int, float]],
               capture_output: bool = False) -> subprocess.CompletedProcess:
    if not shutil.which('colmap'):
        raise RuntimeError('colmap executable not found. Make sure it is installed and in your PATH.')
    args = [
        'colmap',
        subcommand,
        *[f'--{key}={value}' for key, value in options.items()],
    ]
    print(' '.join(args), flush=True)
    return subprocess.run(
        args,
        capture_output=capture_output,
        check=True,  # Throw an exception if the command fails
    )


def entrypoint():
    tyro.cli(prepare_images)


if __name__ == "__main__":
    entrypoint()
