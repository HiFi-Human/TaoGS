"""Camera metadata for DualGS-style frame directories."""

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


def colmap_directory(source):
    source = Path(source)
    for relative in ("colmap/sparse/0", "sparse/0"):
        directory = source / relative
        if directory.is_dir():
            return directory
    raise FileNotFoundError(f"No COLMAP calibration under {source}")


@lru_cache(maxsize=1)
def colmap_reader():
    # Loading calibration must not initialize scene's CUDA dependencies.
    path = Path(__file__).resolve().parents[1] / "scene/colmap_loader.py"
    spec = importlib.util.spec_from_file_location("taogs_colmap_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_sequence_transforms(source):
    """Read transforms.json or convert COLMAP poses in memory, without writes."""
    source = Path(source)
    transforms = source / "transforms.json"
    if transforms.is_file():
        with transforms.open() as handle:
            return json.load(handle)

    sparse = colmap_directory(source)
    reader = colmap_reader()
    if (sparse / "cameras.bin").is_file():
        cameras = reader.read_intrinsics_binary(str(sparse / "cameras.bin"))
        images = reader.read_extrinsics_binary(str(sparse / "images.bin"))
    else:
        cameras = reader.read_intrinsics_text(str(sparse / "cameras.txt"))
        images = reader.read_extrinsics_text(str(sparse / "images.txt"))
    frames = []
    for image in sorted(images.values(), key=lambda im: int(Path(im.name).stem)):
        camera = cameras[image.camera_id]
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            fx, cx, cy = camera.params
            fy = fx
        else:
            raise ValueError(f"Undistort images first: unsupported camera {camera.model}")
        if not np.allclose([cx, cy], [camera.width / 2, camera.height / 2], atol=1e-4):
            raise ValueError("Centered principal points are required")
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = image.qvec2rotmat()
        world_to_camera[:3, 3] = image.tvec
        frames.append(dict(
            file_path=Path(image.name).name, w=camera.width, h=camera.height,
            fl_x=float(fx), fl_y=float(fy), cx=float(cx), cy=float(cy),
            transform_matrix=(np.linalg.inv(world_to_camera) @
                              np.diag([1, -1, -1, 1])).tolist(),
        ))
    if not frames:
        raise ValueError(f"No calibrated cameras in {sparse}")
    return dict(frames=frames)
