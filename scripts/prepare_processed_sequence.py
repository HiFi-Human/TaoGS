#!/usr/bin/env python3
"""Adapt an undistorted COLMAP sequence and compute real reverse-pair visibility.

All generated inputs live in a separate output directory. Camera completion
files permit restarting preprocessing without repeating finished cameras.
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from cotracker_infer import _load_model
from aggregate_flow_masks import _mask_from_pair
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.sequence_utils import read_sequence_transforms, colmap_directory, colmap_reader


def load_transform_cameras(source):
    """Read the same OpenGL camera poses used by the training loader."""
    frames = read_sequence_transforms(source)['frames']
    cameras, images, paths = {}, [], {}
    for frame in frames:
        relative = Path(frame['file_path'])
        camera_id = int(relative.stem)
        if relative.is_absolute() or '..' in relative.parts or camera_id in cameras or camera_id < 0:
            raise ValueError(f'Invalid or duplicate camera path: {relative}')
        with_shape = cv2.imread(str(source / '0' / relative))
        if with_shape is None:
            raise FileNotFoundError(source / '0' / relative)
        h, w = with_shape.shape[:2]
        if (w, h) != (frame['w'], frame['h']):
            raise ValueError(f'Image/calibration size mismatch: {relative}')
        world_to_camera = np.linalg.inv(np.asarray(frame['transform_matrix']) @ np.diag([1,-1,-1,1]))
        cameras[camera_id] = SimpleNamespace(model='PINHOLE', width=w, height=h,
            params=[frame['fl_x'], frame['fl_y'], frame['cx'], frame['cy']])
        images.append(SimpleNamespace(name=relative.name, camera_id=camera_id,
                                      world_to_camera=world_to_camera))
        paths[camera_id] = relative
    if len(images) < 2:
        raise ValueError('At least two calibrated views are required')
    return cameras, sorted(images, key=lambda image: image.camera_id), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cotracker-root', type=Path, required=True)
    parser.add_argument('--frame-ed', type=int, help='Exclusive end frame')
    parser.add_argument('--input-format', choices=['auto', 'colmap', 'transforms'], default='auto')
    parser.add_argument('--camera-selection', type=Path,
                        help='Use only numeric camera IDs in this transforms JSON; calibration still comes from the selected input format')
    args = parser.parse_args()
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    processed_colmap = (args.input_format != 'transforms'
                        and not (args.source / 'transforms.json').exists()
                        and (args.source / 'manifest.json').exists())
    if not processed_colmap:
        if args.frame_ed is None:
            parser.error('--frame-ed is required for a frame-directory sequence')
        end = args.frame_ed
        cameras, images, source_paths = load_transform_cameras(args.source)
    else:
        source_manifest = json.loads((args.source / 'manifest.json').read_text())
        if source_manifest['status'] != 'complete' or source_manifest['start'] != 0:
            raise ValueError('Expected a completed sequence starting at frame zero')
        available_end = int(source_manifest['end'])
        if source_manifest['frames'] != available_end:
            raise ValueError('Noncontiguous sequence')
        end = args.frame_ed if args.frame_ed is not None else available_end
        if end > available_end:
            raise ValueError('Requested frames exceed source manifest')
        loader_path = Path(__file__).resolve().parents[1] / 'scene/colmap_loader.py'
        spec = importlib.util.spec_from_file_location('colmap_loader_standalone', loader_path)
        colmap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(colmap)
        sparse = args.source / 'sparse/0'
        cameras = colmap.read_intrinsics_binary(str(sparse / 'cameras.bin'))
        images = sorted(colmap.read_extrinsics_binary(str(sparse / 'images.bin')).values(),
                        key=lambda image: int(Path(image.name).stem))
        source_paths = {int(Path(im.name).stem): Path('image_undistortion_white/images') / im.name
                        for im in images}
    if end < 2:
        raise ValueError('At least two frames are required')
    if args.camera_selection is not None:
        selected_frames = json.loads(args.camera_selection.read_text())['frames']
        selected_ids = [int(Path(frame['file_path']).stem) for frame in selected_frames]
        if len(selected_ids) != len(set(selected_ids)) or len(selected_ids) < 2:
            raise ValueError('Camera selection must contain at least two unique numeric IDs')
        available_ids = {int(Path(image.name).stem) for image in images}
        if not set(selected_ids).issubset(available_ids):
            raise ValueError(f'Selected cameras missing calibration: {set(selected_ids) - available_ids}')
        images = [image for image in images if int(Path(image.name).stem) in set(selected_ids)]
    width, height = 1920, 1080
    frames = []
    for image in images:
        camera = cameras[image.camera_id]
        if camera.model != 'PINHOLE':
            raise ValueError(f'Expected undistorted PINHOLE camera: {camera}')
        fx, fy, cx, cy = camera.params
        if not np.allclose([cx, cy], [camera.width / 2, camera.height / 2], atol=1e-4):
            raise ValueError(f'Noncentered principal point: {camera}')
        if not processed_colmap:
            world_to_camera = image.world_to_camera
        else:
            world_to_camera = np.eye(4)
            world_to_camera[:3, :3] = image.qvec2rotmat()
            world_to_camera[:3, 3] = image.tvec
        frames.append(dict(file_path=image.name, w=width, h=height,
                           fl_x=float(fx * width / camera.width),
                           fl_y=float(fy * height / camera.height), cx=width/2, cy=height/2,
                           transform_matrix=(np.linalg.inv(world_to_camera) @
                                             np.diag([1, -1, -1, 1])).tolist()))
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = args.output / 'image_white'
    dataset.mkdir(exist_ok=True)
    metadata = dict(source=str(args.source.resolve()), frame_interval=[0, end],
                    camera_ids=[int(Path(im.name).stem) for im in images],
                    image_size=[width, height], grid_size=81, reverse_pairs=True,
                    checkpoint=str(args.checkpoint.resolve()),
                    output_layout='image_white/frame/camera.png',
                    source_paths={str(k): str(v) for k, v in source_paths.items()},
                    foreground_mask='any RGB channel < 250, identical to training')
    if args.camera_selection is not None:
        metadata['camera_selection'] = str(args.camera_selection.resolve())
    manifest_path = args.output / 'preprocess_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != metadata:
        raise ValueError('Existing preprocessing metadata differs')
    manifest_path.write_text(json.dumps(metadata, indent=2) + '\n')
    (dataset / 'transforms.json').write_text(json.dumps(dict(frames=frames), indent=2) + '\n')
    # Supply the actual calibration cloud, avoiding random fallback/source writes.
    if (args.source / 'points3d.ply').exists():
        shutil.copy2(args.source / 'points3d.ply', dataset / 'points3d.ply')
    elif not (args.source / 'transforms.json').exists():
        colmap = colmap_reader()
        sparse = colmap_directory(args.source)
        if (sparse / 'points3D.bin').exists():
            xyz, rgb, _ = colmap.read_points3D_binary(str(sparse / 'points3D.bin'))
        else:
            xyz, rgb, _ = colmap.read_points3D_text(str(sparse / 'points3D.txt'))
        cloud = np.zeros(len(xyz), dtype=[('x','f4'),('y','f4'),('z','f4'),
                                        ('nx','f4'),('ny','f4'),('nz','f4'),
                                        ('red','u1'),('green','u1'),('blue','u1')])
        for index, name in enumerate(('x', 'y', 'z')):
            cloud[name] = xyz[:, index]
        for index, name in enumerate(('red', 'green', 'blue')):
            cloud[name] = rgb[:, index]
        PlyData([PlyElement.describe(cloud, 'vertex')]).write(str(dataset / 'points3d.ply'))
    else:
        raise FileNotFoundError(args.source / 'points3d.ply')
    for frame in range(end):
        (dataset / str(frame)).mkdir(parents=True, exist_ok=True)
    visibility_root = args.output / 'camera_visibility'
    visibility_root.mkdir(exist_ok=True)
    print(f'Preparing {len(images)} cameras, frames [0,{end}), output={args.output}', flush=True)
    model = _load_model(SimpleNamespace(cotracker_root=args.cotracker_root,
                        checkpoint=args.checkpoint, offline=True, use_v2_model=False),
                        torch.device('cuda'))
    for image in images:
        camera_id = int(Path(image.name).stem)
        completed = visibility_root / f'{camera_id}.npy'
        if completed.exists():
            saved = np.load(completed, mmap_mode='r')
            if saved.shape != (end - 1, 81, 144) or saved.dtype != np.uint8:
                raise ValueError(f'Invalid completed camera: {completed}')
            print(f'Camera {camera_id}: already complete', flush=True)
            continue
        camera = cameras[image.camera_id]
        masks = np.empty((end - 1, 81, 144), dtype=np.uint8)
        previous = None
        for frame in range(end):
            source_image = args.source / str(frame) / source_paths[camera_id]
            bgr = cv2.imread(str(source_image))
            if bgr is None or bgr.shape[:2] != (camera.height, camera.width):
                raise ValueError(f'Missing image or calibration size mismatch: {source_image}')
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LANCZOS4)
            target = dataset / str(frame) / image.name
            if not cv2.imwrite(str(target), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f'Could not write {target}')
            current = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if previous is not None:
                pair = torch.from_numpy(np.stack([current, previous])).permute(0,3,1,2)[None].float().cuda()
                foreground = np.any(current < 250, axis=-1).astype(np.float32) * 255
                mask = torch.from_numpy(foreground)[None, None].cuda()
                if foreground.any():
                    with torch.inference_mode():
                        tracks, visible = model(pair, grid_size=81, grid_query_frame=0,
                                                backward_tracking=False, segm_mask=mask)
                    masks[frame - 1] = _mask_from_pair(
                        tracks.cpu().numpy(), visible.cpu().numpy(),
                        mask_height=81, mask_width=144,
                        scale_x=width/144, scale_y=height/81)
                else:
                    masks[frame - 1].fill(255)
                del pair, mask
            previous = current
            if frame % 25 == 0 or frame == end - 1:
                print(f'camera={camera_id} frame={frame}/{end-1}', flush=True)
        temporary = visibility_root / f'{camera_id}.tmp.npy'
        np.save(temporary, masks)
        temporary.replace(completed)
    flow = np.full((end - 1, max(metadata['camera_ids']) + 1, 81, 144), 255, dtype=np.uint8)
    for camera_id in metadata['camera_ids']:
        flow[:, camera_id] = np.load(visibility_root / f'{camera_id}.npy')
    temporary = args.output / 'visibility.tmp.npz'
    np.savez_compressed(temporary, flow)
    temporary.replace(args.output / 'visibility.npz')
    print(f'Inputs complete: {dataset}; visibility={flow.shape}', flush=True)


if __name__ == '__main__':
    main()
