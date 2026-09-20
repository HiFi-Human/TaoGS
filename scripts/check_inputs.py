#!/usr/bin/env python3
"""Fail fast on TaoGS sequence and CoTracker input inconsistencies."""

import argparse
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.sequence_utils import read_sequence_transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("flow", type=Path)
    parser.add_argument("--frame-st", type=int, default=0)
    parser.add_argument("--frame-ed", type=int, default=10)
    parser.add_argument(
        "--init-mode",
        choices=("ply", "edgs"),
        default="edgs",
        help="require mesh_2w.ply when using PLY initialization",
    )
    args = parser.parse_args()

    mesh_path = args.dataset / "mesh_2w.ply"
    required_paths = [args.flow]
    if args.init_mode == "ply":
        required_paths.insert(1, mesh_path)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    frames = read_sequence_transforms(args.dataset)["frames"]
    if not frames:
        raise ValueError(f"no cameras in {args.dataset}")

    missing_images = []
    camera_ids = []
    for frame in frames:
        image_path = Path(frame["file_path"])
        if not image_path.is_absolute():
            image_path = args.dataset / image_path
        if not image_path.is_file():
            for frame_id in range(args.frame_st, args.frame_ed):
                dynamic_image_path = args.dataset / str(frame_id) / frame["file_path"]
                if not dynamic_image_path.is_file():
                    missing_images.append(str(dynamic_image_path))
                    if len(missing_images) >= 10:
                        break
        camera_ids.append(int(Path(frame["file_path"]).stem))
        if len(missing_images) >= 10:
            break
    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(f"transforms.json references missing images:\n{preview}")
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError("transforms.json contains duplicate numeric camera identifiers")

    with np.load(args.flow) as archive:
        flow = archive["arr_0"]
        if flow.ndim != 4:
            raise ValueError(f"expected a 4D flow visibility array, got {flow.shape}")
        flow_frames, flow_views = flow.shape[:2]
    if min(camera_ids) < 0 or max(camera_ids) >= flow_views:
        raise ValueError(
            f"camera IDs {min(camera_ids)}..{max(camera_ids)} exceed {flow_views} flow views"
        )
    required_transitions = max(0, args.frame_ed - 1)
    if args.frame_st < 0 or args.frame_ed <= args.frame_st:
        raise ValueError("invalid half-open frame interval")
    if flow_frames < required_transitions:
        raise ValueError(
            f"flow has {flow_frames} transitions but frame_ed={args.frame_ed} requires "
            f"at least {required_transitions}"
        )

    omitted = sorted(set(range(flow_views)) - set(camera_ids))
    print(
        f"inputs_ok cameras={len(camera_ids)} flow_shape={flow.shape} "
        f"omitted_flow_views={omitted} frames=[{args.frame_st},{args.frame_ed}) "
        f"init_mode={args.init_mode}"
    )


if __name__ == "__main__":
    main()
