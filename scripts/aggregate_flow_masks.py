#!/usr/bin/env python3
"""Convert TaoGS CoTracker raw dictionaries into training flow masks.

The output is ``np.savez_compressed(..., flow)`` with the conventional
``arr_0`` key and shape ``[transition, camera_slot, H, W]``.  Camera slots are
indexed by the numeric dataset camera id, not by the order in
``transforms.json``. Missing camera slots are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


RAW_FILE_RE = re.compile(r"^pred_(tracks|visibility)_(inv|forward)_(\d+)\.npy$")
KEY_RE = re.compile(r"^pred_(tracks|visibility)_(\d+)_(\d+)$")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return parsed


def _transforms_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        path = path / "transforms.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_transform_camera_ids(path: Path | None) -> list[int]:
    if path is None:
        return []
    with path.open() as handle:
        payload = json.load(handle)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"no frames found in {path}")
    camera_ids: list[int] = []
    for frame in frames:
        try:
            camera_id = int(Path(str(frame["file_path"])).stem)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"transforms.json file_path must end in a numeric stem: {path}"
            ) from exc
        if camera_id < 0:
            raise ValueError(f"negative camera id in {path}: {camera_id}")
        camera_ids.append(camera_id)
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError(f"duplicate numeric camera ids in {path}")
    return sorted(camera_ids)


def _transition_ids(frame_st: int, frame_ed: int, frame_step: int, reverse: bool) -> list[int]:
    source_ids = list(range(frame_st, frame_ed, frame_step))
    if len(source_ids) < 2:
        raise ValueError("the selected half-open interval must contain at least two frames")
    if reverse:
        # A reversed sequence [end-1, ..., start] produces a key for every
        # current frame except the final previous-frame endpoint.  Sort these
        # keys because TopoHandler indexes row frame_id - 1.
        return sorted(list(reversed(source_ids))[:-1])
    return source_ids[:-1]


def _load_raw_mapping(path: Path, kind: str, raw_camera_id: int) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        mapping_obj = np.load(path, allow_pickle=True).item()
    except Exception as exc:
        raise ValueError(f"failed to load raw mapping {path}") from exc
    if not isinstance(mapping_obj, dict):
        raise ValueError(f"raw mapping is not a dictionary: {path}")

    result: dict[int, np.ndarray] = {}
    for key, value in mapping_obj.items():
        if not isinstance(key, str):
            raise ValueError(f"raw mapping contains a non-string key: {path}")
        match = KEY_RE.fullmatch(key)
        if match is None or match.group(1) != kind:
            raise ValueError(f"invalid {kind} key {key!r} in {path}")
        key_camera_id = int(match.group(2))
        frame_id = int(match.group(3))
        if key_camera_id != raw_camera_id:
            raise ValueError(
                f"filename camera {raw_camera_id} disagrees with key camera {key_camera_id}: {path}"
            )
        if frame_id in result:
            raise ValueError(f"duplicate frame key {key!r} in {path}")
        result[frame_id] = np.asarray(value)
    if not result:
        raise ValueError(f"raw mapping is empty: {path}")
    return result


def _validate_pair(
    tracks: np.ndarray,
    visibility: np.ndarray,
    *,
    raw_camera_id: int,
    frame_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    if tracks.ndim != 4 or tracks.shape[0:2] != (1, 2) or tracks.shape[-1] != 2:
        raise ValueError(
            f"camera {raw_camera_id} frame {frame_id}: expected tracks [1,2,N,2], got {tracks.shape}"
        )
    if visibility.ndim == 4 and visibility.shape[-1] == 1:
        visibility = visibility[..., 0]
    if visibility.ndim != 3 or visibility.shape[0:2] != (1, 2):
        raise ValueError(
            f"camera {raw_camera_id} frame {frame_id}: expected visibility [1,2,N], got {visibility.shape}"
        )
    if tracks.shape[2] != visibility.shape[2]:
        raise ValueError(
            f"camera {raw_camera_id} frame {frame_id}: track/visibility point count mismatch"
        )
    if not np.issubdtype(tracks.dtype, np.number):
        raise ValueError(f"camera {raw_camera_id} frame {frame_id}: tracks are not numeric")
    if not np.isfinite(tracks).all():
        raise ValueError(f"camera {raw_camera_id} frame {frame_id}: non-finite track coordinate")
    if not np.issubdtype(visibility.dtype, np.bool_) and not np.issubdtype(
        visibility.dtype, np.integer
    ):
        raise ValueError(f"camera {raw_camera_id} frame {frame_id}: visibility is not bool/integer")
    if np.issubdtype(visibility.dtype, np.integer) and not np.isin(visibility, (0, 1)).all():
        raise ValueError(f"camera {raw_camera_id} frame {frame_id}: visibility must contain only 0/1")
    return tracks.astype(np.float64, copy=False), visibility.astype(bool, copy=False)


def _mask_from_pair(
    tracks: np.ndarray,
    visibility: np.ndarray,
    *,
    mask_height: int,
    mask_width: int,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    coords = np.rint(tracks[0, 1] / np.array([scale_x, scale_y], dtype=np.float64)).astype(np.int64)
    hidden = ~visibility[0, 1]
    x = coords[:, 0]
    y = coords[:, 1]
    x = np.clip(x, 0, mask_width - 1)
    y = np.clip(y, 0, mask_height - 1)

    output = np.full((mask_height, mask_width), 255, dtype=np.uint8)
    output[y[hidden], x[hidden]] = 0
    return output


def _raw_ids(raw_dir: Path, direction: str) -> list[int]:
    ids: list[int] = []
    for path in raw_dir.glob("pred_visibility_*.npy"):
        match = RAW_FILE_RE.fullmatch(path.name)
        if match and match.group(1) == "visibility" and match.group(2) == direction:
            ids.append(int(match.group(3)))
    return sorted(set(ids))


def _write_npz(path: Path, flow: np.ndarray, overwrite: bool) -> None:
    if path.suffix != ".npz":
        raise ValueError(f"output must have a .npz suffix: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {path}; choose a new path or pass --overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    if temporary.exists():
        temporary.unlink()
    np.savez_compressed(temporary, flow)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate CoTracker raw dictionaries into TaoGS training flow masks."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--transforms-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-st", type=_nonnegative_int, default=0)
    parser.add_argument("--frame-ed", type=_positive_int, default=300, help="exclusive frame bound")
    parser.add_argument("--frame-step", type=_positive_int, default=1)
    parser.add_argument("--raw-camera-offset", type=int, default=1)
    parser.add_argument("--mask-height", type=_positive_int, default=81)
    parser.add_argument("--mask-width", type=_positive_int, default=144)
    parser.add_argument("--track-height", type=_positive_int, default=1080)
    parser.add_argument("--track-width", type=_positive_int, default=1920)
    parser.add_argument(
        "--coordinate-scale",
        type=_positive_float,
        help="use one scale for x/y instead of track-width/mask-width and track-height/mask-height",
    )
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--reverse", dest="reverse", action="store_true")
    direction.add_argument("--forward", dest="reverse", action="store_false")
    parser.set_defaults(reverse=True)
    parser.add_argument(
        "--pad-before",
        dest="pad_before",
        action="store_true",
        help="place rows at global frame_id-1 indices (default)",
    )
    parser.add_argument("--no-pad-before", dest="pad_before", action="store_false")
    parser.set_defaults(pad_before=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.raw_dir.is_dir():
        raise FileNotFoundError(args.raw_dir)
    if args.frame_ed <= args.frame_st:
        raise ValueError("expected frame-ed > frame-st")
    if args.raw_camera_offset < 0:
        raise ValueError("raw-camera-offset must be non-negative")
    metadata_path = args.output.with_suffix(".json")
    if not args.overwrite:
        if args.output.exists():
            raise FileExistsError(
                f"refusing to overwrite {args.output}; choose a new path or pass --overwrite"
            )
        if metadata_path.exists():
            raise FileExistsError(
                f"refusing to overwrite metadata {metadata_path}; pass --overwrite"
            )
    transforms_path = _transforms_path(args.transforms_json)
    transform_ids = _load_transform_camera_ids(transforms_path)
    direction = "inv" if args.reverse else "forward"
    raw_ids = _raw_ids(args.raw_dir, direction)
    if not raw_ids:
        raise ValueError(f"no {direction} visibility files found in {args.raw_dir}")

    dataset_ids = {raw_id - args.raw_camera_offset for raw_id in raw_ids}
    if min(dataset_ids) < 0:
        raise ValueError("raw camera offset maps a raw id to a negative dataset id")
    max_camera_id = max(dataset_ids)
    missing_transform_raw = [
        camera_id + args.raw_camera_offset
        for camera_id in transform_ids
        if camera_id + args.raw_camera_offset not in raw_ids
    ]
    if missing_transform_raw:
        raise ValueError(
            "raw directory does not cover all transforms cameras; missing raw ids "
            f"{missing_transform_raw[:20]}"
        )

    transition_ids = _transition_ids(args.frame_st, args.frame_ed, args.frame_step, args.reverse)
    if args.coordinate_scale is None:
        scale_x = args.track_width / args.mask_width
        scale_y = args.track_height / args.mask_height
    else:
        scale_x = scale_y = args.coordinate_scale

    if args.pad_before:
        if not args.reverse:
            raise ValueError("--pad-before follows TopoHandler's reverse-pair convention; use --no-pad-before for forward mode")
        row_for_frame = {frame_id: frame_id - 1 for frame_id in transition_ids}
        output_rows = max(row_for_frame.values()) + 1
    else:
        output_rows = len(transition_ids)
        row_for_frame = {frame_id: row_idx for row_idx, frame_id in enumerate(transition_ids)}
    flow = np.full(
        (output_rows, max_camera_id + 1, args.mask_height, args.mask_width),
        255,
        dtype=np.uint8,
    )

    for raw_camera_id in raw_ids:
        dataset_camera_id = raw_camera_id - args.raw_camera_offset
        visibility_path = args.raw_dir / f"pred_visibility_{direction}_{raw_camera_id}.npy"
        tracks_path = args.raw_dir / f"pred_tracks_{direction}_{raw_camera_id}.npy"
        visibility = _load_raw_mapping(visibility_path, "visibility", raw_camera_id)
        tracks = _load_raw_mapping(tracks_path, "tracks", raw_camera_id)
        if set(visibility) != set(tracks):
            raise ValueError(
                f"camera {raw_camera_id}: track/visibility frame keys differ "
                f"({len(tracks)} vs {len(visibility)})"
            )
        for frame_id in transition_ids:
            if frame_id not in visibility:
                raise KeyError(
                    f"camera {raw_camera_id}: missing frame key {frame_id} for "
                    f"interval [{args.frame_st}, {args.frame_ed})"
                )
            tracks_pair, visibility_pair = _validate_pair(
                tracks[frame_id],
                visibility[frame_id],
                raw_camera_id=raw_camera_id,
                frame_id=frame_id,
            )
            flow[row_for_frame[frame_id], dataset_camera_id] = _mask_from_pair(
                tracks_pair,
                visibility_pair,
                mask_height=args.mask_height,
                mask_width=args.mask_width,
                scale_x=scale_x,
                scale_y=scale_y,
            )
        print(
            f"camera raw={raw_camera_id} dataset={dataset_camera_id} "
            f"frames={len(visibility)}"
        )

    _write_npz(args.output, flow, args.overwrite)
    digest = hashlib.sha256(flow.tobytes()).hexdigest()
    metadata = {
        "format": "taogs-flow-mask-v1",
        "shape": list(flow.shape),
        "dtype": str(flow.dtype),
        "frame_interval": [args.frame_st, args.frame_ed],
        "frame_step": args.frame_step,
        "reverse": args.reverse,
        "pad_before": args.pad_before,
        "raw_camera_offset": args.raw_camera_offset,
        "raw_camera_ids": raw_ids,
        "transforms_camera_ids": transform_ids,
        "mask_size": [args.mask_height, args.mask_width],
        "track_size": [args.track_height, args.track_width],
        "coordinate_scale": [scale_y, scale_x],
        "array_sha256": digest,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved {args.output}")
    print(f"flow_shape={flow.shape} dtype={flow.dtype} array_sha256={digest}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
