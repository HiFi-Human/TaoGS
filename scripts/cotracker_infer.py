#!/usr/bin/env python3
"""Run CoTracker on adjacent video frames for the TaoGS preprocessing path.

The released TaoGS training code consumes one dictionary per camera, with keys
of the form ``pred_visibility_<raw-camera>_<current-frame>`` and
``pred_tracks_<raw-camera>_<current-frame>``.  The default direction here is
the reverse-pair convention: for a window keyed by ``t`` the input
frames are ``[t, t-1]`` and the query frame is frame zero.

CoTracker itself is intentionally not vendored.  Supply a checkout with
``--cotracker-root`` (or install it as a Python package) and a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return parsed


def _load_numeric_camera_ids(transforms_json: Path) -> list[int]:
    with transforms_json.open() as handle:
        payload = json.load(handle)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"no frames found in {transforms_json}")

    camera_ids: list[int] = []
    for frame in frames:
        try:
            camera_id = int(Path(str(frame["file_path"])).stem)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "every transforms.json file_path must end in a numeric stem"
            ) from exc
        if camera_id < 0:
            raise ValueError(f"negative camera id in {transforms_json}: {camera_id}")
        camera_ids.append(camera_id)
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError(f"duplicate numeric camera ids in {transforms_json}")
    return sorted(camera_ids)


def _format_pattern(pattern: str, *, raw_camera_id: int, dataset_camera_id: int, frame_idx: int) -> str:
    """Expand both documented names and a few harmless aliases."""

    values = {
        "raw_camera_id": raw_camera_id,
        "camera_id": dataset_camera_id,
        "dataset_camera_id": dataset_camera_id,
        "frame_idx": frame_idx,
        "frame_id": frame_idx,
    }
    try:
        return pattern.format(**values)
    except KeyError as exc:
        raise ValueError(
            f"unsupported placeholder {exc.args[0]!r} in pattern {pattern!r}; "
            "use {raw_camera_id}, {dataset_camera_id}, or {frame_idx}"
        ) from exc


def _resolve_camera_ids(args: argparse.Namespace) -> list[int]:
    if args.raw_camera_ids:
        raw_ids = sorted({int(item) for item in args.raw_camera_ids.split(",") if item.strip()})
        if not raw_ids:
            raise ValueError("--raw-camera-ids did not contain an id")
        if min(raw_ids) < 0:
            raise ValueError("raw camera ids must be non-negative")
        return raw_ids

    if args.raw_camera_st is not None or args.raw_camera_ed is not None:
        if args.raw_camera_st is None or args.raw_camera_ed is None:
            raise ValueError("--raw-camera-st and --raw-camera-ed must be supplied together")
        if args.raw_camera_ed <= args.raw_camera_st:
            raise ValueError("expected raw-camera-ed > raw-camera-st")
        return list(range(args.raw_camera_st, args.raw_camera_ed))

    if args.transforms_json is None:
        raise ValueError(
            "supply --transforms-json when camera range is not given explicitly"
        )
    return [camera_id + args.raw_camera_offset for camera_id in _load_numeric_camera_ids(args.transforms_json)]


def _read_selected_frames(
    video_path: Path,
    frame_st: int,
    frame_ed: int,
    frame_step: int,
    resize_scale: float,
    reverse: bool,
) -> tuple[list[np.ndarray], list[int]]:
    """Read frames in temporal order and resize them for tracking."""

    if frame_ed <= frame_st:
        raise ValueError("expected frame-ed > frame-st")
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for CoTracker video decoding "
            "semantics; install the requirements.txt dependencies"
        ) from exc
    try:
        reader = imageio.get_reader(str(video_path))
    except Exception as exc:
        raise RuntimeError(f"failed to open video: {video_path}") from exc

    frames: list[np.ndarray] = []
    source_ids: list[int] = []
    try:
        decoded_count = 0
        for source_idx, image in enumerate(reader):
            if source_idx >= frame_ed:
                decoded_count = frame_ed
                break
            decoded_count = source_idx + 1
            if source_idx < frame_st:
                continue
            image_np = np.array(image)
            if image_np.ndim != 3 or image_np.shape[2] < 3:
                raise ValueError(f"expected a colour video, got frame {source_idx} in {video_path}")
            target_w = int(image_np.shape[1] // resize_scale)
            target_h = int(image_np.shape[0] // resize_scale)
            if target_w <= 0 or target_h <= 0:
                raise ValueError(
                    f"resize scale {resize_scale} is too large for {video_path}"
                )
            frames.append(cv2.resize(image_np, (target_w, target_h)))
            source_ids.append(source_idx)
    finally:
        reader.close()

    if decoded_count < frame_ed:
        raise ValueError(
            f"video {video_path} ended at frame {decoded_count}; "
            f"requested half-open interval [{frame_st}, {frame_ed})"
        )

    frames = frames[::frame_step]
    source_ids = source_ids[::frame_step]
    if reverse:
        frames.reverse()
        source_ids.reverse()
    if len(frames) < 2:
        raise ValueError("at least two selected frames are required")
    return frames, source_ids


def _load_mask(
    mask_dir: Path | None,
    mask_pattern: str,
    mask_fallback_pattern: str | None,
    *,
    raw_camera_id: int,
    dataset_camera_id: int,
    frame_idx: int,
    target_hw: tuple[int, int],
    allow_missing: bool,
) -> torch.Tensor | None:
    if mask_dir is None:
        return None

    patterns = [mask_pattern]
    if mask_fallback_pattern:
        patterns.append(mask_fallback_pattern)
    candidates = [
        mask_dir
        / _format_pattern(
            pattern,
            raw_camera_id=raw_camera_id,
            dataset_camera_id=dataset_camera_id,
            frame_idx=frame_idx,
        )
        for pattern in patterns
    ]
    mask_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if mask_path is None:
        if allow_missing:
            return None
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"no mask found for frame {frame_idx}; tried {tried}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"failed to decode segmentation mask: {mask_path}")
    if mask.ndim == 3:
        # A colour mask is accepted only when all channels encode the same mask.
        if not np.array_equal(mask[..., 0], mask[..., 1]) or not np.array_equal(mask[..., 0], mask[..., 2]):
            raise ValueError(f"mask has non-identical colour channels: {mask_path}")
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D segmentation mask, got {mask.shape}: {mask_path}")
    mask = np.where(mask > 0, 255, 0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask)[None, None]
    return F.interpolate(mask_tensor, size=target_hw, mode="nearest")


def _load_model(args: argparse.Namespace, device: torch.device):
    if args.cotracker_root is not None:
        root = args.cotracker_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        sys.path.insert(0, str(root))

    try:
        from cotracker.predictor import CoTrackerPredictor
    except ImportError as exc:
        raise ImportError(
            "CoTracker is not importable; install it or pass its checkout with "
            "--cotracker-root"
        ) from exc

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for reproducible preprocessing")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    window_len = 60 if args.offline else 16
    model = CoTrackerPredictor(
        checkpoint=str(checkpoint),
        v2=args.use_v2_model,
        offline=args.offline,
        window_len=window_len,
    )
    return model.to(device).eval()


def _save_mapping(path: Path, mapping: dict[str, np.ndarray], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {path}; choose a new output directory or pass --overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mapping)


def _run_camera(args: argparse.Namespace, model, device: torch.device, raw_camera_id: int) -> dict[str, object]:
    dataset_camera_id = raw_camera_id - args.raw_camera_offset
    video_path = args.video_dir / _format_pattern(
        args.video_pattern,
        raw_camera_id=raw_camera_id,
        dataset_camera_id=dataset_camera_id,
        frame_idx=args.frame_st,
    )
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    frames, source_ids = _read_selected_frames(
        video_path,
        args.frame_st,
        args.frame_ed,
        args.frame_step,
        args.resize_scale,
        args.reverse,
    )
    if len(source_ids) != len(frames):
        raise AssertionError("frame/source index bookkeeping mismatch")

    tracks: dict[str, np.ndarray] = {}
    visibility: dict[str, np.ndarray] = {}
    # For reverse pairs, the current frame is the first element in each window;
    for window_idx in range(len(frames) - 1):
        current_frame = source_ids[window_idx]
        pair = np.stack(frames[window_idx : window_idx + 2], axis=0)
        video = torch.from_numpy(pair).permute(0, 3, 1, 2)[None].float().to(device)
        mask = _load_mask(
            args.mask_dir,
            args.mask_pattern,
            args.mask_fallback_pattern,
            raw_camera_id=raw_camera_id,
            dataset_camera_id=dataset_camera_id,
            frame_idx=current_frame,
            target_hw=pair.shape[1:3],
            allow_missing=args.allow_missing_mask,
        )
        if mask is not None:
            mask = mask.to(device)

        with torch.no_grad():
            pred_tracks, pred_visibility = model(
                video,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
                backward_tracking=args.backward_tracking,
                segm_mask=mask,
            )
        tracks_np = pred_tracks.detach().cpu().numpy()
        visibility_np = pred_visibility.detach().cpu().numpy()
        if tracks_np.ndim != 4 or tracks_np.shape[:2] != (1, 2) or tracks_np.shape[-1] != 2:
            raise ValueError(
                f"unexpected CoTracker track shape for camera {raw_camera_id}, "
                f"frame {current_frame}: {tracks_np.shape}"
            )
        if visibility_np.ndim not in (3, 4) or visibility_np.shape[:2] != (1, 2):
            raise ValueError(
                f"unexpected CoTracker visibility shape for camera {raw_camera_id}, "
                f"frame {current_frame}: {visibility_np.shape}"
            )
        if visibility_np.ndim == 4 and visibility_np.shape[-1] == 1:
            visibility_np = visibility_np[..., 0]
        if visibility_np.ndim != 3 or visibility_np.shape[-1] != tracks_np.shape[-2]:
            raise ValueError("track and visibility point counts do not match")
        if not np.isfinite(tracks_np).all():
            raise ValueError("CoTracker returned non-finite track coordinates")
        tracks_key = f"pred_tracks_{raw_camera_id}_{current_frame}"
        visibility_key = f"pred_visibility_{raw_camera_id}_{current_frame}"
        tracks[tracks_key] = tracks_np.astype(np.float32, copy=False)
        visibility[visibility_key] = visibility_np.astype(bool, copy=False)
        if args.log_every > 0 and (window_idx % args.log_every == 0 or window_idx == len(frames) - 2):
            print(
                f"camera {raw_camera_id}: window {window_idx + 1}/{len(frames) - 1} "
                f"current_frame={current_frame} points={tracks_np.shape[-2]}"
            )

    suffix = "inv" if args.reverse else "forward"
    _save_mapping(
        args.output / f"pred_tracks_{suffix}_{raw_camera_id}.npy",
        tracks,
        args.overwrite,
    )
    _save_mapping(
        args.output / f"pred_visibility_{suffix}_{raw_camera_id}.npy",
        visibility,
        args.overwrite,
    )
    return {
        "raw_camera_id": raw_camera_id,
        "dataset_camera_id": dataset_camera_id,
        "video": str(video_path),
        "selected_source_frames": source_ids,
        "track_file": f"pred_tracks_{suffix}_{raw_camera_id}.npy",
        "visibility_file": f"pred_visibility_{suffix}_{raw_camera_id}.npy",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CoTracker on adjacent frame pairs and save TaoGS raw dictionaries."
    )
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument(
        "--video-pattern",
        default="data{raw_camera_id}.mp4",
        help="relative path pattern; placeholders: {raw_camera_id}, {dataset_camera_id}, {frame_idx}",
    )
    parser.add_argument("--transforms-json", type=Path)
    parser.add_argument("--raw-camera-offset", type=int, default=1)
    parser.add_argument("--raw-camera-st", type=_nonnegative_int)
    parser.add_argument("--raw-camera-ed", type=_positive_int, help="exclusive raw camera bound")
    parser.add_argument(
        "--raw-camera-ids",
        help="comma-separated raw camera ids; overrides --raw-camera-st/--raw-camera-ed",
    )
    parser.add_argument("--frame-st", type=_nonnegative_int, default=0)
    parser.add_argument("--frame-ed", type=_positive_int, default=300, help="exclusive frame bound")
    parser.add_argument("--frame-step", type=_positive_int, default=1)
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument(
        "--reverse",
        dest="reverse",
        action="store_true",
        help="use reverse pairs [t, t-1] (default)",
    )
    direction.add_argument(
        "--forward",
        dest="reverse",
        action="store_false",
        help="use forward pairs [t, t+1]",
    )
    parser.set_defaults(reverse=True)
    parser.add_argument("--resize-scale", type=_positive_float, default=2.0)
    parser.add_argument("--grid-size", type=_positive_int, default=81)
    parser.add_argument("--grid-query-frame", type=_nonnegative_int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cotracker-root", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string")
    parser.add_argument("--offline", action="store_true", help="use the CoTracker offline predictor")
    parser.add_argument("--use-v2-model", action="store_true")
    parser.add_argument("--backward-tracking", action="store_true")
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument(
        "--mask-pattern",
        default="{frame_idx}/mask/{dataset_camera_id}.png",
        help="relative mask pattern under --mask-dir",
    )
    parser.add_argument(
        "--mask-fallback-pattern",
        default="{frame_idx}/masks_person_basketball_undistortion/{dataset_camera_id}.png",
    )
    parser.add_argument("--allow-missing-mask", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=_nonnegative_int, default=25)
    return parser


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    args = build_parser().parse_args()
    if not args.video_dir.is_dir():
        raise FileNotFoundError(args.video_dir)
    if args.frame_ed <= args.frame_st:
        raise ValueError("expected frame-ed > frame-st")
    if args.raw_camera_offset < 0:
        raise ValueError("raw-camera-offset must be non-negative")
    if args.transforms_json is not None and not args.transforms_json.is_file():
        raise FileNotFoundError(args.transforms_json)
    if args.mask_dir is not None and not args.mask_dir.is_dir():
        raise FileNotFoundError(args.mask_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "cotracker_raw_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {manifest_path}; choose a new output directory or pass --overwrite"
        )
    raw_ids = _resolve_camera_ids(args)
    suffix = "inv" if args.reverse else "forward"
    if not args.overwrite:
        existing = [
            args.output / f"pred_{kind}_{suffix}_{raw_camera_id}.npy"
            for raw_camera_id in raw_ids
            for kind in ("tracks", "visibility")
            if (args.output / f"pred_{kind}_{suffix}_{raw_camera_id}.npy").exists()
        ]
        if existing:
            raise FileExistsError(
                "refusing to overwrite raw outputs; first existing file is "
                f"{existing[0]} (choose a new output directory or pass --overwrite)"
            )
    device = _device_from_arg(args.device)
    print(f"using device={device}; cameras={raw_ids}; reverse={args.reverse}")
    model = _load_model(args, device)

    results = []
    for raw_camera_id in raw_ids:
        results.append(_run_camera(args, model, device, raw_camera_id))

    manifest = {
        "format": "taogs-cotracker-raw-v1",
        "video_dir": str(args.video_dir.resolve()),
        "video_pattern": args.video_pattern,
        "transforms_json": str(args.transforms_json.resolve()) if args.transforms_json else None,
        "raw_camera_offset": args.raw_camera_offset,
        "frame_interval": [args.frame_st, args.frame_ed],
        "frame_step": args.frame_step,
        "reverse": args.reverse,
        "resize_scale": args.resize_scale,
        "grid_size": args.grid_size,
        "grid_query_frame": args.grid_query_frame,
        "mask_dir": str(args.mask_dir.resolve()) if args.mask_dir else None,
        "mask_pattern": args.mask_pattern,
        "mask_fallback_pattern": args.mask_fallback_pattern,
        "cameras": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {len(results)} camera raw outputs and {manifest_path}")


if __name__ == "__main__":
    main()
