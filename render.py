#!/usr/bin/env python3
"""Render TaoGS frame checkpoints into paired ``render`` and ``gt`` folders.

This entry point deliberately does not construct :class:`scene.Scene`: that
class writes ``cameras.json`` and copies an input PLY into the model folder as
part of training setup.  Rendering should be read-only with respect to a
finished run, so cameras are loaded directly from the dataset reader.

Examples::

    python render.py -s /path/to/sequence -m /path/to/run \
        --stage motion --frame_st 0 --frame_ed 3 \
        --camera_start 25 --camera_end 36 \
        --output_dir /tmp/taogs-render

``frame_ed`` and ``camera_end`` are exclusive.  If ``-m`` is a run root, the
stage-specific directory is selected automatically; a stage directory can
also be passed directly.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional
from utils.sequence_utils import read_sequence_transforms


_NUMERIC_NAME = re.compile(r"^[+-]?\d+$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render TaoGS motion or appearance frame checkpoints."
    )
    parser.add_argument(
        "-s",
        "--source_path",
        type=Path,
        default=None,
        help="dataset root; defaults to source_path in the stage cfg_args",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=Path,
        required=True,
        help="run root or stage root containing ckt/point_cloud_<frame>.ply",
    )
    parser.add_argument(
        "--stage",
        choices=("motion", "appearance"),
        required=True,
        help="checkpoint branch to render",
    )
    parser.add_argument("--frame_st", type=int, default=0, help="inclusive frame start")
    parser.add_argument("--frame_ed", type=int, default=1, help="exclusive frame end")
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument(
        "--camera_ids",
        type=int,
        nargs="+",
        default=None,
        help="explicit numeric camera IDs (mutually exclusive with camera range)",
    )
    parser.add_argument("--camera_start", type=int, default=None, help="inclusive camera ID")
    parser.add_argument("--camera_end", type=int, default=None, help="exclusive camera ID")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="override the default stage_root/ckt directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="new output root; creates render/ and gt/ below it",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=None,
        help="dataset image downsampling factor; defaults to cfg_args",
    )
    background = parser.add_mutually_exclusive_group()
    background.add_argument(
        "--white_background",
        dest="white_background",
        action="store_true",
        help="override cfg_args and use a white background",
    )
    background.add_argument(
        "--black_background",
        dest="white_background",
        action="store_false",
        help="override cfg_args and use a black background",
    )
    parser.set_defaults(white_background=None)
    parser.add_argument(
        "--data_device",
        default=None,
        help="image device passed to the camera loader; defaults to cfg_args",
    )
    return parser


def _read_cfg_args(path: Path) -> dict[str, Any]:
    """Read the simple ``Namespace(...)`` config written by train.py.

    ``eval`` is intentionally not used here.  A run config is still treated
    as untrusted input, and only literal keyword values are accepted.
    """

    try:
        source = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise FileNotFoundError(f"Could not read TaoGS config: {path}") from exc
    try:
        expression = ast.parse(source, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Malformed cfg_args (expected Namespace(...)): {path}") from exc
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError(f"Malformed cfg_args (expected Namespace(...)): {path}")
    if expression.func.id != "Namespace" or expression.args:
        raise ValueError(f"Malformed cfg_args (expected Namespace(...)): {path}")
    values: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError(f"Malformed cfg_args (**kwargs are not supported): {path}")
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise ValueError(
                f"Malformed cfg_args value for '{keyword.arg}' in {path}"
            ) from exc
    return values


def _resolve_stage_root(model_path: Path, stage: str) -> Path:
    model_path = model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not model_path.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_path}")

    candidate = model_path / (Path("motion") / "track" if stage == "motion" else Path("appearance"))
    # A run root has the branch directory; a direct stage root has ckt/.
    if candidate.is_dir():
        return candidate
    return model_path


def _load_render_config(args: argparse.Namespace, stage_root: Path) -> dict[str, Any]:
    cfg_path = stage_root / "cfg_args"
    if not cfg_path.exists():
        # This fallback is useful for a run root whose branch was supplied via
        # an explicit checkpoint_dir, while still producing a clear error if
        # no training config exists anywhere.
        cfg_path = args.model_path.expanduser().resolve() / "cfg_args"
    cfg = _read_cfg_args(cfg_path)
    source_path = args.source_path or cfg.get("source_path")
    if source_path is None or str(source_path).strip() == "":
        raise ValueError(
            "Dataset root is missing: pass --source_path or provide source_path in cfg_args"
        )
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {source_path}")
    read_sequence_transforms(source_path)

    resolution = args.resolution if args.resolution is not None else cfg.get("resolution", 1)
    if not isinstance(resolution, int) or resolution <= 0:
        raise ValueError(f"Resolution must be a positive integer, got {resolution!r}")
    data_device = args.data_device or cfg.get("data_device", "cuda")
    white_background = (
        bool(args.white_background)
        if args.white_background is not None
        else bool(cfg.get("white_background", True))
    )
    return {
        "source_path": source_path,
        "resolution": resolution,
        "data_device": data_device,
        "white_background": white_background,
        "sh_degree": int(cfg.get("sh_degree", 3)),
    }


def _validate_ranges(args: argparse.Namespace) -> range:
    if args.frame_step <= 0:
        raise ValueError(f"frame_step must be positive, got {args.frame_step}")
    if args.frame_ed <= args.frame_st:
        raise ValueError(
            f"frame range must be non-empty and half-open: [{args.frame_st}, {args.frame_ed})"
        )
    if args.camera_ids is not None and (args.camera_start is not None or args.camera_end is not None):
        raise ValueError("camera_ids and camera_start/camera_end are mutually exclusive")
    if (args.camera_start is None) != (args.camera_end is None):
        raise ValueError("camera_start and camera_end must be supplied together")
    if args.camera_start is not None and args.camera_end <= args.camera_start:
        raise ValueError(
            f"camera range must be non-empty and half-open: [{args.camera_start}, {args.camera_end})"
        )
    return range(args.frame_st, args.frame_ed, args.frame_step)


def _numeric_camera_id(view: Any) -> Optional[int]:
    name = Path(str(view.image_name)).stem
    if not _NUMERIC_NAME.fullmatch(name):
        return None
    return int(name)


def _select_views(
    views: Iterable[Any],
    camera_ids: Optional[list[int]],
    camera_start: Optional[int],
    camera_end: Optional[int],
) -> list[tuple[Optional[int], Any]]:
    indexed: list[tuple[Optional[int], Any]] = []
    seen: set[int] = set()
    for view in views:
        camera_id = _numeric_camera_id(view)
        if camera_id is not None:
            if camera_id in seen:
                raise ValueError(f"Duplicate numeric camera ID in dataset: {camera_id}")
            seen.add(camera_id)
        indexed.append((camera_id, view))

    if camera_ids is not None:
        requested = list(dict.fromkeys(camera_ids))
        if any(camera_id is None for camera_id, _ in indexed):
            raise ValueError("Numeric camera selection requires numeric image names")
        missing = [camera_id for camera_id in requested if camera_id not in seen]
        if missing:
            raise ValueError(f"Requested camera IDs are not present in the calibration: {missing}")
        selected_ids = set(requested)
        return [(camera_id, view) for camera_id, view in indexed if camera_id in selected_ids]

    if camera_start is not None:
        if any(camera_id is None for camera_id, _ in indexed):
            raise ValueError("Numeric camera ranges require numeric image names")
        selected = [
            (camera_id, view)
            for camera_id, view in indexed
            if camera_start <= int(camera_id) < int(camera_end)
        ]
        if not selected:
            raise ValueError(
                f"No dataset cameras fall in requested range [{camera_start}, {camera_end})"
            )
        return selected

    return indexed


def _preflight_output(output_dir: Path) -> tuple[Path, Path]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing render output directory: {output_dir}"
        )
    return output_dir / "render", output_dir / "gt"


def _write_manifest(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Write the successful render summary without replacing an old file."""

    manifest_path = output_dir / "manifest.json"
    try:
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing render manifest: {manifest_path}"
        ) from exc
    return manifest_path


def _save_tensor_image(image: Any, path: Path) -> None:
    import numpy as np
    from PIL import Image

    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected one image, got tensor shape {tuple(image.shape)}")
        image = image[0]
    if image.ndim != 3 or image.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
    image = image[:3].detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    image = np.asarray(np.round(image * 255.0), dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def render_sequence(args: argparse.Namespace) -> dict[str, Any]:
    """Render the requested sequence and return a small run summary."""

    frames = _validate_ranges(args)
    stage_root = _resolve_stage_root(args.model_path, args.stage)
    config = _load_render_config(args, stage_root)
    checkpoint_dir = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir is not None
        else stage_root / "ckt"
    )
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    checkpoint_paths = {
        frame_idx: checkpoint_dir / f"point_cloud_{frame_idx}.ply" for frame_idx in frames
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frame checkpoint(s): " + ", ".join(missing))

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else stage_root / "render_eval"
    )
    render_dir, gt_dir = _preflight_output(output_dir)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "TaoGS rendering requires CUDA because the repository rasterizer and "
            "GaussianModel allocate on CUDA"
        )
    from gaussian_renderer import render_taming
    from scene.dataset_readers import readCamerasFromTransforms
    from scene.gaussian_model import GaussianModel
    from utils.camera_utils import cameraList_from_camInfos

    # The CUDA extensions in this repository use the current CUDA device; this
    # also respects CUDA_VISIBLE_DEVICES used by the training launchers.
    torch.cuda.set_device(torch.cuda.current_device())
    render_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    camera_loader_args = SimpleNamespace(
        resolution=config["resolution"],
        data_device=config["data_device"],
    )
    pipeline = SimpleNamespace(
        separate_sh=True,
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    background_values = [1.0, 1.0, 1.0] if config["white_background"] else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")
    stage_number = 1 if args.stage == "motion" else 2
    gaussians = GaussianModel(config["sh_degree"])
    rendered_count = 0
    selected_camera_ids: Optional[list[int]] = None

    try:
        with torch.no_grad():
            for frame_idx, checkpoint_path in checkpoint_paths.items():
                cam_infos = readCamerasFromTransforms(
                    str(config["source_path"]),
                    "transforms.json",
                    config["white_background"],
                    load_frame_id=frame_idx,
                    stage=stage_number,
                )
                views = cameraList_from_camInfos(cam_infos, 1.0, camera_loader_args)
                selected = _select_views(
                    views, args.camera_ids, args.camera_start, args.camera_end
                )
                if not selected:
                    raise ValueError(f"No cameras selected for frame {frame_idx}")
                if selected_camera_ids is None:
                    selected_camera_ids = [camera_id for camera_id, _ in selected if camera_id is not None]

                try:
                    gaussians.load_ply(str(checkpoint_path), 0.0)
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not load {args.stage} checkpoint for frame {frame_idx}: "
                        f"{checkpoint_path}"
                    ) from exc

                for camera_id, view in selected:
                    view_name = str(camera_id) if camera_id is not None else str(view.image_name)
                    filename = f"{frame_idx}_{view_name}.png"
                    render_path = render_dir / filename
                    gt_path = gt_dir / filename
                    if render_path.exists() or gt_path.exists():
                        raise FileExistsError(
                            f"Refusing to overwrite existing rendered pair: {render_path} / {gt_path}"
                        )
                    rendered = render_taming(view, gaussians, pipeline, background)["render"]
                    _save_tensor_image(rendered, render_path)
                    if view.original_image is None:
                        raise RuntimeError(
                            "Camera image is unavailable; GT output requires image loading"
                        )
                    _save_tensor_image(view.original_image[:3], gt_path)
                    rendered_count += 1
                print(
                    f"Rendered {args.stage} frame {frame_idx}: "
                    f"points={gaussians.get_xyz.shape[0]} views={len(selected)}"
                )
    except Exception:
        # Do not hide the original exception.  The partially created output is
        # intentionally retained for diagnosis and will never be reused by a
        # subsequent invocation because output roots are no-overwrite.
        raise

    summary = {
        "stage": args.stage,
        "stage_root": str(stage_root),
        "source_path": str(config["source_path"]),
        "frame_start": args.frame_st,
        "frame_end": args.frame_ed,
        "frame_step": args.frame_step,
        "camera_ids": selected_camera_ids,
        "white_background": config["white_background"],
        "resolution": config["resolution"],
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "render_dir": str(render_dir),
        "gt_dir": str(gt_dir),
        "rendered_images": rendered_count,
    }
    manifest_path = _write_manifest(output_dir, summary)
    print(
        f"Rendering complete: {rendered_count} image pair(s); "
        f"render={render_dir} gt={gt_dir} manifest={manifest_path}"
    )
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    render_sequence(args)
    return 0


if __name__ == "__main__":
    main()
