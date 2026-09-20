#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import json
import torch
from random import randint, shuffle
from utils.loss_utils import l1_loss, fast_ssim
from gaussian_renderer import render_taming
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import (ModelParams, PipelineParams, OptimizationParams,
                       MotionOptimizationParams, LossParamsS1, LossParamsS2, TopoParams)
import numpy as np
import copy
from pathlib import Path
from warp import S2NRaySamples 
from utils.graph_utils import node_graph, dynamicGS_term
from rich.console import Console

from utils.topo_utils import TopoHandler
from utils.corr_init import init_gaussians_with_corr
CONSOLE = Console(width=120)


def resolve_run_root(
    requested_model_path,
    source_path,
    first_frame_init="ply",
):
    """Resolve the shared output root; EDGS requires an explicit path."""
    if requested_model_path:
        return os.path.abspath(requested_model_path)
    if first_frame_init == "edgs":
        raise ValueError(
            "EDGS first-frame runs require an explicit non-empty --model_path/-m"
        )
    unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
    normalized_source = os.path.abspath(source_path)
    sequence_name = normalized_source.split(os.sep)[-1]
    automatic_path = os.path.join(
        "./output/", sequence_name + "_" + str(unique_str)[:2]
    )
    return os.path.abspath(automatic_path)


def prepare_output_and_logger(args, run_manifest=None, manifest_root=None):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/",args.source_path.split('/')[-1] + '_' + unique_str[0:2])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    manifest_path = None
    if run_manifest is not None:
        manifest_root = manifest_root or args.model_path
        os.makedirs(manifest_root, exist_ok=True)
        manifest_path = os.path.join(manifest_root, "run_manifest.json")
        if os.path.exists(manifest_path):
            raise FileExistsError(
                "Refusing to overwrite existing run manifest: %s" % manifest_path
            )
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if manifest_path is not None:
        with open(manifest_path, "x") as manifest_f:
            json.dump(run_manifest, manifest_f, indent=2, sort_keys=True)
            manifest_f.write("\n")

    tb_writer = None
    return tb_writer


def edgs_foreground_mask(scene_masks, cameras):
    """Convert dataset foreground masks to the EDGS convention.

    Dataset masks use non-zero values for foreground.  EDGS uses zero for
    valid pixels, so passing the dataset masks directly would triangulate the
    background instead.  This helper also keeps camera/mask mismatches
    explicit for first-frame initialization.
    """
    if scene_masks is None:
        raise ValueError("EDGS first-frame initialization requires scene masks")
    try:
        masks = np.asarray(scene_masks)
    except Exception as exc:
        raise ValueError("EDGS scene masks must be a rectangular view array") from exc
    if masks.ndim != 3:
        raise ValueError("EDGS scene masks must have shape [views, height, width], got %s" % (masks.shape,))
    if masks.shape[0] != len(cameras):
        raise ValueError(
            "EDGS mask/view mismatch: masks have %d views but cameras have %d"
            % (masks.shape[0], len(cameras))
        )
    if masks.shape[1] < 1 or masks.shape[2] < 1:
        raise ValueError("EDGS scene masks must have positive spatial dimensions")
    if not np.isfinite(masks).all():
        raise ValueError("EDGS scene masks contain non-finite values")
    for view_idx, camera in enumerate(cameras):
        camera_shape = (camera.image_height, camera.image_width)
        if tuple(masks[view_idx].shape) != camera_shape:
            raise ValueError(
                "EDGS mask/camera shape mismatch at view %d: mask=%s camera=%s"
                % (view_idx, tuple(masks[view_idx].shape), camera_shape)
            )

    # Foreground is zero (valid) for corr_init.py; background is one.
    foreground_mask = np.logical_not(masks != 0).astype(np.uint8)
    valid_per_view = (foreground_mask == 0).reshape(foreground_mask.shape[0], -1).sum(axis=1)
    if np.any(valid_per_view == 0):
        empty_views = np.flatnonzero(valid_per_view == 0).tolist()
        raise ValueError("EDGS scene masks contain no foreground pixels in views %s" % empty_views)
    CONSOLE.log(
        "EDGS mask convention: valid=foreground==0; views=%d valid_pixels[min,max]=[%d,%d]"
        % (len(cameras), int(valid_per_view.min()), int(valid_per_view.max()))
    )
    return foreground_mask


def fps_downsample_gaussian_data(data, target_points):
    """Select one synchronized subset of all EDGS Gaussian attributes."""
    target_points = int(target_points)
    if target_points <= 0:
        raise ValueError("EDGS FPS target must be positive, got %d" % target_points)
    xyz = data["new_xyz"]
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("EDGS new_xyz must have shape [N, 3], got %s" % (tuple(xyz.shape),))
    raw_points = int(xyz.shape[0])
    if raw_points < target_points:
        raise ValueError(
            "EDGS produced %d points, fewer than the requested FPS target %d; "
            "increase --edgs_init_matches_per_ref instead of duplicating points"
            % (raw_points, target_points)
        )

    required = (
        "new_xyz", "new_features_dc", "new_features_rest", "new_opacities",
        "new_scaling", "new_rotation",
    )
    for name in required:
        if name not in data:
            raise ValueError("EDGS initialization is missing Gaussian field %s" % name)
        if data[name].shape[0] != raw_points:
            raise ValueError(
                "EDGS %s has %d rows but new_xyz has %d"
                % (name, data[name].shape[0], raw_points)
            )

    if raw_points == target_points:
        indices = np.arange(raw_points, dtype=np.int64)
    else:
        import open3d as o3d
        from scipy.spatial import cKDTree

        xyz_cpu = xyz.detach().cpu().numpy().astype(np.float64, copy=False)
        _, unique_first = np.unique(xyz_cpu, axis=0, return_index=True)
        unique_first.sort()
        if unique_first.size < target_points:
            raise ValueError(
                "EDGS has only %d unique XYZ positions, fewer than FPS target %d"
                % (unique_first.size, target_points)
            )
        unique_xyz = xyz_cpu[unique_first]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(unique_xyz)
        sampled_xyz = np.asarray(
            cloud.farthest_point_down_sample(target_points, start_index=0).points
        )
        distances, local_indices = cKDTree(unique_xyz).query(sampled_xyz, k=1)
        tolerance = max(1.0, float(np.abs(unique_xyz).max())) * 1e-10
        if np.any(distances > tolerance):
            raise RuntimeError(
                "Open3D FPS returned points that could not be matched to EDGS input "
                "(max distance %.3g)" % float(distances.max())
            )
        indices = unique_first[np.asarray(local_indices, dtype=np.int64)]
        if np.unique(indices).size != target_points:
            raise RuntimeError("Open3D FPS did not produce unique source indices")

    torch_indices = torch.as_tensor(indices, dtype=torch.long, device=xyz.device)
    sampled = {name: data[name].index_select(0, torch_indices) for name in required}
    return sampled, indices


def training_motion(dataset, opt, pipe, lossp, testing_iterations, saving_iterations,
              debug_from, is_start_frame, tp, frame_idx = 1, ply_path = None, args = None):

    skip_frame = args.skip_frame
    parallel_load = args.parallel_load
    rest_iters = args.motion_rest_iters
    sparse_view = args.sparse_view
    first_iter = 0
    target_points = int(args.densify_target_points)
    target_tolerance = float(args.densify_target_tolerance)
    target_lower = max(1, int(np.floor(target_points * (1.0 - target_tolerance))))
    target_upper = max(target_lower, int(np.ceil(target_points * (1.0 + target_tolerance))))
    run_manifest = {
        "schema_version": 1,
        "stage": "motion",
        "local_graph": bool(args.local_graph),
        "candidate_filter": "paired_visibility_and_photometric_error",
        "motion_locked_grad_scale": float(args.motion_locked_grad_scale),
        "motion_opt_rgb": True,
        "motion_scaling_threshold_coefficient": float(lossp.scaling_threshold_coefficient),
        "motion_alpha_scaling": float(lossp.alpha_scaling),
        "frame_idx": int(frame_idx),
        "frame_range": {
            "start": int(args.frame_st),
            "end": int(args.frame_ed),
            "step": 1,
        },
        "first_frame_init": str(args.first_frame_init),
        "edgs_fps_target_points": int(args.edgs_fps_target_points),
        "edgs_init_matches_per_ref": args.edgs_init_matches_per_ref,
        "densify": True,
        "densify_target_points": target_points,
        "densify_target_tolerance": target_tolerance,
        "densify_target_range": [target_lower, target_upper],
        "densify_min_opacity": float(args.densify_min_opacity),
        "defer_densify_target_cap": True,
        "final_opacity_cap_iteration": int(opt.densify_until_iter),
        "densify_max_screen_size": int(args.densify_max_screen_size),
        "densify_screen_size_schedule": (
            "disabled when densify_max_screen_size <= 0; otherwise enabled "
            "only when iteration > opacity_reset_interval"
        ),
        "densify_screen_size_activation_iteration": (
            int(opt.opacity_reset_interval) + 1
            if args.densify_max_screen_size > 0
            else None
        ),
        "iterations": int(opt.iterations),
        "test_iterations": [int(value) for value in testing_iterations],
        "save_iterations": list(dict.fromkeys(int(value) for value in saving_iterations)),
        "densify_from_iter": int(opt.densify_from_iter),
        "densify_until_iter": int(opt.densify_until_iter),
        "densification_interval": int(opt.densification_interval),
        "opacity_reset_interval": int(opt.opacity_reset_interval),
        "densify_grad_threshold": float(opt.densify_grad_threshold),
        "flow_path": args.flow_path,
        "dataset_source_path": dataset.source_path,
    }
    tb_writer = prepare_output_and_logger(
        dataset,
        run_manifest=(
            run_manifest
            if is_start_frame and skip_frame < 0 and args.first_frame_init == "edgs"
            else None
        ),
        manifest_root=getattr(args, "run_root", None),
    )
    load_graph = args.load_graph

    gaussians_canonical = GaussianModel(dataset.sh_degree)
    use_edgs_init = is_start_frame and args.first_frame_init == "edgs"
    
    if frame_idx < skip_frame:
        return
    if (skip_frame >= 0 and frame_idx == skip_frame):
        
        gaussians_canonical.load_ply(os.path.join(dataset.model_path,
                                                            "ckt",
                                                            "point_cloud_%d.ply") % (frame_idx), 0.0)
        S2NRaySamples.update_history_gaussians(gaussians_canonical, dataset.model_path, frame_idx)
        S2NRaySamples.update_points_num(gaussians_canonical._xyz.shape[0])
        if frame_idx == skip_frame:
            if load_graph:
                path = os.path.join(dataset.model_path, 'graph_%d.obj' % (frame_idx))
                dynamicGS_term.load_graph_from_file(path)
            else:
                dynamicGS_term.graph_init(gaussians_canonical, os.path.join(dataset.model_path, 'graph_%d.obj' % frame_idx), k=8)

            dynamicGS_term.regular_term_setup(gaussians_canonical)
            if not os.path.exists(os.path.join(dataset.model_path,
                                                        "cpc",
                                                        "point_cloud_%d.ply"% (frame_idx + S2NRaySamples.step_))):
                gaussians_canonical.save_ply(os.path.join(dataset.model_path,
                                                            "cpc",
                                                            "point_cloud_%d.ply"% (frame_idx + S2NRaySamples.step_)))
                
        return 


    scene = Scene(dataset, gaussians_canonical, dynamic = True, load_frame_id = frame_idx, ply_path = ply_path, parallel_load=parallel_load, sparse_view=sparse_view, stage = 1, shuffle=False,
                  load_point_cloud=not use_edgs_init)

    if use_edgs_init:
        edgs_cameras = scene.getTrainCameras().copy()
        edgs_mask = edgs_foreground_mask(gaussians_canonical.masks, edgs_cameras)
        CONSOLE.log("Initializing first motion frame from EDGS correspondences")
        edgs_cfg = copy.copy(tp.cfg)
        if args.edgs_init_matches_per_ref is not None:
            edgs_cfg.matches_per_ref = int(args.edgs_init_matches_per_ref)
            if edgs_cfg.matches_per_ref <= 0:
                raise ValueError("--edgs_init_matches_per_ref must be positive")
        edgs_data = init_gaussians_with_corr(
            gaussians_canonical,
            edgs_cameras,
            edgs_cfg,
            gaussians_canonical._xyz.device,
            roma_model=tp.roma_model,
            mask=edgs_mask,
            return_new_data=True,
        )
        if edgs_data is None or edgs_data["new_xyz"].shape[0] == 0:
            raise RuntimeError("EDGS first-frame initialization produced no valid points")
        raw_count = int(edgs_data["new_xyz"].shape[0])
        gaussians_canonical.initialize_from_data(edgs_data, scene.cameras_extent)
        dense_path = os.path.join(
            dataset.model_path, "edgs_init", "dense_point_cloud_%d.ply" % frame_idx
        )
        if os.path.exists(dense_path):
            raise FileExistsError(
                "Refusing to overwrite existing EDGS initialization artifact: %s"
                % dense_path
            )
        gaussians_canonical.save_ply(dense_path)
        CONSOLE.log(
            "Saved raw EDGS initialization: %s (points=%d)"
            % (dense_path, raw_count)
        )

        fps_target = int(args.edgs_fps_target_points)
        fps_indices = np.arange(raw_count, dtype=np.int64)
        if fps_target > 0:
            CONSOLE.log(
                "Running Open3D farthest-point sampling: %d -> %d points; this CPU step may take several minutes"
                % (raw_count, fps_target)
            )
            edgs_data, fps_indices = fps_downsample_gaussian_data(edgs_data, fps_target)
            gaussians_canonical.initialize_from_data(edgs_data, scene.cameras_extent)

        edgs_init_dir = Path(dataset.model_path) / "edgs_init"
        sampled_path = edgs_init_dir / ("point_cloud_%d.ply" % frame_idx)
        indices_path = edgs_init_dir / ("fps_indices_%d.npy" % frame_idx)
        metadata_path = edgs_init_dir / ("metadata_%d.json" % frame_idx)
        for path in (sampled_path, indices_path, metadata_path):
            if path.exists():
                raise FileExistsError("Refusing to overwrite EDGS FPS artifact: %s" % path)
        gaussians_canonical.save_ply(str(sampled_path))
        np.save(indices_path, fps_indices)
        with metadata_path.open("x") as metadata_file:
            json.dump(
                {
                    "schema_version": 1,
                    "source": "edgs_correspondences",
                    "sampling": "open3d_farthest_point_down_sample",
                    "raw_point_count": raw_count,
                    "sampled_point_count": int(len(fps_indices)),
                    "target_point_count": fps_target,
                    "matches_per_ref": int(edgs_cfg.matches_per_ref),
                    "num_refs": int(edgs_cfg.num_refs),
                    "nns_per_ref": int(edgs_cfg.nns_per_ref),
                },
                metadata_file,
                indent=2,
                sort_keys=True,
            )
            metadata_file.write("\n")
        CONSOLE.log("Saved EDGS FPS initialization: %s (points=%d)" % (sampled_path, len(fps_indices)))

    gaussians_canonical.training_setup_motion(opt, frame_idx=frame_idx)

    if is_start_frame:
            dynamicGS_term.graph_init(gaussians_canonical, os.path.join(dataset.model_path, 'graph_jyh_%d.obj' % frame_idx), k = 8)
            
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    if is_start_frame==False:
        opt.iterations = rest_iters if rest_iters else (opt.iterations // 2)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    cnt = 0
    loss = 0
    CONSOLE.log("gaussians num: ", gaussians_canonical._xyz.shape[0])
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        if args.save_seq and iteration % 1000 == 1:
            gaussians_canonical.save_ply(os.path.join(dataset.model_path,
                                                    "change",
                                                    "point_cloud_%d.ply"% (cnt )))
            cnt += 1

        gaussians_canonical.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians_canonical.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            shuffle(viewpoint_stack)

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_canonical = render_taming(viewpoint_cam, gaussians_canonical, pipe, background)
        image_canonical , viewspace_point_tensor, visibility_filter, radii = render_canonical["render"], render_canonical["viewspace_points"], render_canonical["visibility_filter"], render_canonical["radii"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image_canonical, gt_image)
        if is_start_frame:
            ssim_value = fast_ssim(image_canonical, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        else:
            loss = Ll1
            
        loss_info = {}
        loss_info["rgb_loss"] = loss.item()

        reg_loss, reg_loss_info = dynamicGS_term.compute_loss(gaussians_canonical, lossp, is_start_frame, stage=1)
        loss_info.update(reg_loss_info)

        loss = loss + reg_loss
        loss.backward()

        iter_end.record()

        if iteration == opt.topo_densify_from_iter:
            error_masks = dynamicGS_term.get_photometric_mask(gaussians_canonical, scene.getTrainCameras().copy(), pipe, background, render_func=render_taming)

        with torch.no_grad():
            # Progress bar

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.7f}", **{k: f"{v:.7f}" for k, v in loss_info.items()}})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, gaussians_canonical, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_taming, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            

            gaussians_canonical.max_radii2D[visibility_filter] = torch.max(gaussians_canonical.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians_canonical.add_densification_stats(viewspace_point_tensor, visibility_filter)


            if is_start_frame and iteration >= opt.densify_from_iter and iteration < opt.densify_until_iter and iteration % opt.densification_interval == 0:
                # Keep the released schedule: screen-size pruning is a late
                # criterion, enabled only after opacity reset.  A non-positive
                # CLI value disables it for the complete run.
                max_screen_size = None
                if args.densify_max_screen_size > 0 and iteration > opt.opacity_reset_interval:
                    max_screen_size = args.densify_max_screen_size
                gaussians_canonical.densify_and_prune_to_target(
                    opt.densify_grad_threshold,
                    args.densify_min_opacity,
                    scene.cameras_extent,
                    max_screen_size,
                    args.densify_target_points,
                    args.densify_target_tolerance,
                    force_target_cap=False,
                )
                # The first-frame Laplacian term indexes the rigid graph on
                # every iteration; rebuild it after any count-changing step.
                dynamicGS_term.graph_init(
                    gaussians_canonical,
                    os.path.join(dataset.model_path, "graph_jyh_%d.obj" % frame_idx),
                    k=8,
                )

            if (
                is_start_frame
                and iteration == opt.densify_until_iter
            ):
                gaussians_canonical.prune_to_target_by_opacity(
                    args.densify_target_points
                )
                dynamicGS_term.graph_init(
                    gaussians_canonical,
                    os.path.join(dataset.model_path, "graph_jyh_%d.obj" % frame_idx),
                    k=8,
                )


            topology_prune_start = opt.topo_densify_from_iter + opt.topo_densification_interval
            if is_start_frame==False and iteration > topology_prune_start and iteration % opt.topo_densification_interval == 0:
                size_threshold = 20 
                gaussians_canonical.topo_densify_and_prune(
                    opt.densify_grad_threshold,
                    dynamicGS_term,
                    args.motion_topo_min_opacity,
                    scene.cameras_extent,
                    size_threshold,
                    tp=tp,
                    local_graph=args.local_graph,
                )


            if not is_start_frame and iteration == opt.topo_densify_from_iter and not args.no_topo:
                tp.topo_add(
                    gaussians_canonical,
                    scene.getTrainCameras().copy(),
                    frame_idx,
                    error_masks=error_masks,
                )


            if iteration < opt.iterations:
                if  is_start_frame==False:
                    gaussians_canonical.lock_gradient(
                        lock_opacity=False, lock_scaling=True,
                        grad_scale=args.motion_locked_grad_scale)
                gaussians_canonical.optimizer.step()
                gaussians_canonical.optimizer.zero_grad(set_to_none = True)

    gaussians_canonical.save_ply(os.path.join(dataset.model_path,
                                                        "ckt",
                                                        "point_cloud_%d.ply"% (frame_idx )))
    try:
        aval_indices_np = gaussians_canonical._aval_indices.cpu().numpy()
        print(aval_indices_np.shape, aval_indices_np.max())
        os.makedirs(os.path.join(dataset.model_path, 'aval_indices'), exist_ok=True)
        np.save(os.path.join(dataset.model_path, 'aval_indices', "aval_indices_%d.npy" % frame_idx), aval_indices_np)
    except:
        print("aval_indices not saved")
    
    
    if is_start_frame:
            dynamicGS_term.graph_init(gaussians_canonical, os.path.join(dataset.model_path, 'graph_jyh_%d.obj' % frame_idx), k = 8)
            
   
    dynamicGS_term.regular_term_setup(gaussians_canonical)
    dynamicGS_term.add_velocity_next(gaussians_canonical)
    CONSOLE.log('gaussian_number: %d diff %d :' %(gaussians_canonical.get_xyz.shape[0], gaussians_canonical.get_xyz.shape[0] - gaussians_canonical.last_frame_points_origin))

    gaussians_canonical.save_ply(os.path.join(dataset.model_path,
                                                        "cpc",
                                                        "point_cloud_%d.ply"% (frame_idx + S2NRaySamples.step_)))

    path = os.path.join(dataset.model_path, 'graph_%d.obj' % (frame_idx))
    dynamicGS_term.save_graph_to_file(gaussians_canonical, filename = path)


def next_frame_init(appearance_gaussian, motion_gaussian, dataset, spatial_lr_scale, frame_idx, args):


    warpDQB.skin2JointInterpolation_np(appearance_gaussian._xyz )

    appearance_gaussian.warping(warpDQB)

    motion_gaussian.load_ply(os.path.join(args.motion_folder,
                                    "ckt",
                                    "point_cloud_%d.ply") % warpDQB.next_frame_idx, 0.0)
    
    motion_graph.graph_init(motion_gaussian.get_xyz, os.path.join(dataset.model_path, 'graph_%d.obj' % frame_idx), k = args.k)

    mask = torch.ones_like(motion_gaussian.get_xyz[:, 0], dtype=torch.bool, device="cuda")
    mask[:min(warpDQB.next_number, warpDQB.current_aval_number)] = False
    appearance_gaussian.edge_based_densify(
        motion_gaussian,
        motion_graph,
        mask,
        spatial_lr_scale,
        appearace_graph=appearace_graph,
        random=args.random,
        sanitize_new_points=True,
    )


    appearace_graph.graph_init(appearance_gaussian.get_xyz, k=args.k)
    appearace_graph.regular_term_setup(appearance_gaussian, velocity_option=False)

    appearance_gaussian.save_ply(os.path.join(dataset.model_path,
                                                        "cpc",
                                                        "point_cloud_%d.ply"% (warpDQB.next_frame_idx )))
    CONSOLE.log("Number of points after densify : ", appearance_gaussian._xyz.shape[0])

def init_next_frame_gs(appearance_gaussian, motion_gaussian, appearace_graph, aval_indices_path, k):
    aval_indices = torch.from_numpy(np.load(aval_indices_path))
    total_motion_num = motion_gaussian.get_xyz.shape[0]
    invalid_mask = torch.ones(total_motion_num, dtype=torch.bool)
    invalid_mask[aval_indices] = False
    appearace_graph.prune_father_graph(appearance_gaussian, invalid_mask)
    motion_gaussian.prune_points_no_opt(invalid_mask)
    M = motion_gaussian._xyz.shape[0]
    appearace_graph.father_indices = torch.arange(M * k, device=motion_gaussian._xyz.device).reshape(M, k)
    print(appearace_graph.father_indices.shape)


def training_appearance(dataset, opt, pipe, lossp, testing_iterations, debug_from, is_start_frame, frame_idx = 1, args = None, propagate_to_next=True):
    CONSOLE.log("Training appearance:", frame_idx)
    ply_path = args.ply_path
    cpc_path = args.cpc_path
    skip_frame = args.skip_frame
    parallel_load = args.parallel_load
    rest_iters = args.appearance_rest_iters
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    appearance_gaussian = GaussianModel(dataset.sh_degree)
    motion_gaussian = GaussianModel(dataset.sh_degree)
    if skip_frame >= 0 and frame_idx < skip_frame:
        return

    if propagate_to_next:
        warpDQB.loadMotion(args.motion_folder, frame_idx, aval_indices_path=os.path.join(args.motion_folder,
                                                                                                             'aval_indices',
                                                                                                             f'aval_indices_{str(frame_idx+1)}.npy'))


    if (skip_frame >= 0 and frame_idx == skip_frame):
        appearance_gaussian.load_ply(os.path.join(cpc_path,
                                                            "ckt",
                                                            "point_cloud_%d.ply") % (frame_idx), 0.0)
        motion_gaussian.load_ply(os.path.join(args.motion_folder,
                                "ckt",
                                "point_cloud_%d.ply") % frame_idx, 0.0)
        M = motion_gaussian._xyz.shape[0]
        new_indices = torch.arange(M * args.k, device=motion_gaussian._xyz.device).reshape(M, args.k) # (M, K)
        appearace_graph.father_indices = new_indices
        
        init_next_frame_gs(appearance_gaussian, motion_gaussian, appearace_graph, aval_indices_path=os.path.join(args.motion_folder, 
                                                                                                            'aval_indices', 
                                                                                                            f'aval_indices_{str(frame_idx+1)}.npy'), k=args.k)
        
        
        next_frame_init(appearance_gaussian, motion_gaussian, dataset, 0.0, frame_idx, args)
        appearace_graph.regular_father_term_setup(appearance_gaussian, motion_gaussian, k=args.k, warpDQB=warpDQB, adaptive=args.adaptive_rigid)


        return 


    scene = Scene(dataset, appearance_gaussian, dynamic = True, load_frame_id = frame_idx, ply_path = ply_path, cpc_path = cpc_path, parallel_load=parallel_load, stage = 2)

    if is_start_frame == False and rest_iters:
            opt.position_lr_max_steps = rest_iters


    if is_start_frame:


        appearance_gaussian._xyz = torch.empty(0).cuda()
        appearance_gaussian._features_dc = torch.empty(0).cuda()
        appearance_gaussian._features_rest = torch.empty(0).cuda()
        appearance_gaussian._scaling = torch.empty(0).cuda()
        appearance_gaussian._rotation = torch.empty(0).cuda()
        appearance_gaussian._opacity = torch.empty(0).cuda()

        motion_gaussian.load_ply(os.path.join(args.motion_folder,
                                        "ckt",
                                        "point_cloud_%d.ply") % frame_idx, scene.cameras_extent)
        
        motion_graph.graph_init(motion_gaussian.get_xyz, os.path.join(dataset.model_path, 'graph_%d.obj' % frame_idx), k= args.k)

        mask = torch.ones_like(motion_gaussian.get_xyz[:, 0], dtype=torch.bool, device="cuda")

        
        appearance_gaussian.edge_based_densify(motion_gaussian, motion_graph, mask, scene.cameras_extent, appearace_graph=appearace_graph, random=args.random)

        CONSOLE.log("Number of points after densify : ", appearance_gaussian._xyz.shape[0])
        
        appearace_graph.graph_init(appearance_gaussian.get_xyz, os.path.join(dataset.model_path, 'graph_%d.obj' % frame_idx), k= args.k)

        appearance_gaussian.save_ply(os.path.join(dataset.model_path,
                                                            "ckt",
                                                            "point_cloud_%d.ply"% (frame_idx-1 )))

    appearance_gaussian.training_setup_t2(
        opt,
        frame_idx,
        is_start_frame=is_start_frame,
        stop1000_xyz=args.stop1000_xyz,
        appearance_dc_lr_scale=args.appearance_dc_lr_scale,
    )


    print("number of gaussians:", len(appearance_gaussian.get_xyz))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    if is_start_frame == False:
        opt.iterations = rest_iters if rest_iters else (opt.iterations // 2)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        appearance_gaussian.update_learning_rate(iteration,stop1000_xyz=args.stop1000_xyz)
        if iteration % 1000 == 0:
            appearance_gaussian.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_canonical = render_taming(viewpoint_cam, appearance_gaussian, pipe, background)
        image_canonical , viewspace_point_tensor, visibility_filter, radii = render_canonical["render"], render_canonical["viewspace_points"], render_canonical["visibility_filter"], render_canonical["radii"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image_canonical, gt_image)
        ssim_weight = opt.lambda_dssim
        use_ssim = is_start_frame or args.ssim
        if use_ssim:
            ssim_value = fast_ssim(image_canonical, gt_image)
            loss = (1.0 - ssim_weight) * Ll1 + ssim_weight * (1.0 - ssim_value)
        else:
            loss = Ll1

        loss_info = {}
        loss_info["rgb_loss"] = loss.item()
        reg_loss, reg_loss_info = appearace_graph.compute_loss(appearance_gaussian, lossp, is_start_frame)
        loss_info.update(reg_loss_info)
        loss = loss + reg_loss        
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.7f}", **{k: f"{v:.7f}" for k, v in loss_info.items()}})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()
            
            training_report(tb_writer, appearance_gaussian, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_taming, (pipe, background))


            # first frame Densification
            if is_start_frame and iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                appearance_gaussian.max_radii2D[visibility_filter] = torch.max(appearance_gaussian.max_radii2D[visibility_filter], radii[visibility_filter])
                appearance_gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)


            if iteration < opt.iterations:
                    appearance_gaussian.optimizer.step()
                    appearance_gaussian.optimizer.zero_grad(set_to_none = True)


    appearance_gaussian.save_ply(os.path.join(dataset.model_path,
                                                        "ckt",
                                                        "point_cloud_%d.ply"% (frame_idx )))

    if not propagate_to_next:
        return


    motion_gaussian.load_ply(os.path.join(args.motion_folder,
                                    "ckt",
                                    "point_cloud_%d.ply") % frame_idx, 0.0)
    init_next_frame_gs(appearance_gaussian, motion_gaussian, appearace_graph, aval_indices_path=os.path.join(args.motion_folder, 
                                                                                                         'aval_indices', 
                                                                                                         f'aval_indices_{str(frame_idx+1)}.npy'), k=args.k)

    next_frame_init(appearance_gaussian, motion_gaussian, dataset, scene.cameras_extent, frame_idx, args)
    appearace_graph.regular_father_term_setup(appearance_gaussian, motion_gaussian, k=args.k, warpDQB=warpDQB, adaptive=args.adaptive_rigid)


def training_report(tb_writer, gaussian, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, gaussian, *renderArgs)["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                CONSOLE.log("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="TaoGS two-stage training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    loss_motion = LossParamsS1(parser)
    topo_params = TopoParams(parser)

    parser.add_argument("--stage", choices=["all", "motion", "appearance"], default="all")
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[8_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[16_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--frame_st", type=int, default=0)
    parser.add_argument("--frame_ed", type=int, default=500)
    parser.add_argument("--k", type=int, default=9)
    parser.add_argument("--motion_folder", type=str)
    parser.add_argument("--ply_path", type=str)
    parser.add_argument(
        "--first_frame_init",
        choices=["ply", "edgs"],
        default="edgs",
        help="first motion frame initialization source (default: EDGS)",
    )
    parser.add_argument(
        "--edgs_fps_target_points",
        type=int,
        default=20000,
        help="FPS-sample first-frame EDGS output to this count; 0 keeps all points",
    )
    parser.add_argument(
        "--edgs_init_matches_per_ref",
        type=int,
        default=16000,
        help="first-frame-only EDGS correspondence budget (topology default is unchanged)",
    )
    parser.add_argument("--densify_target_points", type=int, default=20000)
    parser.add_argument("--densify_target_tolerance", type=float, default=0.0)
    parser.add_argument("--densify_min_opacity", type=float, default=0.2)
    parser.add_argument(
        "--densify_max_screen_size",
        type=int,
        default=0,
        help="screen-space radius prune threshold; <=0 disables this criterion",
    )
    parser.add_argument("--cpc_path", type=str)
    parser.add_argument("--skip_frame", type=int, default=-1)
    parser.add_argument("--motion_rest_iters", type=int, default=6_000)
    parser.add_argument("--appearance_rest_iters", type=int, default=10_000)
    parser.add_argument(
        "--appearance_dc_lr_scale",
        type=float,
        default=1.0,
        help="multiply the appearance DC color learning rate",
    )
    parser.add_argument(
        "--motion_percent_dense",
        type=float,
        help="override motion optimizer percent_dense",
    )
    parser.add_argument(
        "--motion_densify_grad_threshold",
        type=float,
        help="override motion optimizer densification gradient threshold",
    )
    parser.add_argument(
        "--motion_topo_min_opacity",
        type=float,
        default=0.1,
        help="minimum opacity used by motion topology pruning (default: 0.1)",
    )
    parser.add_argument("--flow_path", type=str)
    parser.add_argument("--parallel_load", action="store_true")
    parser.add_argument("--sparse_view", action="store_true")
    parser.add_argument("--load_graph", action="store_true")
    parser.add_argument("--no_topo", action="store_true")
    parser.add_argument("--local_graph", action="store_true", default=True,
                        help="default: update candidate two-ring neighbors and edges affected by pruning")
    parser.add_argument("--no_local_graph", dest="local_graph", action="store_false",
                        help="update the complete motion graph")
    parser.add_argument("--motion_locked_grad_scale", type=float, default=0.001,
                        help="gradient multiplier [0,1] for restricted old motion attributes; not an Adam LR multiplier")
    parser.add_argument("--save_seq", action="store_true")
    parser.add_argument("--ssim", action="store_true")
    parser.add_argument("--adaptive_rigid", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--stop1000_xyz", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    if not 0.0 <= args.motion_locked_grad_scale <= 1.0:
        parser.error("--motion_locked_grad_scale must be finite and in [0,1]")
    args.save_iterations.append(args.iterations)
    requested_model_path = args.model_path
    run_root = resolve_run_root(
        requested_model_path,
        args.source_path,
        first_frame_init=args.first_frame_init,
    )
    # The launcher checks the root before creating its tee log.  It passes this
    # marker because that log necessarily exists by the time train.py starts;
    # direct invocations do not get the exemption below.
    launcher_prechecked = os.getenv("TAOGS_RUN_ROOT_PRECHECKED") == "1"
    if (
        args.first_frame_init == "edgs"
        and args.stage in ("all", "motion")
        and args.skip_frame < 0
        and requested_model_path
        and not launcher_prechecked
    ):
        if os.path.exists(run_root):
            nonempty = True
            if os.path.isdir(run_root):
                with os.scandir(run_root) as entries:
                    nonempty = any(entries)
            if nonempty:
                raise FileExistsError(
                    "Refusing to start EDGS run in non-empty or conflicting root: %s"
                    % run_root
                )
    args.run_root = run_root
    motion_root = os.path.join(run_root, "motion", "track")
    appearance_root = os.path.join(run_root, "appearance")
    args.motion_folder = args.motion_folder or motion_root
    args.cpc_path = args.cpc_path or appearance_root

    safe_state(args.quiet)
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    S2NRaySamples.setInterval(args.frame_st, args.frame_ed)
    frame_indices = list(range(args.frame_st, args.frame_ed))

    if args.stage in ("all", "motion"):
        args.model_path = motion_root
        os.makedirs(args.model_path, exist_ok=True)
        topo = TopoHandler(args.flow_path, args.source_path, cfg=topo_params.extract(args))
        motion_opt = MotionOptimizationParams(args.iterations)
        if args.motion_percent_dense is not None:
            motion_opt.percent_dense = args.motion_percent_dense
        if args.motion_densify_grad_threshold is not None:
            motion_opt.densify_grad_threshold = args.motion_densify_grad_threshold
        motion_loss = loss_motion.extract(args)
        is_start_frame = True
        for frame_idx in frame_indices:
            training_motion(
                lp.extract(args), motion_opt, pp.extract(args), motion_loss,
                args.test_iterations, args.save_iterations,
                args.debug_from, is_start_frame, topo, frame_idx,
                ply_path=args.ply_path, args=args)
            is_start_frame = False

    if args.stage in ("all", "appearance"):
        args.model_path = appearance_root
        os.makedirs(args.model_path, exist_ok=True)
        warpDQB = S2NRaySamples()
        motion_graph = node_graph()
        appearace_graph = node_graph()
        appearance_loss = LossParamsS2()
        is_start_frame = True
        for frame_idx in frame_indices:
            training_appearance(
                lp.extract(args), op.extract(args), pp.extract(args), appearance_loss,
                args.test_iterations, args.debug_from, is_start_frame, frame_idx, args=args,
                propagate_to_next=(frame_idx != frame_indices[-1]))
            is_start_frame = False

    print("\nTraining complete.")
